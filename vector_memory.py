import chromadb
import openai
import uuid
import asyncio
import os
import json
import math
import logging
import time
from datetime import datetime
from error_handler import log_exception, catch_and_log
from dotenv import load_dotenv
from pyignite import Client  # Apache Ignite thin client
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

# Setup logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UltimateMemoryAgent")

load_dotenv()
GLOBAL_CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "IGNITE_HOST": os.getenv("IGNITE_HOST", "127.0.0.1"),
    "IGNITE_PORT": int(os.getenv("IGNITE_PORT", "10800")),
    "IGNITE_CACHE_NAME": os.getenv("IGNITE_CACHE_NAME", "vector_memory_cache"),
    "CHROMA_DB_PATH": os.getenv("CHROMA_DB_PATH", "./chroma_db_store")
}
openai.api_key = GLOBAL_CONFIG["OPENAI_API_KEY"]

# Helper function for JSON serialization
def json_serial(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# --------------------------------------------------
# Define RL Modules
# --------------------------------------------------
class RNDModule(nn.Module):
    """
    Random Network Distillation module.
    The target network is fixed; the predictor learns to match it.
    The prediction error serves as an intrinsic reward.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(RNDModule, self).__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for param in self.target.parameters():
            param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        target_out = self.target(x)
        predictor_out = self.predictor(x)
        return predictor_out, target_out

class PPOAgent(nn.Module):
    """
    A simple actor-critic network to simulate PPO updates.
    In production, replace this with a full-featured RL algorithm.
    """
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state)
        return action, value

class GraphMemoryNetwork:
    """
    Simple graph-based memory network using networkx.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_memory(self, memory_id, content, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.graph.add_node(memory_id, content=content, timestamp=timestamp)
    
    def add_relation(self, from_id, to_id, weight=1.0):
        self.graph.add_edge(from_id, to_id, weight=weight)
    
    def get_related_memories(self, memory_id, top_k=5):
        if memory_id not in self.graph:
            return []
        neighbors = list(self.graph[memory_id].items())
        neighbors.sort(key=lambda x: x[1].get("weight", 1.0), reverse=True)
        return [n for n, attr in neighbors[:top_k]]

# --------------------------------------------------
# Ultimate Memory Agent Class with FAISS Integration (Upgraded)
# --------------------------------------------------
class UltimateMemoryAgent:
    def __init__(self):
        self.logger = logger
        # OpenAI client for embeddings
        self.async_client = openai.AsyncOpenAI(api_key=GLOBAL_CONFIG["OPENAI_API_KEY"])
        # ChromaDB collection for vector storage
        try:
            self.client_chroma = chromadb.PersistentClient(path=GLOBAL_CONFIG["CHROMA_DB_PATH"])
            self.collection = self.client_chroma.get_or_create_collection("cybermutants_collection")
        except Exception as e:
            log_exception(e, "Error initializing ChromaDB client")
            raise
        
        # Apache Ignite caching
        self.ignite_client = Client()
        self.ignite_client.connect(GLOBAL_CONFIG["IGNITE_HOST"], GLOBAL_CONFIG["IGNITE_PORT"])
        self.cache = self.ignite_client.get_or_create_cache(GLOBAL_CONFIG["IGNITE_CACHE_NAME"])
        self.logger.info("Connected to Apache Ignite and obtained cache '%s'.", GLOBAL_CONFIG["IGNITE_CACHE_NAME"])
        
        # Base caching parameters
        self.BASE_CACHE_TTL = 300  # seconds
        self.HIT_THRESHOLD = 10
        self.adaptive_similarity_threshold = 0.75
        
        # RL modules
        self.rnd = RNDModule(input_dim=3072)
        self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=1e-4)
        self.ppo_agent = PPOAgent(state_dim=3072)
        self.ppo_optimizer = optim.Adam(self.ppo_agent.parameters(), lr=1e-4)
        
        # Graph memory network for relational storage
        self.graph_memory = GraphMemoryNetwork()
        
        # Initialize HNSW FAISS index for ANN search
        self.faiss_dim = 3072
        # Using HNSW index for efficient ANN search with inner product similarity.
        self.faiss_index = faiss.IndexHNSWFlat(self.faiss_dim, 32)
        self.faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT
        self.faiss_mapping = {}  # Mapping from FAISS index id to document metadata
        self.faiss_counter = 0

    # -----------------------------
    # Utility Functions
    # -----------------------------
    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0
    
    def adjust_adaptive_thresholds(self, hit_ratio, latency):
        adjustment = 0.0
        if hit_ratio < 0.5:
            adjustment -= 0.05
        if latency > 1.0:
            adjustment -= 0.05
        else:
            adjustment += 0.02
        self.adaptive_similarity_threshold = min(max(self.adaptive_similarity_threshold + adjustment, 0.5), 0.9)
        self.logger.info(f"[Telemetry] Adjusted similarity threshold to {self.adaptive_similarity_threshold:.2f}")
    
    def update_rl_parameters(self, hit_ratio, latency, reward):
        adjustment = 0.01 if reward > 0.5 else -0.01
        self.adaptive_similarity_threshold = min(max(self.adaptive_similarity_threshold + adjustment, 0.5), 0.9)
        self.logger.info(f"[RL] Updated adaptive similarity threshold to {self.adaptive_similarity_threshold:.2f}")
    
    # -----------------------------
    # Apache Ignite Cache Helpers (JSON-based)
    # -----------------------------
    async def ignite_cache_get(self, key):
        raw = await asyncio.to_thread(lambda: self.cache.get(key))
        if raw is not None:
            try:
                return json.loads(raw)
            except Exception as e:
                self.logger.error(f"Failed to load cached JSON for key {key}: {e}")
                return None
        return None
    
    async def ignite_cache_set(self, key, value, expire):
        data = value.copy()
        data["expire_at"] = time.time() + expire
        json_data = json.dumps(data, default=json_serial)
        await asyncio.to_thread(lambda: self.cache.put(key, json_data))
    
    async def ignite_cache_update(self, key, value):
        json_data = json.dumps(value, default=json_serial)
        await asyncio.to_thread(lambda: self.cache.put(key, json_data))
    
    # -----------------------------
    # Embedding and Storage with FAISS Integration
    # -----------------------------
    async def get_modality_embedding(self, input_text, modality="text", model="text-embedding-3-large"):
        # Optionally, if you have additional context (for example, a conversation summary),
        # you could combine it with input_text here.
        if modality == "text":
            resp = await self.async_client.embeddings.create(input=[input_text], model=model)
            embedding = resp.data[0].embedding
        elif modality == "image":
            # For images, plug in your image embedding model; for now, return a dummy vector.
            embedding = [0.1] * self.faiss_dim
        else:
            raise ValueError("Unsupported modality")
        np_embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(np_embedding)
        if norm > 0:
            np_embedding = np_embedding / norm
        return np_embedding.tolist()
    
    async def get_context_aware_embedding(self, input_text, modality="text", model="text-embedding-3-large", additional_context=None):
        """
        Optionally augment the input text with additional context (e.g., from a conversation summary)
        before computing the embedding.
        """
        if additional_context:
            input_text = additional_context + " " + input_text
        return await self.get_modality_embedding(input_text, modality, model)
    
    async def add_embedding(self, agent_id, text, embedding=None, embedding_model="text-embedding-3-large", modality="text"):
        if embedding is None:
            embedding = await self.get_modality_embedding(text, modality=modality, model=embedding_model)
        doc_id = f"{agent_id}_{uuid.uuid4()}"
        # Wrap the synchronous upsert call in to_thread.
        await asyncio.to_thread(
            self.collection.upsert,
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{"agent_id": agent_id}]
        )
        try:
            await asyncio.to_thread(self.client_chroma.persist)
        except Exception:
            pass
        self.logger.info(f"Added embedding for agent {agent_id} with doc_id {doc_id}")
        # Update HNSW FAISS index.
        np_embedding = np.array(embedding, dtype=np.float32)
        np_embedding = np.expand_dims(np_embedding, axis=0)
        self.faiss_index.add(np_embedding)
        # Store timestamp for recency re-ranking.
        self.faiss_mapping[self.faiss_counter] = {"doc_id": doc_id, "text": text, "timestamp": time.time()}
        self.faiss_counter += 1
    
    # -----------------------------
    # Background Cache Refresh
    # -----------------------------
    async def background_refresh_cache(self, cache_key, agent_id, query_text, n_results, similarity_threshold, embedding_model, alternative_embedding_models, modality):
        new_results = await self._query_embedding(agent_id, query_text, n_results, similarity_threshold, embedding_model, alternative_embedding_models, modality, use_ann=True)
        data = {
            "result": new_results,
            "timestamp": time.time(),
            "hits": 0
        }
        await self.ignite_cache_set(cache_key, data, self.BASE_CACHE_TTL)
        self.logger.info(f"Cache refreshed for query: {query_text}")
    
    # -----------------------------
    # Query Embedding & Re-Ranking with HNSW-based ANN
    # -----------------------------
    async def _query_embedding(self, agent_id, query_text, n_results, similarity_threshold, embedding_model, alternative_embedding_models, modality, use_ann=True):
        if alternative_embedding_models:
            embeddings = []
            models = [embedding_model] + alternative_embedding_models
            for model in models:
                emb = await self.get_modality_embedding(query_text, modality=modality, model=model)
                embeddings.append(emb)
            query_embedding_val = [sum(x)/len(x) for x in zip(*embeddings)]
        else:
            query_embedding_val = await self.get_modality_embedding(query_text, modality=modality, model=embedding_model)
        
        if use_ann and self.faiss_index.ntotal > 0:
            np_query = np.array(query_embedding_val, dtype=np.float32)
            norm = np.linalg.norm(np_query)
            if norm > 0:
                np_query = np_query / norm
            np_query = np.expand_dims(np_query, axis=0)
            D, I = self.faiss_index.search(np_query, n_results)
            ann_docs = []
            ann_distances = []
            ann_indices = []
            for idx, score in zip(I[0], D[0]):
                if idx in self.faiss_mapping and score >= similarity_threshold:
                    ann_docs.append(self.faiss_mapping[idx]["text"])
                    ann_distances.append(1 - score)  # converting distance to similarity
                    ann_indices.append(idx)
            self.logger.info(f"HNSW ANN search returned {len(ann_docs)} documents for query: {query_text}")
            results = {"documents": [ann_docs], "distances": [ann_distances], "indices": [ann_indices]}
        else:
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding_val],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            if similarity_threshold is not None:
                filtered_docs = []
                filtered_distances = []
                for doc, distance in zip(results.get("documents", [])[0], results.get("distances", [])[0]):
                    similarity = 1 - distance
                    if similarity >= similarity_threshold:
                        filtered_docs.append(doc)
                        filtered_distances.append(similarity)
                results["documents"] = [filtered_docs]
                results["distances"] = [filtered_distances]
        results = self.rerank_results(results, time.time())
        return results
    
    def rerank_results(self, results, current_time):
        docs = results.get("documents", [])[0]
        distances = results.get("distances", [])[0]
        # Use indices if available; otherwise, create a placeholder list.
        indices = results.get("indices", [])[0] if "indices" in results and results.get("indices") else [None]*len(docs)
        reranked = []
        for doc, distance, idx in zip(docs, distances, indices):
            mapping = self.faiss_mapping.get(idx, {}) if idx is not None else {}
            timestamp = mapping.get("timestamp", current_time)
            # Compute a recency bonus that decays with time difference.
            recency_bonus = 0.05 * math.exp(-0.0001 * (current_time - timestamp))
            # Incorporate graph-based context: if this memory is in our graph, boost by its degree.
            graph_bonus = 0
            doc_id = mapping.get("doc_id")
            if doc_id and doc_id in self.graph_memory.graph:
                degree = self.graph_memory.graph.degree(doc_id)
                graph_bonus = math.log(1 + degree) * 0.02  # weight factor for graph context
            score = distance + recency_bonus + graph_bonus
            reranked.append((doc, score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        sorted_docs = [doc for doc, score in reranked]
        sorted_scores = [score for doc, score in reranked]
        results["documents"] = [sorted_docs]
        results["distances"] = [sorted_scores]
        results.pop("indices", None)
        return results
    
    @catch_and_log("Querying embedding")
    async def query_embedding(self, agent_id, query_text, n_results=3, similarity_threshold=None,
                              embedding_model="text-embedding-3-large", alternative_embedding_models=None, modality="text", use_ann=True):
        similarity_threshold = similarity_threshold or self.adaptive_similarity_threshold
        cache_key = f"{agent_id}_{query_text}_{n_results}_{similarity_threshold}_{embedding_model}_{alternative_embedding_models}_{modality}"
        current_time = time.time()
        cached = await self.ignite_cache_get(cache_key)
        if cached is not None:
            hits = cached.get("hits", 0) + 1
            cached["hits"] = hits
            await self.ignite_cache_update(cache_key, cached)
            age = current_time - cached.get("timestamp", current_time)
            effective_ttl = self.BASE_CACHE_TTL * 2 if hits > self.HIT_THRESHOLD else self.BASE_CACHE_TTL
            self.logger.info(f"[Telemetry] Cache hit for '{query_text}' (hits: {hits}, age: {age:.2f}s)")
            self.adjust_adaptive_thresholds(hit_ratio=hits/20, latency=age)
            query_emb = await self.get_modality_embedding(query_text, modality=modality, model=embedding_model)
            state = torch.tensor(query_emb).float().unsqueeze(0)
            action, value = self.ppo_agent(state)
            self.logger.info(f"[RL] PPO action: {action.detach().numpy()}")
            reward = hits / 20
            self.update_rl_parameters(hit_ratio=hits/20, latency=age, reward=reward)
            if age < effective_ttl and current_time < cached.get("expire_at", current_time + 1):
                return cached["result"]
            else:
                self.logger.info(f"Cache stale for query: {query_text}. Returning stale data and refreshing in background.")
                asyncio.create_task(self.background_refresh_cache(cache_key, agent_id, query_text, n_results, similarity_threshold, embedding_model, alternative_embedding_models, modality))
                return cached["result"]
        results = await self._query_embedding(agent_id, query_text, n_results, similarity_threshold, embedding_model, alternative_embedding_models, modality, use_ann)
        data = {
            "result": results,
            "timestamp": current_time,
            "hits": 1
        }
        await self.ignite_cache_set(cache_key, data, self.BASE_CACHE_TTL)
        self.logger.info(f"Cached new result for query: {query_text}")
        return results
    
    @catch_and_log("Precomputing common query embeddings")
    async def precompute_common_query_embeddings(self, agent_id, common_queries, n_results=3, similarity_threshold=None,
                                                 embedding_model="text-embedding-3-large", alternative_embedding_models=None, modality="text"):
        tasks = []
        for query in common_queries:
            tasks.append(self.query_embedding(agent_id, query, n_results, similarity_threshold, embedding_model, alternative_embedding_models, modality))
        await asyncio.gather(*tasks)
        self.logger.info("Precomputed embeddings for common queries.")
    
    # -----------------------------
    # RND Intrinsic Reward Update
    # -----------------------------
    def update_rnd(self, embedding):
        x = torch.tensor(embedding).float().unsqueeze(0)
        predictor_out, target_out = self.rnd(x)
        loss = ((predictor_out - target_out) ** 2).mean()
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        intrinsic_reward = loss.item()
        self.logger.info(f"[RND] Intrinsic reward: {intrinsic_reward:.4f}")
        return intrinsic_reward
    
    # -----------------------------
    # Graph Memory Storage
    # -----------------------------
    def store_interaction(self, memory_id, content):
        timestamp = time.time()
        self.graph_memory.add_memory(memory_id, content, timestamp)
        self.logger.info(f"Stored interaction {memory_id} in graph memory.")
    
    # -----------------------------
    # Document Re-embedding Routine (Upgraded)
    # -----------------------------
    async def reembed_documents(self, agent_id, embedding_model="text-embedding-3-large", modality="text"):
        docs = await asyncio.to_thread(lambda: self.collection.get())
        if not docs or "documents" not in docs:
            self.logger.info("No documents found for re-embedding.")
            return
        tasks = [self.get_modality_embedding(doc, modality=modality, model=embedding_model) for doc in docs.get("documents", [])]
        new_embeddings = await asyncio.gather(*tasks)
        update_tasks = []
        for doc, new_embedding in zip(docs.get("documents", []), new_embeddings):
            doc_id = doc.get("id")
            if doc_id:
                update_tasks.append(
                    asyncio.to_thread(
                        self.collection.upsert,
                        documents=[doc],
                        embeddings=[new_embedding],
                        ids=[doc_id]
                    )
                )
                self.logger.info(f"Re-embedded document {doc_id}.")
        if update_tasks:
            await asyncio.gather(*update_tasks)
        try:
            await asyncio.to_thread(self.client_chroma.persist)
        except Exception:
            pass
    
    # -----------------------------
    # Full Interaction Cycle
    # -----------------------------
    async def interact(self, agent_id, user_input):
        self.store_interaction(f"{agent_id}_{uuid.uuid4()}", user_input)
        vector_context = await self.query_embedding(agent_id, user_input, n_results=5)
        embedding = await self.get_modality_embedding(user_input)
        intrinsic_reward = self.update_rnd(embedding)
        return vector_context

# --------------------------------------------------
# Main loop for testing the Ultimate Memory Agent
# --------------------------------------------------
if __name__ == "__main__":
    agent = UltimateMemoryAgent()
    
    async def main():
        print("Ultimate Memory Agent is running. Type 'exit' to quit.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
            context = await agent.interact("meta_agent", user_input)
            print("Context:", context)
    
    asyncio.run(main())
