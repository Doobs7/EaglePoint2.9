import os
import asyncio
import math
import logging
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.cluster import DBSCAN  # Make sure scikit-learn is installed
from openai import AsyncOpenAI
from error_handler import log_exception, catch_and_log
from db_manager import get_connection, release_connection

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize the LLM client for summarization (o3-mini only)
API_KEY = os.getenv("OPENAI_API_KEY")
llm_client = AsyncOpenAI(api_key=API_KEY)

# --- Adaptive decay parameters ---
adaptive_decay_rate = 0.001
adaptive_critical_decay_rate = 0.0003

def update_decay_parameters(performance_metric):
    """
    Stub for updating decay parameters based on performance feedback.
    """
    global adaptive_decay_rate, adaptive_critical_decay_rate
    if performance_metric < 0.5:
        adaptive_decay_rate *= 0.95
        adaptive_critical_decay_rate *= 0.95
    else:
        adaptive_decay_rate *= 1.05
        adaptive_critical_decay_rate *= 1.05
    logger.info(f"Updated decay rates: normal={adaptive_decay_rate:.6f}, critical={adaptive_critical_decay_rate:.6f}")

def decay_weight(event_timestamp, current_time, event_type, metadata=None, decay_rate=adaptive_decay_rate, critical_decay_rate=adaptive_critical_decay_rate):
    """
    Computes an exponential decay weight.
    Uses slower decay for critical events or those with high 'importance'.
    """
    # Ensure event_timestamp is timezone-aware (assume UTC if naive)
    if event_timestamp.tzinfo is None:
        event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)
    age = (current_time - event_timestamp).total_seconds()
    importance = float(metadata.get("importance", 1)) if metadata and "importance" in metadata else 1
    if event_type.lower() in ["critical", "error"] or importance > 1:
        decay_rate = critical_decay_rate
    return math.exp(-decay_rate * age)

async def get_event_embedding(text):
    """
    Retrieves an embedding for the given text using the o3-mini model.
    """
    resp = await llm_client.embeddings.create(input=[text], model="text-embedding-3-large")
    return resp.data[0].embedding

async def hybrid_cluster_events(rows, current_time, time_eps=300, time_min_samples=2, semantic_eps=0.3, semantic_min_samples=2, decay_rate=adaptive_decay_rate):
    """
    Performs a two-stage (hybrid) clustering:
      1. Time-based clustering using DBSCAN.
      2. Within each time cluster, further semantic clustering using asynchronous batched embedding retrieval.
    """
    clusters = []
    if not rows:
        return clusters

    # Group events by type.
    events_by_type = {}
    for row in rows:
        events_by_type.setdefault(row['event_type'], []).append(row)

    for event_type, events in events_by_type.items():
        events.sort(key=lambda r: r['timestamp'])
        timestamps = np.array([[r['timestamp'].timestamp()] for r in events])
        time_db = DBSCAN(eps=time_eps, min_samples=time_min_samples).fit(timestamps)
        time_labels = time_db.labels_
        for label in set(time_labels):
            cluster_events = [r for r, l in zip(events, time_labels) if l == label]
            if not cluster_events:
                continue
            # For clusters with multiple events, perform semantic clustering.
            if len(cluster_events) > 1:
                embedding_tasks = [get_event_embedding(row['content']) for row in cluster_events]
                embeddings = await asyncio.gather(*embedding_tasks)
                embeddings_np = np.array(embeddings)
                norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                normalized = embeddings_np / (norms + 1e-8)
                sem_db = DBSCAN(eps=semantic_eps, min_samples=semantic_min_samples, metric="euclidean").fit(normalized)
                sem_labels = sem_db.labels_
                for s_label in set(sem_labels):
                    subcluster = [r for r, l in zip(cluster_events, sem_labels) if l == s_label]
                    if subcluster:
                        ts = [r['timestamp'] for r in subcluster]
                        start_time = min(ts)
                        end_time = max(ts)
                        event_lines = []
                        for row in subcluster:
                            meta = json.loads(row.get("metadata", "{}")) if row.get("metadata") else {}
                            weight = decay_weight(row['timestamp'], current_time, row['event_type'], meta, decay_rate)
                            event_lines.append(f"{row['timestamp']} - Weight: {weight:.2f} - {row['content']}")
                        meta = json.loads(subcluster[-1].get("metadata", "{}")) if subcluster[-1].get("metadata") else {}
                        cluster_summary = f"[{event_type} | Layer: {meta.get('memory_layer', 'episodic')}] from {start_time} to {end_time}:\n" + "\n".join(event_lines)
                        clusters.append(cluster_summary)
            else:
                row = cluster_events[0]
                meta = json.loads(row.get("metadata", "{}")) if row.get("metadata") else {}
                weight = decay_weight(row['timestamp'], current_time, row['event_type'], meta, decay_rate)
                clusters.append(f"{row['timestamp']} - Weight: {weight:.2f} - {row['content']}")
    return clusters

@catch_and_log("Initializing consolidation database")
async def init_consolidation_db():
    """
    Creates the consolidated_memory table if it does not exist.
    """
    conn = await get_connection()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_memory (
                id SERIAL PRIMARY KEY,
                agent_id TEXT,
                consolidated_from TIMESTAMP,
                consolidated_to TIMESTAMP,
                summary TEXT,
                context_tags JSONB,
                memory_layer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("Consolidation database initialized.")
    except Exception as e:
        log_exception(e, "Error creating consolidated_memory table")
    finally:
        await release_connection(conn)

@catch_and_log("Consolidating memory")
async def consolidate_memory(agent_id: str, start_time: datetime, end_time: datetime, previous_summary: str = None, memory_layer="episodic") -> str:
    """
    Fetches events, performs hybrid clustering, and summarizes them with the LLM.
    Includes memory layer information to support hierarchical memory.
    """
    conn = await get_connection()
    try:
        # Convert start_time and end_time to naive datetimes to match the DB timestamps.
        start_time_naive = start_time.replace(tzinfo=None)
        end_time_naive = end_time.replace(tzinfo=None)
        rows = await conn.fetch("""
            SELECT timestamp, event_type, content, metadata
            FROM agent_logs
            WHERE agent_id = $1 AND timestamp >= $2 AND timestamp < $3
            ORDER BY timestamp ASC;
        """, agent_id, start_time_naive, end_time_naive)
        
        if not rows:
            logger.info(f"No events to consolidate from {start_time} to {end_time}.")
            return ""
        
        current_time = datetime.now(timezone.utc)
        clustered_events = await hybrid_cluster_events(rows, current_time)
        events_text = "\n\n".join(clustered_events)
        
        system_prompt = (
            "You are an expert summarization engine. Given the following clustered events with context tags, decay weights, "
            "and memory layer information, produce a comprehensive and detailed summary that covers all important aspects, including both recent and historical context. "
            "Incorporate previous context if available and provide a summary 9 to 15 sentances"
        )
        user_prompt = events_text
        if previous_summary:
            user_prompt = f"Previous summary context: {previous_summary}\n\n" + user_prompt
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = await llm_client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            reasoning_effort="medium"
        )
        summary = response.choices[0].message.content.strip()
        
        update_decay_parameters(performance_metric=0.5)
        
        context_tags = json.dumps({"event_count": len(rows)})
        await conn.execute("""
            INSERT INTO consolidated_memory (agent_id, consolidated_from, consolidated_to, summary, context_tags, memory_layer)
            VALUES ($1, $2, $3, $4, $5, $6);
        """, agent_id, start_time_naive, end_time_naive, summary, context_tags, memory_layer)
        logger.info(f"Memory consolidated for agent {agent_id} from {start_time} to {end_time}.")
        return summary
    except Exception as e:
        log_exception(e, "Error during memory consolidation")
        raise
    finally:
        await release_connection(conn)

@catch_and_log("Reindexing memory")
async def reindex_memory(agent_id: str):
    """
    Placeholder for re-computing clusters or updating indices.
    """
    logger.info(f"Reindexing memory for agent {agent_id}.")
    await asyncio.sleep(0)

async def memory_consolidation_loop(agent_id: str, interval_seconds: int = 600):
    """
    Periodically consolidates memory, passing previous summary context and supporting hierarchical memory layers.
    """
    await init_consolidation_db()
    last_consolidated_time = datetime.now(timezone.utc) - timedelta(seconds=interval_seconds)
    previous_summary = None
    iteration = 0
    while True:
        current_time = datetime.now(timezone.utc)
        summary = await consolidate_memory(agent_id, last_consolidated_time, current_time, previous_summary, memory_layer="episodic")
        if summary:
            logger.info(f"Consolidated memory from {last_consolidated_time} to {current_time}:\n{summary}")
            previous_summary = summary
        else:
            logger.info(f"No events to consolidate from {last_consolidated_time} to {current_time}.")
        last_consolidated_time = current_time
        iteration += 1
        if iteration % 5 == 0:
            await reindex_memory(agent_id)
        await asyncio.sleep(interval_seconds)
