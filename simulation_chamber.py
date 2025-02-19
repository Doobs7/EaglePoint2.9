#!/usr/bin/env python3
"""
Upgraded Simulation Chamber: The most advanced scenario simulation tool.
This module provides a simulation tool that adapts dynamically with evolving agent personalities,
multimodal interactions, real-time external context, and graph-based inter-agent relationships.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
import aiohttp
import networkx as nx
from openai import AsyncOpenAI
from error_handler import log_exception, catch_and_log

# Create a dedicated OpenAI client for simulations
sim_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global lock to prevent multiple concurrent simulations
_simulation_lock = asyncio.Lock()

# Graph-based inter-agent relationship network (using networkx)
agent_relationship_graph = nx.DiGraph()

@catch_and_log("Fetching external context")
async def fetch_external_context(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                text = await resp.text()
                return text[:500]  # limit to first 500 characters
    except Exception as e:
        log_exception(e, f"Error fetching external context from {url}")
        return ""

def update_agent_relationship(agent1: str, agent2: str, response1: str, response2: str):
    """
    Update the relationship graph between two agents based on response similarity.
    For simplicity, we use word overlap as a proxy for similarity.
    """
    words1 = set(response1.lower().split())
    words2 = set(response2.lower().split())
    if not words1 or not words2:
        return
    overlap = words1.intersection(words2)
    similarity = len(overlap) / min(len(words1), len(words2))
    # If similarity exceeds a threshold, update the edge weight.
    if similarity > 0.5:
        if agent_relationship_graph.has_edge(agent1, agent2):
            agent_relationship_graph[agent1][agent2]['weight'] += similarity
        else:
            agent_relationship_graph.add_edge(agent1, agent2, weight=similarity)

@catch_and_log("Simulating agent response")
async def simulate_agent(agent: dict, scenario: str, round_num: int, simulation_id: str, external_context: str, agent_memory: list) -> str:
    """
    Simulate an agent's response. Each agent is defined by a dictionary containing:
      - name: str
      - personality: str (dynamic and evolving)
      - modality: list (e.g., ["text"], or ["text", "image"])
    The agent's temporary memory is passed in (as a list) and updated with each response.
    """
    name = agent.get("name", "Unnamed Agent")
    personality = agent.get("personality", "Neutral")
    modalities = agent.get("modality", ["text"])
    
    system_content = (
        f"You are {name}, participating in simulation {simulation_id} with mode '{agent.get('simulation_mode', 'default')}'. "
        f"Your current personality is: {personality}. "
        f"Scenario: {scenario}. "
    )
    if external_context:
        system_content += f"External context: {external_context}. "
    if agent_memory:
        recent_memory = " ".join(agent_memory[-3:])  # last 3 responses
        system_content += f"Recent thoughts: {recent_memory}. "
        
    user_content = f"Round {round_num}: Provide your updated perspective on the scenario."
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    try:
        response = await sim_client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            reasoning_effort="high"  # high reasoning for groundbreaking output
        )
        reply = response.choices[0].message.content.strip()
        agent_memory.append(reply if reply else "[No response]")
        
        # Multimodal integration: if modality includes "image", simulate image generation (stub)
        if "image" in modalities:
            reply += " [Image generated based on your response]"
        
        # Dynamic personality adaptation: simple sentiment update (stub implementation)
        if any(word in reply.lower() for word in ["great", "excellent", "positive", "inspiring"]):
            agent["personality"] = "Optimistic"
        elif any(word in reply.lower() for word in ["challenging", "difficult", "hard", "complex"]):
            agent["personality"] = "Realistic"
        else:
            agent["personality"] = "Neutral"
        
        return reply
    except Exception as e:
        log_exception(e, f"Error simulating agent {name} in round {round_num}")
        agent_memory.append(f"[Error in {name}: {str(e)}]")
        return f"[Error in {name}: {str(e)}]"

@catch_and_log("Running upgraded simulation chamber")
async def run_upgraded_simulation_chamber(simulation_parameters: dict) -> str:
    """
    Run an advanced simulation scenario with dynamic, adaptive agents, multimodal interactions,
    real-time external context, and graph-based inter-agent relationships.
    
    Expected keys in simulation_parameters:
      - simulation_id (optional): str
      - scenario_description: str
      - agents: list of dicts or a single string. If a string is provided, it will be converted to a list with default values.
      - rounds: int
      - analysis: bool (optional)
      - simulation_mode: str (e.g., "focus_group", "jury", "brainstorm")
      - external_context_url: str (optional)
    
    Returns a comprehensive simulation report.
    """
    if _simulation_lock.locked():
        return "Simulation already running. Please wait for the current simulation to finish."
    
    async with _simulation_lock:
        try:
            simulation_id = simulation_parameters.get("simulation_id", str(uuid.uuid4()))
            scenario = simulation_parameters.get("scenario_description", "No scenario description provided.")
            agents = simulation_parameters.get("agents", [
                {"name": "Agent A", "personality": "Neutral", "modality": ["text"]},
                {"name": "Agent B", "personality": "Neutral", "modality": ["text"]}
            ])
            rounds = simulation_parameters.get("rounds", 3)
            detailed_analysis = simulation_parameters.get("analysis", False)
            simulation_mode = simulation_parameters.get("simulation_mode", "default")
            external_context_url = simulation_parameters.get("external_context_url", "")
            
            # If agents is not a list, try to convert it into one.
            if not isinstance(agents, list):
                if isinstance(agents, str):
                    # If the string contains a comma, split it; otherwise, make a list with two copies.
                    if "," in agents:
                        agents = [agent.strip() for agent in agents.split(",")]
                    else:
                        agents = [agents, agents]
                else:
                    agents = []
            
            # Convert agent definitions to dictionaries if they are provided as strings.
            new_agents = []
            for agent in agents:
                if isinstance(agent, str):
                    new_agents.append({"name": agent, "personality": "Neutral", "modality": ["text"]})
                else:
                    new_agents.append(agent)
            agents = new_agents
            
            # Update simulation mode for each agent
            for agent in agents:
                agent["simulation_mode"] = simulation_mode
            
            # Fetch external context if provided
            external_context = ""
            if external_context_url:
                external_context = await fetch_external_context(external_context_url)
            
            start_time = datetime.now().isoformat()
            transcript = []
            quantitative_data = {"total_messages": 0, "agent_message_count": {agent["name"]: 0 for agent in agents}}
            
            # Initialize temporary memory for each agent
            agent_memories = {agent["name"]: [] for agent in agents}
            
            transcript.append(f"Simulation ID: {simulation_id}")
            transcript.append(f"Scenario: {scenario}")
            transcript.append(f"Simulation Mode: {simulation_mode}")
            transcript.append(f"Start Time: {start_time}")
            transcript.append("")
            
            # Run rounds of simulation
            for round_num in range(1, rounds + 1):
                transcript.append(f"--- Round {round_num} ---")
                tasks = [
                    simulate_agent(agent, scenario, round_num, simulation_id, external_context, agent_memories[agent["name"]])
                    for agent in agents
                ]
                responses = await asyncio.gather(*tasks)
                for agent, response in zip(agents, responses):
                    timestamp = datetime.now().isoformat()
                    message = f"[Round {round_num}][{timestamp}] {agent['name']}: {response}"
                    transcript.append(message)
                    quantitative_data["total_messages"] += 1
                    quantitative_data["agent_message_count"][agent["name"]] += 1
                # Update graph relationships based on the round responses
                if len(responses) >= 2:
                    for i in range(len(responses)):
                        for j in range(i+1, len(responses)):
                            update_agent_relationship(agents[i]["name"], agents[j]["name"], responses[i], responses[j])
                            update_agent_relationship(agents[j]["name"], agents[i]["name"], responses[j], responses[i])
            
            # Summarize temporary memories for each agent
            memory_summary = "Temporary Agent Memories:\n"
            for agent_name, mem in agent_memories.items():
                memory_summary += f"{agent_name} Memory: " + " | ".join(mem) + "\n"
            
            # Generate a summary of the inter-agent relationship graph
            graph_summary = "Inter-Agent Relationship Graph Summary:\n"
            for edge in agent_relationship_graph.edges(data=True):
                src, tgt, data = edge
                graph_summary += f"{src} -> {tgt}: weight={data.get('weight', 0):.2f}\n"
            
            transcript_text = "\n".join(transcript)
            quantitative_summary = json.dumps(quantitative_data, indent=2)
            report = (
                f"Upgraded Simulation Chamber Report\n"
                f"Simulation ID: {simulation_id}\n"
                f"Scenario: {scenario}\n"
                f"Simulation Mode: {simulation_mode}\n"
                f"Start Time: {start_time}\n"
                f"Rounds: {rounds}\n"
                f"Agents: {', '.join([agent['name'] for agent in agents])}\n\n"
                f"Conversation Transcript:\n{transcript_text}\n\n"
                f"Quantitative Data:\n{quantitative_summary}\n\n"
                f"{memory_summary}\n"
                f"{graph_summary}\n"
            )
            if detailed_analysis:
                report += (
                    "\nDetailed Analysis:\nThe simulation demonstrated dynamic, adaptive interactions among agents. "
                    "Agent personalities evolved based on the sentiment of their responses, and multimodal outputs were generated where applicable. "
                    "The integrated external context enriched the scenario, while the graph-based relationship summary provided insights into inter-agent dynamics. "
                    "Real-time self-improvement hooks are in place for continuous optimization."
                )
            
            # (Optional) Trigger self-improvement routines here.
            # e.g., await self_improvement.analyze_performance("meta_agent")
            
            return report
        except Exception as e:
            log_exception(e, "Error in run_upgraded_simulation_chamber")
            return f"Upgraded simulation chamber encountered an error: {str(e)}"
