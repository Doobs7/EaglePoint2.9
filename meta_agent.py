#!/usr/bin/env python3
"""
Enhanced Meta Agent Orchestrator with Advanced Contextual Awareness, Adaptive Self-Improvement, Vision Capabilities,
and Asynchronous RND Exploration.

This module defines an enhanced Meta Agent that:
  - Retrieves and summarizes past interactions via vector memory.
  - Uses the context summary to inform and shape its responses.
  - Manages child agents, file operations, logging, and vision functions.
  - Maintains background loops for memory consolidation, self-improvement, and asynchronous RND exploration.
  - Runs a scheduled big RND exploration at 3am Pacific time and small idle RND batches if the agent remains idle.
"""

import os
import json
import asyncio
import aiofiles
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo  # For handling Pacific Time
from dotenv import load_dotenv

# Load environment variables from .env (update the path if needed)
load_dotenv()  # or load_dotenv(dotenv_path="path/to/.env")
if os.getenv("OPENAI_API_KEY") is None:
    print("Warning: OPENAI_API_KEY not found. Check your .env file.")

# ANSI escape sequences for terminal colors
USER_COLOR = "\033[32m"   # Green for user text
AGENT_COLOR = "\033[34m"  # Blue for meta agent text
RESET_COLOR = "\033[0m"   # Reset to default

# Core modules and utilities
from error_handler import log_exception
from memory_manager import log_event, get_recent_events, init_db
from vector_memory import UltimateMemoryAgent
from agent_spawner import create_agent_package
from file_tools import write_text_file, read_text_file, list_directory_contents
from executor import run_agent_script
from agent_tasks_writer import write_agent_tasks
# Import the upgraded simulation chamber using an alias for compatibility.
from simulation_chamber import run_upgraded_simulation_chamber as run_simulation_chamber
import memory_consolidator
import self_improvement
from openai import AsyncOpenAI

# Import the agent functions metadata
from agent_functions import AGENT_FUNCTIONS

# Initialize the asynchronous OpenAI client with the API key from the environment
API_KEY = os.getenv("OPENAI_API_KEY")
async_client = AsyncOpenAI(api_key=API_KEY)

# Instantiate the UltimateMemoryAgent
memory_agent = UltimateMemoryAgent()

# Global variable to track the last time a user input was received.
last_user_input_time = datetime.now(timezone.utc)

async def generate_context_summary(user_input: str) -> str:
    """
    Retrieve contextual memory using vector embeddings and generate a comprehensive summary,
    including key points, what you learned, themes, and emotional nuances.
    Provide a detailed summary in 5-7 sentences.
    """
    try:
        vector_context = await memory_agent.query_embedding("meta_agent", user_input, n_results=5)
        if vector_context and "documents" in vector_context and vector_context["documents"]:
            docs = []
            for doc_list in vector_context["documents"]:
                docs.extend(doc_list)
            combined_context = "\n".join(docs)
            prompt = f'''
You are an advanced context summarizer. Analyze the following past interactions and extract all key details, themes, and emotional nuances.
Provide a comprehensive and detailed summary in 9 to 15 sentences.
Past Interactions:
{combined_context}
'''
            messages = [{"role": "system", "content": prompt}]
            response = await async_client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                reasoning_effort="high"
            )
            summary = response.choices[0].message.content.strip()
            return summary
    except Exception as e:
        log_exception(e, "Error generating context summary")
    return ""

async def log_conversation_to_file(conversation: list, filename="conversation_log.json"):
    """
    Log conversation history to a file (for session tracking or debugging).
    """
    try:
        async with aiofiles.open(filename, mode="w") as f:
            await f.write(json.dumps(conversation, indent=2))
    except Exception as e:
        log_exception(e, "Error logging conversation to file")

async def prune_conversation(conversation: list, max_history: int = 100, recent_keep: int = 75) -> list:
    """
    If the conversation length exceeds max_history, summarize the older messages 
    (excluding the initial system message and the last recent_keep messages) and replace them with a summary message.
    """
    if len(conversation) > max_history:
        preserved = [conversation[0]]  # Preserve initial system message
        old_messages = conversation[1:-recent_keep]
        recent_messages = conversation[-recent_keep:]
        combined_text = "\n".join(msg.get("content", "") for msg in old_messages)
        summary = await generate_context_summary(combined_text)
        preserved.append({
            "role": "system",
            "content": f"Conversation summary:\n{summary}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        preserved.extend(recent_messages)
        return preserved
    return conversation

# RND Exploration Functions
async def run_rnd_exploration(agent_id: str, batch_size: int = 10):
    """
    Run an RND exploration cycle by sampling a batch of stored memories,
    computing their embeddings, and updating the RND module.
    """
    print(f"Starting RND exploration with batch size {batch_size}...")
    count = 0
    for mapping in memory_agent.faiss_mapping.values():
        text = mapping.get("text")
        if text:
            embedding = await memory_agent.get_modality_embedding(text, modality="text")
            reward = memory_agent.update_rnd(embedding)
            print(f"Processed memory with RND reward: {reward:.4f}")
            count += 1
            if count >= batch_size:
                break
    print("Completed RND exploration.")

async def idle_monitor(agent_id: str, idle_threshold: int = 900, batch_size: int = 15, check_interval: int = 60):
    """
    Monitors for inactivity. If no user input is detected for 'idle_threshold' seconds,
    trigger a small RND exploration. This repeats continuously while the agent remains idle.
    """
    global last_user_input_time
    while True:
        await asyncio.sleep(check_interval)
        idle_time = (datetime.now(timezone.utc) - last_user_input_time).total_seconds()
        if idle_time > idle_threshold:
            print(f"Agent idle for {idle_time:.0f} seconds, initiating small RND exploration.")
            await run_rnd_exploration(agent_id, batch_size=batch_size)

async def scheduled_big_rnd(agent_id: str, batch_size: int = 1000, scheduled_hour: int = 3):
    """
    Schedules a big RND exploration to run once a day at the specified hour (Pacific Time).
    """
    while True:
        now_pacific = datetime.now(ZoneInfo("America/Los_Angeles"))
        next_run = now_pacific.replace(hour=scheduled_hour, minute=0, second=0, microsecond=0)
        if now_pacific >= next_run:
            next_run += timedelta(days=1)
        wait_seconds = (next_run - now_pacific).total_seconds()
        print(f"Scheduled big RND exploration will run in {wait_seconds/3600:.2f} hours.")
        await asyncio.sleep(wait_seconds)
        print("Initiating scheduled big RND exploration...")
        await run_rnd_exploration(agent_id, batch_size=batch_size)
        await asyncio.sleep(60)

async def meta_agent_chat():
    """
    Main loop for the Enhanced Meta Agent with advanced context awareness, self-improvement, vision capabilities,
    and asynchronous RND exploration.
    """
    agent_id = "meta_agent"
    await init_db()

    # Launch background tasks for memory consolidation, self-improvement, idle monitoring, and scheduled big RND.
    consolidation_task = asyncio.create_task(
        memory_consolidator.memory_consolidation_loop(agent_id, interval_seconds=600)
    )
    improvement_task = asyncio.create_task(
        self_improvement.self_improvement_loop(agent_id, interval_seconds=300)
    )
    idle_monitor_task = asyncio.create_task(idle_monitor(agent_id, idle_threshold=900, batch_size=15))
    scheduled_big_rnd_task = asyncio.create_task(scheduled_big_rnd(agent_id, batch_size=1000, scheduled_hour=3))

    processed_calls = set()
    system_msg = {
        "role": "system",
        "content": (
            "You are the Enhanced Meta Agent, a central orchestrator with advanced contextual awareness, self-improvement, "
            "vision capabilities, and internal RND exploration. You manage child agents, assign tasks, coordinate file operations and logging, "
            "and continuously refine your performance based on feedback and context."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    conversation = [system_msg]

    # Include recent events in context.
    recent_context = await get_recent_events(agent_id, limit=5)
    if recent_context:
        conversation.append({
            "role": "system",
            "content": f"Recent actions:\n{recent_context}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    print(f"{AGENT_COLOR}Enhanced Meta Agent is running. Type 'exit' to quit.{RESET_COLOR}\n")
    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, f"{USER_COLOR}User: {RESET_COLOR}")
            except (EOFError, KeyboardInterrupt):
                print(f"{AGENT_COLOR}Exiting Enhanced Meta Agent.{RESET_COLOR}")
                break

            global last_user_input_time
            last_user_input_time = datetime.now(timezone.utc)

            print(f"{USER_COLOR}User: {user_input}{RESET_COLOR}")

            if user_input.lower() in ["exit", "quit"]:
                break

            message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            conversation.append(message)
            await log_event(agent_id, "user_message", user_input)

            # Enhanced contextual awareness: generate a context summary based on vector memory.
            context_summary = await generate_context_summary(user_input)
            if context_summary:
                conversation.append({
                    "role": "system",
                    "content": f"Enhanced Context Summary:\n{context_summary}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            # Also include additional vector context if available.
            vector_context = await memory_agent.query_embedding(agent_id, user_input, n_results=5)
            if vector_context and "documents" in vector_context and vector_context["documents"]:
                context_str = "\n".join(vector_context["documents"][0])
                conversation.append({
                    "role": "system",
                    "content": f"Additional relevant past interactions:\n{context_str}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            # Prune conversation history if it grows too long.
            conversation = await prune_conversation(conversation)

            while True:
                try:
                    response = await asyncio.wait_for(
                        async_client.chat.completions.create(
                            model="o3-mini",
                            messages=conversation,
                            functions=AGENT_FUNCTIONS,
                            function_call="auto",
                            reasoning_effort="medium"
                        ),
                        timeout=60
                    )
                except asyncio.TimeoutError:
                    error_msg = "LLM request timed out after 60 seconds."
                    conversation.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    print(f"\n{AGENT_COLOR}Enhanced Meta Agent: {error_msg}{RESET_COLOR}")
                    await log_event(agent_id, "error", error_msg)
                    break

                message_obj = response.choices[0].message

                if message_obj.function_call is not None:
                    call_signature = f"{message_obj.function_call.name}-{message_obj.function_call.arguments}"
                    if call_signature in processed_calls:
                        break
                    processed_calls.add(call_signature)

                    try:
                        args = json.loads(message_obj.function_call.arguments)
                    except Exception as e:
                        error_msg = f"Error: invalid JSON for function arguments: {message_obj.function_call.arguments}"
                        conversation.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        print(f"\n{AGENT_COLOR}Enhanced Meta Agent: {error_msg}{RESET_COLOR}")
                        await log_event(agent_id, "error", error_msg, {"raw_args": message_obj.function_call.arguments})
                        break

                    if message_obj.function_call.name == "create_agent_package":
                        agent_name = args["agent_name"]
                        base_path = args.get("base_path", "F:/agents")
                        path_created = await create_agent_package(agent_name, base_path)
                        tool_result = f"Created agent '{agent_name}' at: {path_created}"

                    elif message_obj.function_call.name == "write_text_file":
                        filepath = args["filepath"]
                        content = args["content"]
                        await write_text_file(filepath, content)
                        tool_result = f"Successfully wrote text to: {filepath}"

                    elif message_obj.function_call.name == "read_text_file":
                        filepath = args["filepath"]
                        try:
                            file_content = await read_text_file(filepath)
                            tool_result = f"File Content of {filepath}:\n{file_content}"
                        except Exception as e:
                            tool_result = f"Error reading file '{filepath}': {str(e)}"

                    elif message_obj.function_call.name == "run_agent_script":
                        agent_name = args["agent_name"]
                        base_path = args.get("base_path", "F:/agents")
                        try:
                            code, stdout, stderr = await run_agent_script(agent_name, base_path)
                            if code == 0:
                                tool_result = (
                                    f"Agent '{agent_name}' executed successfully.\n"
                                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                                )
                            else:
                                tool_result = (
                                    f"Agent '{agent_name}' exited with code {code}.\n"
                                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                                )
                        except Exception as e:
                            tool_result = f"Error executing agent script: {e}"

                    elif message_obj.function_call.name == "write_agent_tasks":
                        agent_name = args["agent_name"]
                        tasks = args["tasks"]
                        base_path = args.get("base_path", "F:/agents")
                        await write_agent_tasks(agent_name, tasks, base_path)
                        try:
                            code, stdout, stderr = await run_agent_script(agent_name, base_path)
                            if code == 0:
                                tool_result = (
                                    f"Agent '{agent_name}' executed successfully.\n"
                                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                                )
                            else:
                                tool_result = (
                                    f"Agent '{agent_name}' exited with code {code}.\n"
                                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                                )
                        except Exception as e:
                            tool_result = f"Error executing agent script: {e}"

                    elif message_obj.function_call.name == "list_directory_contents":
                        directory_path = args["directory_path"]
                        tool_result = await list_directory_contents(directory_path)

                    elif message_obj.function_call.name == "run_simulation_chamber":
                        sim_params = args["simulation_parameters"]
                        tool_result = await run_simulation_chamber(sim_params)

                    elif message_obj.function_call.name == "analyze_image":
                        image_path = args["image_path"]
                        question = args["question"]
                        try:
                            from dalle3_vision import analyze_image
                            tool_result = await analyze_image(image_path, question)
                        except Exception as e:
                            tool_result = f"Error analyzing image: {str(e)}"

                    elif message_obj.function_call.name == "create_image":
                        prompt = args["prompt"]
                        output_path = args.get("output_path")
                        try:
                            from dalle3_vision import create_image
                            tool_result = await create_image(prompt, output_path)
                        except Exception as e:
                            tool_result = f"Error generating image: {str(e)}"

                    else:
                        tool_result = f"Function '{message_obj.function_call.name}' not recognized."

                    tool_result_str = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result

                    conversation.append({
                        "role": "assistant",
                        "content": tool_result_str,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    print(f"\n{AGENT_COLOR}Enhanced Meta Agent (tool result): {tool_result_str}{RESET_COLOR}")

                    await log_event(agent_id, "function_call", tool_result_str, {
                        "function": message_obj.function_call.name,
                        "args": args
                    })
                    await memory_agent.add_embedding(agent_id, tool_result_str)
                    continue  # Process further function calls if present
                else:
                    assistant_reply = message_obj.content or ""
                    conversation.append({
                        "role": "assistant",
                        "content": assistant_reply,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    print(f"\n{AGENT_COLOR}Enhanced Meta Agent: {assistant_reply}{RESET_COLOR}")
                    await log_event(agent_id, "assistant_reply", assistant_reply)
                    await memory_agent.add_embedding(agent_id, assistant_reply)
                    break

                processed_calls.clear()
            processed_calls.clear()
            await log_conversation_to_file(conversation)
    finally:
        consolidation_task.cancel()
        improvement_task.cancel()
        idle_monitor_task.cancel()
        scheduled_big_rnd_task.cancel()

if __name__ == "__main__":
    asyncio.run(meta_agent_chat())
