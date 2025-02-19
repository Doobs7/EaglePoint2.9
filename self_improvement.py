import asyncio
import json
import os
import logging
import re
from memory_manager import get_recent_events
from error_handler import log_exception
from openai import AsyncOpenAI
from pythonjsonlogger import jsonlogger

# Configure a structured JSON logger.
logger = logging.getLogger("self_improvement")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Global configuration parameters to be tuned by the self-improvement loop.
CONFIG_PARAMS = {
    "prompt_temperature": 0.7,
    "max_tokens": 1500,
    "retry_limit": 3,
    "context_window": 50  # Number of recent messages to consider for context
}

API_KEY = os.getenv("OPENAI_API_KEY")
async_client = AsyncOpenAI(api_key=API_KEY)

def parse_log_line(line: str) -> dict:
    """
    Parse a log line as JSON. If parsing fails, fallback to regex extraction.
    """
    try:
        data = json.loads(line)
        return data if isinstance(data, dict) else {}
    except Exception:
        # Fallback: try to extract key-value pairs using regex.
        m = re.findall(r'(\w+)=([0-9\.]+)', line)
        return {key: float(value) for key, value in m}

def compute_metrics(recent_logs: str) -> dict:
    """
    Parse recent logs to compute performance metrics from structured JSON log lines.
    """
    lines = recent_logs.splitlines()
    total_response_time = 0.0
    total_contextual_relevance = 0.0
    success_count = 0
    error_count = 0
    valid_entries = 0

    for line in lines:
        if not line.strip():
            continue
        entry = parse_log_line(line)
        if not entry:
            continue
        valid_entries += 1
        # Count errors if level is ERROR or message contains "error".
        if entry.get("level", "").upper() == "ERROR" or "error" in entry.get("message", "").lower():
            error_count += 1
        # Accumulate response time if present.
        if "response_time" in entry:
            total_response_time += float(entry["response_time"])
        # Accumulate task success.
        if "task_success" in entry:
            success_count += 1 if entry["task_success"] is True else 0
        # Accumulate contextual relevance if present.
        if "contextual_relevance" in entry:
            total_contextual_relevance += float(entry["contextual_relevance"])

    avg_response_time = total_response_time / valid_entries if valid_entries > 0 else 0.0
    success_rate = success_count / valid_entries if valid_entries > 0 else 0.0
    avg_contextual_relevance = total_contextual_relevance / valid_entries if valid_entries > 0 else 0.0

    return {
        "error_count": error_count,
        "avg_response_time": avg_response_time,
        "success_rate": success_rate,
        "avg_contextual_relevance": avg_contextual_relevance
    }

def detect_anomaly(metrics: dict) -> bool:
    """
    Trigger anomaly detection if error_count or avg_response_time exceed thresholds.
    """
    if metrics.get("error_count", 0) > 5:
        return True
    if metrics.get("avg_response_time", 0) > 2.0:
        return True
    return False

async def get_adaptive_recommendations(agent_id: str, recent_logs: str, metrics: dict, timeout: int = 177) -> dict:
    """
    Use the LLM to analyze recent logs along with performance metrics and provide JSON recommendations.
    Expected output: a JSON object with keys like "prompt_temperature", "max_tokens", "retry_limit", or "context_window".
    A timeout is applied to avoid hanging on the request.
    """
    prompt = f"""
You are a system optimization engine. Analyze the following recent logs and performance metrics from the meta agent and provide recommendations to adjust system parameters for improved performance and robustness.
Recent Logs:
{recent_logs}

Performance Metrics:
- Error count: {metrics.get('error_count', 'N/A')}
- Average response time: {metrics.get('avg_response_time', 'N/A')}
- Success rate: {metrics.get('success_rate', 'N/A')}
- Average contextual relevance: {metrics.get('avg_contextual_relevance', 'N/A')}

Output only a JSON object with any of the following keys: "prompt_temperature", "max_tokens", "retry_limit", "context_window". Do not include any extra commentary.
"""
    messages = [{"role": "system", "content": prompt}]
    try:
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                reasoning_effort="high"
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error("LLM request timed out for adaptive recommendations.")
        return {}
    except Exception as e:
        log_exception(e, "Error during adaptive recommendation request")
        return {}

    result_text = response.choices[0].message.content.strip()
    try:
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        json_str = json_match.group(0) if json_match else result_text
        recommendations = json.loads(json_str)
        if not isinstance(recommendations, dict):
            recommendations = {}
    except Exception as e:
        logger.error(f"Failed to parse recommendations: {e}. Raw response: {result_text}")
        recommendations = {}
    return recommendations

async def analyze_performance(agent_id: str):
    """
    Analyze recent performance logs and adjust configuration parameters adaptively.
    Combines heuristic adjustments based on performance metrics with LLM-driven recommendations.
    """
    try:
        recent_logs = await get_recent_events(agent_id, limit=10)
        if not recent_logs.strip():
            logger.info("No recent logs available for analysis.")
            return

        metrics = compute_metrics(recent_logs)
        error_count = metrics.get("error_count", 0)
        avg_response_time = metrics.get("avg_response_time", 1.0)

        # Heuristic adjustment: modify prompt_temperature based on error frequency and response time.
        if error_count > 3 or avg_response_time > 1.0:
            CONFIG_PARAMS["prompt_temperature"] = max(0.4, CONFIG_PARAMS["prompt_temperature"] - 0.15)
        else:
            CONFIG_PARAMS["prompt_temperature"] = min(0.9, CONFIG_PARAMS["prompt_temperature"] + 0.05)
        
        if detect_anomaly(metrics):
            logger.warning(f"Anomaly detected for {agent_id}: {metrics}. Triggering aggressive self-improvement measures.")
            CONFIG_PARAMS["retry_limit"] = max(5, CONFIG_PARAMS["retry_limit"] + 1)
        
        recommendations = await get_adaptive_recommendations(agent_id, recent_logs, metrics)
        for key, value in recommendations.items():
            if key in CONFIG_PARAMS and isinstance(value, (int, float)):
                CONFIG_PARAMS[key] = value
        
        logger.info(f"Self-improvement analysis complete for {agent_id}. Updated config: {json.dumps(CONFIG_PARAMS)}")
    except Exception as e:
        log_exception(e, f"Error during self-improvement analysis for {agent_id}")

async def self_improvement_loop(agent_id: str, interval_seconds: int = 300):
    """
    Continuously run self-improvement analysis every interval_seconds.
    """
    while True:
        await analyze_performance(agent_id)
        await asyncio.sleep(interval_seconds)

if __name__ == "__main__":
    agent_id = "meta_agent"
    asyncio.run(self_improvement_loop(agent_id))
