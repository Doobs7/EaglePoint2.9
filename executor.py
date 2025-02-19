import os
import asyncio
from error_handler import log_exception, catch_and_log

@catch_and_log("Running agent script")
async def run_agent_script(agent_name, base_path="F:/agents"):
    agent_script = os.path.join(base_path, agent_name, f"{agent_name}.py")
    exists = await asyncio.to_thread(os.path.isfile, agent_script)
    if not exists:
        raise FileNotFoundError(f"Agent script {agent_script} not found.")
    process = await asyncio.create_subprocess_exec(
        "python", agent_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')

