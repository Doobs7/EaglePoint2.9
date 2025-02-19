import os
import asyncio
import aiofiles
from error_handler import log_exception, catch_and_log

@catch_and_log("Ensuring directory exists")
async def ensure_directory(path):
    if not path:
        return
    if not os.path.exists(path):
        await asyncio.to_thread(os.makedirs, path, exist_ok=True)

@catch_and_log("Writing text file")
async def write_text_file(filepath, content):
    await ensure_directory(os.path.dirname(filepath))
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(content)

@catch_and_log("Reading text file")
async def read_text_file(filepath):
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        return await f.read()

@catch_and_log("Listing directory contents")
async def list_directory_contents(directory_path):
    if not os.path.isdir(directory_path):
        return f"Path not found or not a directory: {directory_path}"
    items = await asyncio.to_thread(os.listdir, directory_path)
    if not items:
        return f"No files or folders in: {directory_path}"
    lines = [f"Contents of {directory_path}:"]
    for item in items:
        full_path = os.path.join(directory_path, item)
        if os.path.isdir(full_path):
            lines.append(f" [DIR]  {item}")
        else:
            lines.append(f" [FILE] {item}")
    return "\n".join(lines)
