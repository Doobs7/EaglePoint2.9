import base64
import os
import asyncio
import aiofiles
import aiohttp
import openai

# Set your OpenAI API key from environment variables.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use an asyncio lock to ensure only one vision call runs at a time.
lock = asyncio.Lock()

async def encode_image(image_path: str) -> str:
    """
    Asynchronously reads and encodes an image file to a Base64 string.
    """
    async with aiofiles.open(image_path, "rb") as image_file:
        image_data = await image_file.read()
    return base64.b64encode(image_data).decode("utf-8")

def sync_chat_completion_create(model, messages):
    """
    Synchronous helper for ChatCompletion.create.
    """
    return openai.chat.completions.create(model=model, messages=messages)

def sync_image_create(prompt, n, size):
    """
    Synchronous helper for Image.create.
    """
    return openai.images.generate(prompt=prompt, n=n, size=size)

async def analyze_image(image_path: str, question: str) -> str:
    """
    Asynchronously analyzes an image using GPT-4o-mini.
    
    Reads and encodes the image, constructs a message payload that includes:
      - A text part with the question.
      - An image part with the Base64-encoded image.
    
    Calls the synchronous ChatCompletion.create via asyncio.to_thread and returns the model's response.
    """
    async with lock:
        base64_image = await encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
        response = await asyncio.to_thread(sync_chat_completion_create, "gpt-4o-mini", messages)
        return response.choices[0].message.content.strip()

async def create_image(prompt: str, output_path: str = None) -> dict:
    """
    Asynchronously generates an image based on a text prompt using DALL‑E 3.
    
    This function wraps the synchronous call in asyncio.to_thread. If an output_path is provided,
    it ensures the destination directory exists (creating it if necessary), downloads the generated
    image, and saves it to that location.
    
    Returns a dictionary with keys:
      - "image_url": The URL of the generated image.
      - "local_file": The local file path where the image is saved (if output_path was provided).
    """
    async with lock:
        response = await asyncio.to_thread(sync_image_create, prompt, 1, "1024x1024")
        # Use attribute access to retrieve the URL:
        image_url = response.data[0].url
        result = {"image_url": image_url}
        if output_path:
            # Ensure the directory exists; create if necessary.
            directory = os.path.dirname(output_path)
            await asyncio.to_thread(os.makedirs, directory, exist_ok=True)
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        async with aiofiles.open(output_path, "wb") as f:
                            await f.write(content)
                        result["local_file"] = output_path
                    else:
                        result["local_file"] = None
        return result
