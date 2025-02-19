import sys
import asyncio
from dalle3_vision import analyze_image

async def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_image_tool.py <image_path> <question>")
        return
    image_path = sys.argv[1]
    question = sys.argv[2]
    try:
        result = await analyze_image(image_path, question)
        print(result)
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
