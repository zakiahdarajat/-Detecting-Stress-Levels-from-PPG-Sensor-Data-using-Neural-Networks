import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Use the correct environment variable name
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

openrouter = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)

try:
    response = openrouter.chat.completions.create(
        model="openai/gpt-4o-mini",  # Updated model name
        messages=[{"role": "user", "content": "what is 2+2?"}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")