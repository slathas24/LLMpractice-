import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Load config from environment
API_URL = os.getenv("LLM_API_URL") or "https://api.openai.com/v1/chat/completions"
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gpt-4")  # Default to gpt-4 if not set
PROXY = os.getenv("HTTPS_PROXY")  # Optional proxy

# Prepare headers and data
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hello! Can you respond?"}],
    "temperature": 0.3
}

proxies = {"https": PROXY} if PROXY else None

print(f"🔌 Connecting to LLM at: {API_URL}")
if PROXY:
    print(f"🌐 Using proxy: {PROXY}")

try:
    response = requests.post(API_URL, headers=headers, json=payload, proxies=proxies, timeout=15)
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    print("\n✅ Connection successful!\nResponse from LLM:\n")
    print(answer)

except requests.exceptions.SSLError as ssl_err:
    print("\n❌ SSL Error:", ssl_err)
except requests.exceptions.ProxyError as proxy_err:
    print("\n❌ Proxy Error:", proxy_err)
except requests.exceptions.HTTPError as http_err:
    print("\n❌ HTTP Error:", http_err)
    print("Response:", response.text)
except Exception as err:
    print("\n❌ Unexpected Error:", err)
