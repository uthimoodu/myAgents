import requests

KEY = "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"

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {KEY}"}
)

data = response.json()
print("Status:", response.status_code)

# Handle both possible structures
if "data" in data:
    for m in data["data"]:
        print(m["id"])
elif "models" in data:
    for m in data["models"]:
        print(m["id"])
else:
    print("Full response:", data)