import requests

API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
headers = {"Authorization": "Bearer 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"}

response = requests.post(API_URL, headers=headers, json={"inputs": "Hugging Face is cool!"})
print(response.json())