import requests

url = "http://localhost:8000/chat"
headers = {"Content-Type": "application/json"}

queries = [
    "hi",
    "how many projects has he built on GitHub",
    "what is his graduation completed",
    "what does he do actually",
    "what python projects has he built",
    "tell me about your digital brain",
    "list all GitHub repos"
]

for q in queries:
    print(f"\nQ: {q}")
    try:
        resp = requests.post(url, json={"message": q, "history": []}, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            print(f"A: {data.get('answer', 'NO ANSWER')}")
        else:
            print(f"Error: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Connection Error: {e}")
