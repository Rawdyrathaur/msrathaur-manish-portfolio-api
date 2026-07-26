import hmac
import hashlib
import json
import requests
import os

WEBHOOK_URL = "http://localhost:8000/webhook/github"
SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "local_test_secret")

def send_webhook(event_type: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    
    # Compute signature
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    
    headers = {
        "X-GitHub-Event": event_type,
        "X-Hub-Signature-256": f"sha256={signature}",
        "Content-Type": "application/json"
    }
    
    print(f"Sending {event_type} webhook to {WEBHOOK_URL}...")
    try:
        resp = requests.post(WEBHOOK_URL, data=body, headers=headers)
        print(f"Response ({resp.status_code}): {resp.json()}")
    except Exception as e:
        print(f"Failed to connect to backend: {e}")

if __name__ == "__main__":
    # Test 1: Ping event
    send_webhook("ping", {"zen": "Keep it logically awesome."})
    
    # Test 2: Repository created event
    mock_repo = {
        "action": "created",
        "repository": {
            "name": "test-repo-webhook",
            "description": "A repository created via webhook test",
            "language": "Python",
            "stargazers_count": 42,
            "topics": ["test", "webhook", "rag"],
            "html_url": "https://github.com/Rawdyrathaur/test-repo-webhook",
            "private": False,
            "updated_at": "2026-07-25T12:00:00Z"
        }
    }
    send_webhook("repository", mock_repo)
    
    # Test 3: Repository deleted event
    mock_repo_delete = {
        "action": "deleted",
        "repository": {
            "name": "test-repo-webhook"
        }
    }
    send_webhook("repository", mock_repo_delete)
