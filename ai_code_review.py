import os
import sys
import requests
from groq import Groq

# ── Config ─────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN")
REPO          = os.getenv("REPO")
PR_NUMBER     = os.getenv("PR_NUMBER")

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not set in secrets.")
    sys.exit(1)

# ── Fetch PR diff from GitHub API ─────────────────────────
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff",
}

diff_url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}"
response = requests.get(diff_url, headers=headers)

if response.status_code != 200:
    print(f"❌ Failed to fetch PR: {response.status_code}")
    sys.exit(1)

pr_data  = response.json()
pr_title = pr_data.get("title", "No title")
pr_body  = pr_data.get("body", "No description")

# ── Fetch diff ─────────────────────────────────────────────
diff_response = requests.get(
    f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}",
    headers={**headers, "Accept": "application/vnd.github.v3.diff"},
)
diff = diff_response.text[:8000]  # limit to 8000 chars to stay within token limits

# ── Ask Groq to review ────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

prompt = f"""You are an expert code reviewer. Review this Pull Request and give constructive feedback.

PR Title: {pr_title}
PR Description: {pr_body}

Code Changes (diff):
{diff}

Please provide:
1. A brief summary of what this PR does
2. Potential bugs or issues
3. Code quality suggestions
4. Security concerns (if any)
5. Overall verdict: APPROVE / REQUEST CHANGES

Keep the review concise and developer-friendly."""

res = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1024,
)

review = res.choices[0].message.content
print("\n🤖 AI Code Review:\n")
print(review)

# ── Post review as PR comment ──────────────────────────────
comment_url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUMBER}/comments"
comment_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
comment_body = f"## 🤖 AI Code Review\n\n{review}"

comment_response = requests.post(
    comment_url,
    headers=comment_headers,
    json={"body": comment_body},
)

if comment_response.status_code == 201:
    print(" Review posted as PR comment!")
else:
    print(f"Could not post comment: {comment_response.status_code}")