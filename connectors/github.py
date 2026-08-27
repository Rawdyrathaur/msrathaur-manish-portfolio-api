import os
import time
import logging
import requests
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

GITHUB_USERNAME = "Rawdyrathaur"
GITHUB_API_URL = "https://api.github.com"
MAX_README_LENGTH = 8000
REQUEST_TIMEOUT = float(os.getenv("SOURCE_REQUEST_TIMEOUT", "12"))
MAX_GITHUB_REPOS = int(os.getenv("MAX_GITHUB_REPOS", "50"))


def _get(url: str, headers: dict) -> requests.Response:
    return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

def get_installation_token() -> str:
    """Generate a GitHub App installation access token using JWT."""
    app_id = os.getenv("GITHUB_APP_ID")
    private_key = os.getenv("GITHUB_PRIVATE_KEY")
    installation_id = os.getenv("GITHUB_INSTALLATION_ID")
    
    if not (app_id and private_key and installation_id):
        token = os.getenv("GITHUB_TOKEN")
        if token:
            logger.info("Using standard GITHUB_TOKEN fallback.")
            return token
        logger.warning("No GitHub credentials found in environment.")
        return ""

    logger.info(f"Generating GitHub App token for app {app_id}, installation {installation_id}...")
    import jwt
    
    if "-----BEGIN" in private_key and "\n" not in private_key:
        private_key = private_key.replace("\\n", "\n")

    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (10 * 60),
        "iss": app_id
    }
    encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")
    
    headers = {
        "Authorization": f"Bearer {encoded_jwt}",
        "Accept": "application/vnd.github.v3+json"
    }
    resp = requests.post(
        f"{GITHUB_API_URL}/app/installations/{installation_id}/access_tokens",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    
    if resp.status_code == 201:
        return resp.json().get("token", "")
    
    logger.error(f"Failed to generate installation token: {resp.status_code} {resp.text}")
    return ""


def get_repo_details(repo_name: str, owner: str, headers: dict) -> dict:
    """Fetch README and language metadata for a specific repository."""
    details = {
        "readme": "",
        "languages": {},
    }
    
    # 1. Fetch README
    readme_resp = _get(f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/readme", headers)
    if readme_resp.status_code == 200:
        readme_data = readme_resp.json()
        content = readme_data.get("content", "")
        encoding = readme_data.get("encoding", "")
        if encoding == "base64" and content:
            try:
                decoded_readme = base64.b64decode(content).decode("utf-8")
                # Truncate if too long to prevent blowing up the chunk size
                details["readme"] = decoded_readme[:MAX_README_LENGTH]
            except Exception as e:
                logger.warning(f"Could not decode README for {repo_name}: {e}")

    # 2. Fetch Languages
    lang_resp = _get(f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/languages", headers)
    if lang_resp.status_code == 200:
        details["languages"] = lang_resp.json()

    return details


_README_INSTRUCTION = re.compile(
    r"(?i)^\s*(?:ignore|disregard|forget|override|reveal|expose|system prompt|developer message|you are|act as|assistant:|system:)"
)
_TECHNOLOGIES = (
    "FastAPI", "Django", "Flask", "Spring Boot", "React", "Vue", "Angular",
    "Node.js", "TypeScript", "JavaScript", "Python", "Java", "Kotlin", "Swift",
    "Docker", "Kubernetes", "Kafka", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "ChromaDB", "Groq", "Gemini", "AWS", "GCP", "Azure", "Unity",
)


def extract_readme_facts(repo: dict, details: dict) -> str:
    """Convert README prose into bounded facts, never a verbatim prompt chunk."""
    readme = details.get("readme", "")
    readme = re.sub(r"```.*?```", " ", readme, flags=re.DOTALL)
    readme = re.sub(r"<!--.*?-->", " ", readme, flags=re.DOTALL)
    readme = re.sub(r"!\[[^]]*]\([^)]+\)", " ", readme)
    readme = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", readme)
    readme = re.sub(r"<[^>]+>", " ", readme)
    lines = [
        line.strip() for line in readme.splitlines()
        if line.strip() and not _README_INSTRUCTION.search(line)
        and "shields.io" not in line and not line.lstrip().startswith("[!")
    ]

    description = (repo.get("description") or "").strip()
    purpose = description
    if not purpose:
        for line in lines:
            if not line.startswith(("#", "-", "*", "+")) and len(line) >= 30:
                purpose = line[:500]
                break

    features = []
    for line in lines:
        if re.match(r"^[-*+]\s+", line):
            feature = re.sub(r"^[-*+]\s+", "", line).strip()
            if 8 <= len(feature) <= 220 and feature not in features:
                features.append(feature)
        if len(features) >= 6:
            break

    readme_lower = readme.lower()
    stack = list(details.get("languages", {}).keys())
    for technology in _TECHNOLOGIES:
        if technology.lower() in readme_lower and technology not in stack:
            stack.append(technology)

    facts = [f"Repository: {repo.get('name', '')}"]
    if purpose:
        facts.append(f"Purpose: {purpose[:500]}")
    if stack:
        facts.append(f"Technology stack: {', '.join(stack[:15])}")
    if features:
        facts.append("Features: " + "; ".join(features))
    topics = [str(topic) for topic in repo.get("topics", []) if topic]
    if topics:
        facts.append("Topics: " + ", ".join(topics[:12]))
    return "\n".join(facts)


def format_repo_chunks(repo: dict, details: dict) -> list[dict]:
    """Format a single GitHub repository into multiple RAG chunks."""
    chunks = []
    repo_name = repo.get('name')
    owner = repo.get('owner', {}).get('login', GITHUB_USERNAME)
    url = repo.get('html_url', '')
    timestamp = repo.get("updated_at", "")
    visibility = "public" if not repo.get('private') else "private"
    
    # Base metadata template
    base_meta = {
        "title": repo_name,
        "type": "repo",
        "url": url,
        "source_type": "github",
        "visibility": visibility,
        "trust_level": "verified",
        "timestamp": timestamp,
        "last_updated": timestamp
    }

    # Chunk 1: Repository Overview (Metadata & Languages)
    overview_text = f"Repository: {repo_name}\n"
    overview_text += f"Description: {repo.get('description', '')}\n"
    overview_text += f"Stars: {repo.get('stargazers_count', 0)}\n"
    overview_text += f"Topics: {', '.join(repo.get('topics', []))}\n"
    
    if details["languages"]:
        lang_str = ", ".join([f"{lang} ({bytes} bytes)" for lang, bytes in details["languages"].items()])
        overview_text += f"Languages Used: {lang_str}\n"

    chunks.append({
        "id": f"github_repo::{repo_name}::overview",
        "text": overview_text.strip(),
        "source": "github_repos",
        "heading": f"{repo_name} - Overview",
        "content_type": "repo_overview",
        **base_meta
    })

    # Chunk 2: Structured facts extracted from untrusted README text.
    readme_facts = extract_readme_facts(repo, details)
    if readme_facts:
        chunks.append({
            "id": f"github_repo::{repo_name}::facts",
            "text": readme_facts,
            "source": "github_repos",
            "heading": f"{repo_name} - Structured facts",
            "content_type": "repo_facts",
            **{**base_meta, "trust_level": "untrusted_external"},
        })

    return chunks


def get_github_chunks() -> list[dict]:
    """
    Fetches GitHub profile and deeply syncs repositories using the Installation Token.
    Returns the formatted RAG chunks.
    """
    try:
        token = get_installation_token()
    except Exception as exc:
        logger.warning("GitHub App authentication failed; using public API: %s", exc)
        token = ""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
        
    chunks = []
    
    try:
        # 1. Fetch user profile
        user_resp = _get(f"{GITHUB_API_URL}/users/{GITHUB_USERNAME}", headers)
        if user_resp.status_code == 200:
            user_data = user_resp.json()
            profile_text = f"GitHub Profile: {user_data.get('name', GITHUB_USERNAME)}\n"
            profile_text += f"Bio: {user_data.get('bio', '')}\n"
            profile_text += f"Followers: {user_data.get('followers', 0)}\n"
            profile_text += f"Public Repos: {user_data.get('public_repos', 0)}\n"
            profile_text += f"Company: {user_data.get('company', '')}\n"
            profile_text += f"Location: {user_data.get('location', '')}\n"
            
            chunks.append({
                "id": f"github_profile::{GITHUB_USERNAME}",
                "text": profile_text.strip(),
                "source": "github_profile",
                "heading": "GitHub Profile",
                "title": f"{GITHUB_USERNAME} Profile",
                "type": "profile",
                "url": user_data.get('html_url', f"https://github.com/{GITHUB_USERNAME}"),
                "source_type": "github",
                "content_type": "profile",
                "visibility": "public",
                "trust_level": "verified",
                "timestamp": user_data.get("updated_at", ""),
                "last_updated": user_data.get("updated_at", "")
            })
            
        # 2. Fetch repos
        # Public visitors must never receive private repository information,
        # even when the configured GitHub App can access it.
        repos_url = f"{GITHUB_API_URL}/users/{GITHUB_USERNAME}/repos?sort=updated&direction=desc&per_page=100"
        repos_resp = _get(repos_url, headers)
        repos_data = repos_resp.json() if repos_resp.status_code == 200 else []

        public_repos = [
            repo for repo in repos_data
            if not repo.get("fork") and not repo.get("private")
        ][:MAX_GITHUB_REPOS]

        def fetch_repo(repo: dict) -> list[dict]:
            repo_name = repo.get("name")
            owner = repo.get("owner", {}).get("login", GITHUB_USERNAME)
            try:
                details = get_repo_details(repo_name, owner, headers)
                return format_repo_chunks(repo, details)
            except requests.RequestException as exc:
                logger.warning("Could not deep-sync repository %s: %s", repo_name, exc)
                return format_repo_chunks(repo, {"readme": "", "languages": {}})

        with ThreadPoolExecutor(max_workers=min(8, max(len(public_repos), 1))) as executor:
            futures = [executor.submit(fetch_repo, repo) for repo in public_repos]
            for future in as_completed(futures):
                chunks.extend(future.result())

    except Exception as e:
        logger.warning(f"Failed to fetch GitHub data: {e}")
        
    return chunks

# If running directly for local deep sync testing:
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    c = get_github_chunks()
    print(f"Generated {len(c)} chunks from GitHub.")
