import os
import time
import logging
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

GITHUB_USERNAME = "Rawdyrathaur"
GITHUB_API_URL = "https://api.github.com"
MAX_FILE_TREE_ITEMS = 50
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
    """Fetch README, languages, and file tree for a specific repository."""
    details = {
        "readme": "",
        "languages": {},
        "tree": []
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

    # 3. Fetch File Tree (Default Branch)
    # First get the default branch name (usually main or master)
    repo_info_resp = _get(f"{GITHUB_API_URL}/repos/{owner}/{repo_name}", headers)
    if repo_info_resp.status_code == 200:
        default_branch = repo_info_resp.json().get("default_branch", "main")
        tree_resp = _get(f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/git/trees/{default_branch}?recursive=1", headers)
        if tree_resp.status_code == 200:
            tree_data = tree_resp.json().get("tree", [])
            # Filter out some common noise like .git, node_modules, etc.
            filtered_tree = [
                item["path"] for item in tree_data 
                if item["type"] == "blob" and not any(ignored in item["path"] for ignored in ["node_modules", ".git", "dist", "build", "venv", "__pycache__"])
            ]
            # Truncate the list if it's massive
            details["tree"] = filtered_tree[:MAX_FILE_TREE_ITEMS]

    return details


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

    # Chunk 2: README
    if details["readme"]:
        # If README is very long, we should technically chunk it further, but for now we truncated it.
        chunks.append({
            "id": f"github_repo::{repo_name}::readme",
            "text": f"README for {repo_name}:\n\n{details['readme']}",
            "source": "github_repos",
            "heading": f"{repo_name} - README",
            "content_type": "repo_readme",
            **base_meta
        })

    # Chunk 3: File Tree
    if details["tree"]:
        tree_text = f"File Structure for {repo_name}:\n"
        tree_text += "\n".join(details["tree"])
        if len(details["tree"]) == MAX_FILE_TREE_ITEMS:
            tree_text += "\n... (truncated)"
            
        chunks.append({
            "id": f"github_repo::{repo_name}::tree",
            "text": tree_text.strip(),
            "source": "github_repos",
            "heading": f"{repo_name} - File Tree",
            "content_type": "repo_tree",
            **base_meta
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
                return format_repo_chunks(repo, {"readme": "", "languages": {}, "tree": []})

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
