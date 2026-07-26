import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_linkedin_chunks(knowledge_dir: Path) -> list[dict]:
    """
    Parses a static LinkedIn export JSON (if available).
    Expected schema:
    {
      "profile": {"name": "...", "headline": "...", "summary": "...", "url": "..."},
      "experience": [{"title": "...", "company": "...", "description": "...", "date": "..."}],
      "skills": ["...", "..."]
    }
    """
    export_path = knowledge_dir / "linkedin_export.json"
    if not export_path.exists():
        return []
        
    chunks = []
    try:
        data = json.loads(export_path.read_text(encoding="utf-8"))
        
        # 1. Profile
        profile = data.get("profile", {})
        if profile:
            text = f"LinkedIn Profile: {profile.get('name', '')}\n"
            text += f"Headline: {profile.get('headline', '')}\n"
            text += f"Summary: {profile.get('summary', '')}\n"
            
            chunks.append({
                "id": "linkedin::profile",
                "text": text.strip(),
                "source": "linkedin_export.json",
                "heading": "LinkedIn Profile",
                "title": f"{profile.get('name', 'LinkedIn')} Profile",
                "type": "profile",
                "url": profile.get("url", "/"),
                "source_type": "linkedin",
                "content_type": "profile",
                "visibility": "public",
                "trust_level": "user_provided",
                "timestamp": "",
                "last_updated": ""
            })
            
        # 2. Experience
        experience = data.get("experience", [])
        for i, exp in enumerate(experience):
            text = f"Role: {exp.get('title', '')}\n"
            text += f"Company: {exp.get('company', '')}\n"
            text += f"Date: {exp.get('date', '')}\n"
            text += f"Description: {exp.get('description', '')}\n"
            
            chunks.append({
                "id": f"linkedin::exp::{i}",
                "text": text.strip(),
                "source": "linkedin_export.json",
                "heading": f"Experience: {exp.get('title')} at {exp.get('company')}",
                "title": exp.get("company", "Company"),
                "type": "experience",
                "url": profile.get("url", "/"),
                "source_type": "linkedin",
                "content_type": "experience",
                "visibility": "public",
                "trust_level": "user_provided",
                "timestamp": exp.get("date", ""),
                "last_updated": ""
            })
            
        # 3. Skills
        skills = data.get("skills", [])
        if skills:
            text = f"LinkedIn Skills: {', '.join(skills)}"
            chunks.append({
                "id": "linkedin::skills",
                "text": text.strip(),
                "source": "linkedin_export.json",
                "heading": "LinkedIn Skills",
                "title": "LinkedIn Skills",
                "type": "skill",
                "url": profile.get("url", "/"),
                "source_type": "linkedin",
                "content_type": "skill",
                "visibility": "public",
                "trust_level": "user_provided",
                "timestamp": "",
                "last_updated": ""
            })
            
    except Exception as e:
        logger.warning(f"Failed to parse linkedin_export.json: {e}")
        
    return chunks
