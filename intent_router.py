import os
import json
import logging
from typing import Literal

from groq import Groq

logger = logging.getLogger(__name__)

IntentType = Literal["GREETING", "OFF_TOPIC", "PORTFOLIO_QUERY", "FOLLOW_UP"]

# Initialize Groq client lazy or eager
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            _client = Groq(api_key=api_key)
    return _client

SYSTEM_PROMPT = """
You are an intent classifier for a professional portfolio chatbot.
Categorize the user's message into exactly one of these four intents:
1. GREETING: Basic hellos, hi, who are you, how are you.
2. PORTFOLIO_QUERY: Questions about Manish's projects, github, skills, experience, tech stack, or resume.
3. FOLLOW_UP: A continuation of a previous portfolio question (e.g., "what tech does it use?", "how does that work?", "tell me more").
4. OFF_TOPIC: Anything else. Cooking, politics, general coding help, poems, unrelated trivia.

Respond ONLY with a valid JSON object containing exactly one key "intent" with the chosen category as the string value.
Example: {"intent": "PORTFOLIO_QUERY"}
"""

def classify_intent(message: str, history: list = None) -> IntentType:
    """Classifies the user intent using a lightning-fast Groq model."""
    client = get_client()
    if not client:
        # Fallback if no API key is somehow bypassed
        return "PORTFOLIO_QUERY"
        
    try:
        # We pass a brief summary of history if available to help detect FOLLOW_UP
        history_context = ""
        if history and len(history) > 0:
            last_msg = history[-1].content[:100]
            history_context = f"Previous Assistant Message: {last_msg}\n\n"
            
        prompt = f"{history_context}User Message: {message}"
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Fast, lightweight model for routing
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=32
        )
        
        result = json.loads(response.choices[0].message.content)
        intent = result.get("intent", "PORTFOLIO_QUERY")
        
        if intent not in ["GREETING", "OFF_TOPIC", "PORTFOLIO_QUERY", "FOLLOW_UP"]:
            intent = "PORTFOLIO_QUERY"
            
        logger.info(f"[ROUTER] Classified intent: {intent}")
        return intent
        
    except Exception as e:
        logger.error(f"[ROUTER] Intent classification failed: {e}")
        # Default to full RAG pipeline on error
        return "PORTFOLIO_QUERY"
