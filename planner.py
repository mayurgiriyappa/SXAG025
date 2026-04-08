import os
import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

def setup_planner():
    # We will read the API key from the environment variables (loaded via dotenv in main.py)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Planner may fail.")
    else:
        genai.configure(api_key=api_key)

def generate_queries(user_query: str, num_queries: int = 4) -> list[str]:
    """
    Generates semantically diverse, orthogonal search queries based on the user's research query.
    """
    setup_planner()
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    You are an expert AI research assistant. The user is researching the following topic: 
    "{user_query}"
    
    Generate exactly {num_queries} highly diverse, orthogonal search queries that would be useful to search in a scholarly database (like Crossref or PubMed).
    Do NOT just paraphrase the user query. Think about different angles, methodologies, related subfields, or applications.
    
    Return the output STRICTLY as a JSON list of strings, and nothing else. No markdown formatting.
    Example output format:
    ["query 1", "query 2", "query 3", "query 4"]
    """
    
    logger.info(f"Generating {num_queries} queries for topic: '{user_query}'")
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        queries = json.loads(response.text)
        if not isinstance(queries, list):
            raise ValueError("Model didn't return a JSON list")
            
        logger.info(f"Generated queries: {queries}")
        return queries
    except Exception as e:
        logger.error(f"Error generating queries: {e}")
        # Fallback to the original query if it fails
        return [user_query]
