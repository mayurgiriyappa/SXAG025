import os
import logging
import google.generativeai as genai
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def generate_synthesis(papers: List[Dict[str, Any]], query: str) -> str:
    """
    Consumes clustered papers mapped with raw abstracts to generate a localized Markdown Synthesis
    using configured Gemini models.
    """
    logger.info("Executing Gemini Synthesizer across compiled literature array...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "*Warning: GEMINI_API_KEY missing. Local synthesis skipped.*"
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Bundle text securely for model threshold limits
    context_builder = [f"Base Query Context: {query}"]
    
    for idx, p in enumerate(papers):
        title = p.get('title', 'Unknown Title')
        authors = ", ".join(p.get('authors', [])) or "Unknown"
        year = str(p.get('year', ''))
        cluster = str(p.get('cluster_id', 0))
        abstract = p.get('abstract', '')
        
        context_builder.append(f"\n--- Paper {idx+1} [Cluster {cluster}] ---\nTitle: {title}\nAuthors: {authors} ({year})\nAbstract: {abstract}")
        
    context_str = "\n".join(context_builder)
    
    prompt = f"""
You are the ScholAR Literature Synthesizer. A curated collection of highly diverse research papers surrounding the query "{query}" has been passed to you. These papers have been HDBSCAN clustered into thematic vectors.

Generate a comprehensive Markdown research synthesis report following these strict rules:
1. Provide a "Trend Summary" isolating the primary technological velocity connecting these items.
2. Formulate "Contradiction Flags" analyzing where opposing methodologies or assumptions clash across clusters.
3. Highlight "Frontier Papers" identifying precisely which subset contains the most disruptive parameters.
4. Output strictly formatted GitHub Markdown. Do not enclose the output in block code formatting (```markdown) because it will parse natively.

Raw Corpus Data:
{context_str}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini Synthesis failed during extraction: {e}")
        return "*Synthesis generation failed due to API context constraints or network error.*"
