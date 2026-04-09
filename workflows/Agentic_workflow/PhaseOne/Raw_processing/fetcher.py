import os
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "system@example.com")
CROSSREF_API_URL = os.environ.get("CROSSREF_API_URL", "https://api.crossref.org/works")

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def fetch_papers_for_query(session: aiohttp.ClientSession, query: str, limit: int = 20) -> List[Dict[str, Any]]:
    url = CROSSREF_API_URL
    params = {
        "query": query,
        "rows": limit,
        "mailto": CONTACT_EMAIL
    }
    
    logger.info(f"Fetching papers for query: '{query}'")
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            items = data.get("message", {}).get("items", [])
            extracted = []
            
            for item in items:
                title_list = item.get("title", [])
                title = title_list[0] if title_list else "Unknown Title"
                
                # Abstract might be absent or malformed as JATS XML
                abstract = item.get("abstract", "")
                
                # Extract year from published-print or published-online
                year = None
                pub_date = item.get("published-print") or item.get("published-online")
                if pub_date and "date-parts" in pub_date:
                    year = pub_date["date-parts"][0][0]
                    
                authors = []
                for author in item.get("author", []):
                    fam = author.get("family", "")
                    given = author.get("given", "")
                    authors.append(f"{given} {fam}".strip())
                    
                doi = item.get("DOI", "")
                
                # Extract keywords/subjects
                subjects = item.get("subject", [])
                keywords = ", ".join(subjects) if subjects else ""
                
                # Extract journal_name/container-title
                container_titles = item.get("container-title", [])
                journal_name = container_titles[0] if container_titles else ""
                
                extracted.append({
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors,
                    "doi": doi,
                    "keywords": keywords,
                    "journal_name": journal_name,
                    "query": query  # store source query just in case
                })
                
            return extracted
    except Exception as e:
        logger.error(f"Failed to fetch papers for '{query}': {e}")
        raise # Reraise for tenacity to catch


async def fetch_all_queries(queries: List[str], limit_per_query: int = 25) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches papers for all queries concurrently and returns a mapping from query to its papers.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_papers_for_query(session, q, limit_per_query) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        query_results = {}
        for q, res in zip(queries, results):
            if isinstance(res, Exception):
                logger.error(f"Error fetching for query '{q}': {res}")
                query_results[q] = []
            else:
                query_results[q] = res
                
        return query_results

