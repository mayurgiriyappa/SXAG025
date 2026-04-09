import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def fetch_academic_papers(query: str, max_results: int = 5) -> str:
    """
    Searches Crossref for research papers based on a query.
    Args:
        query: The research topic or title.
        max_results: Number of papers to fetch.
    Returns:
        A formatted string of paper titles, authors, and summaries.
    """
    base_url = os.getenv("CROSSREF_URL")
    url = f"{base_url}/works"
    response = requests.get(
        url,
        params={"query": query, "rows": max_results},
        headers={"User-Agent": "luminous-query-agent/0.1"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text