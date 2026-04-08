import os
import aiohttp
import asyncio
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

PUBMED_API_URL = os.environ.get("PUBMED_API_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
ARXIV_API_URL = os.environ.get("ARXIV_API_URL", "http://export.arxiv.org/api/query")


async def fetch_pubmed_count(session: aiohttp.ClientSession, query: str) -> int:
    url = PUBMED_API_URL
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "usehistory": "y"
    }
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                count_str = data.get("esearchresult", {}).get("count", "0")
                return int(count_str)
    except Exception as e:
        logger.error(f"PubMed corpus fetch failed: {e}")
    return 0

async def fetch_arxiv_count(session: aiohttp.ClientSession, query: str) -> int:
    url = ARXIV_API_URL
    params = {
        "search_query": f"all:{query}",
        "max_results": 1
    }
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                xml_text = await response.text()
                root = ET.fromstring(xml_text)
                # Parse atom xml strictly for opensearch:totalResults tag
                ns = {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}
                total_node = root.find("opensearch:totalResults", ns)
                if total_node is not None and total_node.text:
                    return int(total_node.text)
    except Exception as e:
        logger.error(f"arXiv corpus fetch failed: {e}")
    return 0

async def get_corpus_visibility(query: str) -> dict:
    """
    Estimates the available corpus dimension from external resources.
    Returns dictionary with counts prior to API fetches occurring.
    """
    logger.info(f"Estimating corpus visibility for query: '{query}'")
    async with aiohttp.ClientSession() as session:
        pubmed_task = fetch_pubmed_count(session, query)
        arxiv_task = fetch_arxiv_count(session, query)
        
        pubmed_count, arxiv_count = await asyncio.gather(pubmed_task, arxiv_task)
        
        return {
            "pubmed_estimated": pubmed_count,
            "arxiv_estimated": arxiv_count,
            "total_estimated": pubmed_count + arxiv_count
        }
