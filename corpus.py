import os
import aiohttp
import asyncio
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_KEY="AAMYCibSel5gLjGSpX8UY23IDom34AjO3KHHZ8vq"
ARXIV_API_URL = os.environ.get("ARXIV_API_URL", "http://export.arxiv.org/api/query")
TAVILY_API_KEY="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"
EXA_API_KEY="d95cd89f-10b0-483d-9366-43cb54efec72"


async def fetch_pubmed_count(session: aiohttp.ClientSession, query: str) -> int:
    url = SEMANTIC_SCHOLAR_API_KEY
    params = {
        "db": "semantic_scholar",
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
        logger.error(f"Semantic Scholar corpus fetch failed: {e}")
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
        semantic_scholar_task = fetch_semantic_scholar_count(session, query)
        arxiv_task = fetch_arxiv_count(session, query)
        
        semantic_scholar_count, arxiv_count = await asyncio.gather(semantic_scholar_task, arxiv_task)
        
        return {
            "semantic_scholar_estimated": semantic_scholar_count,
            "arxiv_estimated": arxiv_count,
            "total_estimated": semantic_scholar_count + arxiv_count
        }
