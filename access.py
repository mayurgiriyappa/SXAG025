import os
import re
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger(__name__)

# Constants from .env
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "system@example.com")
UNPAYWALL_API_URL = os.environ.get("UNPAYWALL_API_URL", "https://api.unpaywall.org/v2")
EUROPE_PMC_API_URL = os.environ.get("EUROPE_PMC_API_URL", "https://www.ebi.ac.uk/europepmc/webservices/rest/search")
SEMANTIC_SCHOLAR_API_URL = os.environ.get("SEMANTIC_SCHOLAR_API_URL", "https://api.semanticscholar.org/graph/v1/paper")
# Store in a relative tmp folder to avoid permission errors on Windows root volumes
TMP_DIR = os.path.join(os.getcwd(), 'tmp', 'papers')

os.makedirs(TMP_DIR, exist_ok=True)

def safe_doi_filename(doi: str) -> str:
    if not doi:
        return "unknown_doi"
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', doi)

def generate_fallback_pdf(paper: Dict[str, Any], out_path: str):
    """
    Creates an abstract-only PDF compilation.
    """
    try:
        doc = SimpleDocTemplate(out_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = styles['Heading1']
        normal_style = styles['Normal']
        
        meta_style = ParagraphStyle('Meta', parent=normal_style, textColor='grey', spaceAfter=12)
        warning_style = ParagraphStyle('Warning', parent=normal_style, textColor='red', spaceBefore=12, spaceAfter=24)
        
        flowables = []
        
        title = paper.get('title', 'Unknown Title')
        authors = ", ".join(paper.get('authors', [])) or "Unknown Authors"
        journal = paper.get('journal_name', '')
        year = str(paper.get('year', ''))
        doi = paper.get('doi', '')
        abstract = paper.get('abstract', '') or 'No abstract available.'
        
        meta_text = f"Authors: {authors}<br/>Journal: {journal} ({year})<br/>DOI: {doi}"
        
        flowables.append(Paragraph(title, title_style))
        flowables.append(Paragraph(meta_text, meta_style))
        flowables.append(Paragraph("Full text unavailable — institutional access required.", warning_style))
        flowables.append(Paragraph("Abstract", styles['Heading2']))
        flowables.append(Paragraph(abstract, normal_style))
        
        doc.build(flowables)
    except Exception as e:
        logger.error(f"Fallback PDF generation failed: {e}")

async def download_pdf(session: aiohttp.ClientSession, url: str, out_path: str) -> bool:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Some servers don't like missing User-Agent profiles
        async with session.get(url, headers=headers, allow_redirects=True, timeout=15) as response:
            content_type = response.headers.get('Content-Type', '').lower()
            if response.status == 200 and 'application/pdf' in content_type:
                with open(out_path, 'wb') as f:
                    f.write(await response.read())
                return True
    except Exception as e:
        logger.debug(f"Direct PDF download failed for {url}: {e}")
    return False

async def get_paper_access(session: aiohttp.ClientSession, paper: Dict[str, Any]) -> Dict[str, Any]:
    doi = paper.get("doi")
    
    # Path preparation string
    safe_name = safe_doi_filename(doi) if doi else "unknown_doi"
    # To support concurrent identical unknown DOI failures, add a random or hash factor if desired,
    # but we just overwrite them here for simplicity in the mockup structure
    out_path = os.path.join(TMP_DIR, f"{safe_name}.pdf")
    
    if not doi:
        # Fallback instantly if no DOI attached
        generate_fallback_pdf(paper, out_path)
        return {
            "doi": None,
            "access_method": "abstract_only",
            "pdf_path": out_path,
            "full_text_available": False
        }

    # 1. Unpaywall Check
    try:
        url = f"{UNPAYWALL_API_URL}/{doi}?email={CONTACT_EMAIL}"
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("is_oa"):
                    pdf_url = data.get("best_oa_location", {})
                    # best_oa_location could be None if oa lacks strict location mapping
                    if pdf_url and pdf_url.get("url_for_pdf"): 
                        if await download_pdf(session, pdf_url.get("url_for_pdf"), out_path):
                            return {"doi": doi, "access_method": "unpaywall", "pdf_path": out_path, "full_text_available": True}
    except Exception:
        pass

    # 2. Europe PMC Check (requires resolving DOI to PMCID through EBI REST config)
    try:
        epmc_url = f"{EUROPE_PMC_API_URL}?query=DOI:{doi}&format=json&resultType=core"
        async with session.get(epmc_url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get("resultList", {}).get("result", [])
                if results:
                    pmcid = results[0].get("pmcid")
                    if pmcid:
                        render_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                        if await download_pdf(session, render_url, out_path):
                            return {"doi": doi, "access_method": "europepmc", "pdf_path": out_path, "full_text_available": True}
    except Exception:
        pass

    # 3. Semantic Scholar Graph API Check
    try:
        s2_url = f"{SEMANTIC_SCHOLAR_API_URL}/{doi}?fields=openAccessPdf"
        async with session.get(s2_url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                pdf_data = data.get("openAccessPdf")
                if pdf_data and pdf_data.get("url"):
                    if await download_pdf(session, pdf_data.get("url"), out_path):
                        return {"doi": doi, "access_method": "semanticscholar", "pdf_path": out_path, "full_text_available": True}
    except Exception:
        pass

    # 4. DOI Direct Resolution checking against PDF responses
    try:
        doi_url = f"https://doi.org/{doi}"
        if await download_pdf(session, doi_url, out_path):
            return {"doi": doi, "access_method": "doi_redirect", "pdf_path": out_path, "full_text_available": True}
    except Exception:
        pass

    # 5. Fallback Synthesizer triggered exclusively if all 4 vectors fail
    generate_fallback_pdf(paper, out_path)
    return {
        "doi": doi,
        "access_method": "abstract_only",
        "pdf_path": out_path,
        "full_text_available": False
    }

async def process_paper_access_layer(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes the MMR shortlisted papers and systematically attempts to download PDFs, falling back to abstracts.
    Modifies paper dictionaries with tracking variables.
    """
    logger.info("Executing Paper Access Layer for the final candidate pool...")
    async with aiohttp.ClientSession() as session:
        tasks = [get_paper_access(session, p) for p in papers]
        access_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for p, res in zip(papers, access_results):
            if isinstance(res, Exception):
                logger.error(f"Error checking access for DOI {p.get('doi')}: {res}")
                p['paper_access'] = {
                    "doi": p.get('doi'),
                    "access_method": "error",
                    "pdf_path": None,
                    "full_text_available": False
                }
            else:
                p['paper_access'] = res
                
    return papers
