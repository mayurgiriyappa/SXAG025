import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any
import markdown
from xhtml2pdf import pisa
from pypdf import PdfWriter
from io import BytesIO

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def safe_slug(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())[:30]

def create_frontmatter_pdf(query: str, corpus_metadata: dict, synthesis_md: str, papers: List[Dict[str, Any]]) -> BytesIO:
    """
    Renders the cover page and synthesizer markdown into a structured PDF byte stream.
    """
    total_estimated = corpus_metadata.get('total_estimated', 0)
    retrieved = corpus_metadata.get('after_fetcher', 0)
    shortlisted = len(papers)
    
    full_text_count = sum(1 for p in papers if p.get('paper_access', {}).get('full_text_available', False))
    abstract_count = shortlisted - full_text_count
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate list of papers HTML
    papers_html = "<ul>"
    for p in papers:
        title = p.get('title', 'Unknown')
        access = p.get('paper_access', {})
        badge = "🔓 Open Access" if access.get('full_text_available') else "📄 Abstract Only"
        cluster = p.get('cluster_id', 0)
        papers_html += f"<li><b>[Cluster {cluster}]</b> {title} - <i>{badge}</i></li>"
    papers_html += "</ul>"
    
    cover_html = f"""
    <div style="text-align: center; margin-top: 100px;">
        <h1>ScholAR Agent &mdash; Research Report</h1>
        <h2>Topic: {query}</h2>
        <p>Generated: {date_str}</p>
        <hr />
        <p><b>Funnel summary:</b> Total corpus explored: {total_estimated} | Retrieved: {retrieved} | Shortlisted: {shortlisted} | Full text accessed: {full_text_count} | Abstract only: {abstract_count}</p>
    </div>
    <div style="margin-top: 50px;">
        <h3>Selected Papers Map:</h3>
        {papers_html}
    </div>
    """
    
    # Structure Synth Report (convert markdown to HTML)
    synth_html = markdown.markdown(synthesis_md)
    
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{ margin: 2cm; }}
            .page-break {{ page-break-before: always; }}
            body {{ font-family: Helvetica, Arial, sans-serif; font-size: 12pt; line-height: 1.5; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            hr {{ border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        {cover_html}
        <div class="page-break"></div>
        <h2>Synthesizer Report</h2>
        {synth_html}
    </body>
    </html>
    """
    
    out_io = BytesIO()
    pisa_status = pisa.CreatePDF(BytesIO(full_html.encode("utf-8")), dest=out_io)
    if pisa_status.err:
        logger.error("XHTML2PDF rendering encountered an error.")
    out_io.seek(0)
    return out_io

def compile_final_report(query: str, corpus_metadata: dict, synthesis_md: str, papers: List[Dict[str, Any]]):
    logger.info("Initializing Master PDF Compiler sequence...")
    
    slug = safe_slug(query)
    date_str = datetime.now().strftime("%Y%m%d")
    final_path = os.path.join(OUTPUTS_DIR, f"ScholAR_Report_{slug}_{date_str}.pdf")
    
    writer = PdfWriter()
    
    # 1. Inject Frontmatter
    try:
        front_io = create_frontmatter_pdf(query, corpus_metadata, synthesis_md, papers)
        writer.append(front_io)
    except Exception as e:
        logger.error(f"Failed to generate frontmatter PDF components: {e}")
        
    # 2. Append all individual clustered papers sequentially
    # (Since access.py abstracts failures locally into PDF bytes, we safely append all of them)
    for idx, paper in enumerate(papers):
        access_meta = paper.get('paper_access', {})
        pdf_path = access_meta.get('pdf_path')
        
        if pdf_path and os.path.exists(pdf_path):
            try:
                writer.append(pdf_path)
            except Exception as e:
                logger.error(f"Error appending PDF for {paper.get('title')}: {e}")
        else:
            logger.warning(f"Target PDF payload missing for append on paper index {idx}.")
            
    # Flush byte payload to static /outputs directory
    try:
        with open(final_path, "wb") as f_out:
            writer.write(f_out)
        logger.info(f"Successfully compiled aggregate PDF report: {final_path}")
    except Exception as e:
        logger.error(f"Render flush failed for destination {final_path}: {e}")
        
    return final_path
