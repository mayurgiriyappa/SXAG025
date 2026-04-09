import os
from dotenv import load_dotenv
load_dotenv()

import sys
import json
import asyncio
import logging
from corpus import get_corpus_visibility
from planner import generate_queries
from fetcher import fetch_all_queries
from mmr import apply_mmr, embed_papers, embed_query
from access import process_paper_access_layer
from clustering import apply_clustering
from synthesizer import generate_synthesis
from compiler import compile_final_report

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_pipeline(user_query: str):
    logger.info(f"Starting research pipeline for query: '{user_query}'")
    
    # 0. Corpus Visibility
    corpus_metadata = await get_corpus_visibility(user_query)
    
    # 1. Planner Agent
    queries = generate_queries(user_query, num_queries=4)
    if not queries:
        logger.error("Planner failed to generate queries.")
        return None
        
    # 2. Fetcher
    # query_results is Dict[str, List[Dict]]
    query_results = await fetch_all_queries(queries, limit_per_query=25)
    
    # Track metrics
    total_fetched = sum(len(papers) for papers in query_results.values())
    if total_fetched == 0:
        logger.warning("No papers fetched. Exiting.")
        return None
        
    corpus_metadata['after_fetcher'] = total_fetched
        
    # 3. Multi-Query MMR (Federated Pass)
    merged_pool = []
    logger.info("Executing Stage 1 Federated MMR per query...")
    
    for q, papers in query_results.items():
        if not papers:
            continue
            
        # Extract embeddings for the pool of papers
        embed_papers(papers)
        
        # Embed the specific sub-query
        q_emb = embed_query(q)
        
        # MMR to get top 8 for this query
        top_8 = apply_mmr(q_emb, papers, top_k=8, lambda_param=0.35)
        merged_pool.extend(top_8)
        
    # Deduplicate merged pool by DOI and title
    deduplicated_pool = []
    seen_dois = set()
    seen_titles = set()
    
    for p in merged_pool:
        doi = p.get('doi', '')
        title_lower = p.get('title', '').lower().strip()
        
        if doi and doi in seen_dois:
            continue
        if title_lower in seen_titles:
            continue
            
        if doi:
            seen_dois.add(doi)
        seen_titles.add(title_lower)
        
        deduplicated_pool.append(p)
        
    logger.info(f"Deduplicated merged pool contains {len(deduplicated_pool)} unique papers.")
    
    # 4. Final MMR Pass
    # Use original user query
    logger.info("Executing Stage 2 Final MMR on aggregated pool...")
    user_query_emb = embed_query(user_query)
    
    final_selected = apply_mmr(user_query_emb, deduplicated_pool, top_k=15, lambda_param=0.35)
    
    corpus_metadata['after_mmr'] = len(final_selected)
    logger.info(f"Corpus Funnel: Explored {corpus_metadata['total_estimated']} papers -> Retrieved {total_fetched} -> Shortlisted {len(final_selected)}")
    
    # 5. Paper Access Layer
    final_selected = await process_paper_access_layer(final_selected)
    
    # 6. HDBSCAN Clustering
    final_selected = apply_clustering(final_selected)
    
    # Clean up non-serializable fields (like 'embedding' numpy array)
    for p in final_selected:
        p.pop('embedding', None)
        
    # 7. Synthesizer Layer
    synthesis_md = generate_synthesis(final_selected, user_query)
    
    # 8. Compiler Layer
    final_pdf_path = compile_final_report(user_query, corpus_metadata, synthesis_md, final_selected)
    
    # Prepare Output
    output = {
        "corpus_metadata": corpus_metadata,
        "queries": queries,
        "total_papers_fetched": total_fetched,
        "selected_papers": final_selected,
        "compiled_pdf_path": final_pdf_path
    }
    
    return output

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<research_query>\"")
        sys.exit(1)
        
    query = sys.argv[1]
    
    # Handle Python on Windows event loop issues for async
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    output = asyncio.run(process_pipeline(query))
    if output:
        print("\n--- FINAL OUTPUT ---\n")
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
        # Also save to a file for persistence
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        logger.info("Pipeline completed successfully! Output saved to output.json.")
    else:
        logger.error("Pipeline failed to produce output.")

if __name__ == "__main__":
    main()
