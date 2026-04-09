import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Initialize model globally to avoid reloading
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model (allenai/specter2_base)...")
        # Ensure it works with the latest HF
        _model = SentenceTransformer('allenai/specter2_base')
    return _model

def prepare_text(paper: Dict[str, Any]) -> str:
    title = paper.get('title', '')
    abstract = paper.get('abstract', '') or ''
    keywords = paper.get('keywords', '')
    journal_name = paper.get('journal_name', '')
    return f"{title}. {abstract}. Methods: {keywords}. {journal_name}".strip()

def embed_papers(papers: List[Dict[str, Any]]) -> None:
    """
    Computes embeddings for papers and stores them in the paper dicts under the 'embedding' key.
    Reused across MMR calls in a session.
    """
    if not papers:
        return
        
    model = get_model()
    
    # Only embed papers that don't already have an embedding
    needs_embedding = [p for p in papers if 'embedding' not in p]
    
    if needs_embedding:
        texts = [prepare_text(p) for p in needs_embedding]
        logger.info(f"Computing embeddings for {len(texts)} papers...")
        embeddings = model.encode(texts)
        for p, emb in zip(needs_embedding, embeddings):
            p['embedding'] = emb

def embed_query(query: str) -> np.ndarray:
    model = get_model()
    return model.encode([query])

def apply_mmr(
    query_embedding: np.ndarray, 
    papers: List[Dict[str, Any]], 
    top_k: int = 15, 
    lambda_param: float = 0.35,
    current_year: int = None
) -> List[Dict[str, Any]]:
    """
    Selects top papers using Maximal Marginal Relevance (MMR) with time-weighted relevance.
    Expects papers to already have 'embedding' populated.
    """
    if not papers:
        return []
        
    if current_year is None:
        current_year = datetime.now().year
        
    top_k = min(top_k, len(papers))
    
    # Extract embeddings directly from papers
    doc_embeddings = np.vstack([p['embedding'] for p in papers])
    
    # Relevance of each doc to the query (cosine_similarity returns 2D array, flatten to 1D)
    doc_query_sims = cosine_similarity(doc_embeddings, query_embedding).flatten()
    
    # Apply Time-Weighted Relevance
    weighted_doc_query_sims = np.zeros_like(doc_query_sims)
    for i, p in enumerate(papers):
        pub_year = p.get('year')
        if pub_year is None:
            pub_year = current_year - 5
            
        cosine_sim = doc_query_sims[i]
        weighted_score = cosine_sim * (0.7 + 0.3 * np.exp(-0.05 * (current_year - pub_year)))
        weighted_doc_query_sims[i] = weighted_score
        
        # Store for reference
        p['raw_cosine'] = float(cosine_sim)
        p['time_weighted_relevance'] = float(weighted_score)
    
    # Similarity matrix among docs
    doc_doc_sims = cosine_similarity(doc_embeddings)
    
    selected_indices = []
    unselected_indices = list(range(len(papers)))
    
    # Select the most relevant paper first based on weighted score
    first_sel = int(np.argmax(weighted_doc_query_sims))
    selected_indices.append(first_sel)
    unselected_indices.remove(first_sel)
    
    papers[first_sel]['mmr_score'] = float(weighted_doc_query_sims[first_sel])
    
    # Iteratively select the remaining papers
    while len(selected_indices) < top_k and unselected_indices:
        best_score = -np.inf
        best_idx = -1
        
        # Calculate MMR for all unselected docs
        for unselected_idx in unselected_indices:
            # Weighted Query relevance
            rel_score = weighted_doc_query_sims[unselected_idx]
            
            # Max similarity to already selected docs
            sim_to_selected = max([doc_doc_sims[unselected_idx][s] for s in selected_indices])
            
            # MMR Score
            mmr_score = (lambda_param * rel_score) - ((1 - lambda_param) * sim_to_selected)
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = unselected_idx
                
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        papers[best_idx]['mmr_score'] = float(best_score)
        
    selected_papers = [papers[i] for i in selected_indices]
    
    logger.info(f"Selected {len(selected_papers)} papers using MMR.")
    return selected_papers
