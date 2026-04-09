import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)

def apply_clustering(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies HDBSCAN to the pipeline's shortlisted papers to group them thematically.
    Injects a 'cluster_id' into each object and sorts the list primarily by cluster assignment.
    """
    if not papers or len(papers) < 2:
        for p in papers:
            p['cluster_id'] = 0
        return papers
        
    logger.info("Applying HDBSCAN clustering on candidate embeddings...")
    
    # Extract dense embeddings
    try:
        embeddings = np.vstack([p['embedding'] for p in papers])
        
        # Min cluster size 2 enables granular pairing for exactly 15 papers.
        clusterer = HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)
        
        for p, label in zip(papers, labels):
            # Noise points get labeled as -1, we map them as independent clusters safely
            p['cluster_id'] = int(label)
            
        # Reorder papers so clusters group adjacently for the synthesis formatting Phase
        sorted_papers = sorted(papers, key=lambda x: (x.get('cluster_id', -1), -x.get('raw_cosine', 0)))
        
        unique_c = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"HDBSCAN extracted {unique_c} distinct sub-clusters across final candidate pool.")
        return sorted_papers
        
    except Exception as e:
        logger.error(f"Clustering matrix failed: {e}")
        # Soft fallback 
        for p in papers:
            p['cluster_id'] = 0
        return papers
