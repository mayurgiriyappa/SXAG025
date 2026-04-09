from typing import List, Dict, Any
import graphifyy

def cluster_graph_nodes(knowledge_graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies clustering to the graph using graphifyy's clustering algorithms.
    
    Args:
        knowledge_graph_data: The dictionary representation of the graph from the previous step.
        
    Returns:
        A dictionary containing the clustered communities.
    """
    print("[Tool Execution] Clustering graph using graphifyy...")
    
    # Reconstruct the graphifyy object from the ADK dictionary input
    kg = graphifyy.KnowledgeGraph.from_dict(knowledge_graph_data)
    
    # Execute graphifyy's native clustering (e.g., Louvain, K-means, etc.)
    # Assuming graphifyy has a clustering module or method
    clusters = kg.cluster_nodes(algorithm="louvain") 
    
    return {
        "clustering_results": clusters,
        "graph_metrics": kg.get_metrics() # Example of returning metadata
    }