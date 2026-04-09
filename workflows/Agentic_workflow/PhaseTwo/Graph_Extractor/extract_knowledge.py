from typing import List, Dict, Any
import graphifyy

def extract_knowledge_graph(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts entities and relationships from curated articles using graphifyy.
    
    Args:
        articles: The filtered, curated articles from Phase 1.
        
    Returns:
        A dictionary representation of the graphifyy Knowledge Graph.
    """
    print("[Tool Execution] Building knowledge graph with graphifyy...")
    
    # Initialize the graphifyy Knowledge Graph
    kg = graphifyy.KnowledgeGraph()
    
    # Process each article through graphifyy
    for article in articles:
        abstract = article.get("abstract", "")
        if abstract:
            # Assuming graphifyy has an extraction or document ingestion method
            kg.add_text(abstract)
            
    # ADK requires JSON-serializable returns, so we export the graphifyy object to a dict
    return kg.to_dict()