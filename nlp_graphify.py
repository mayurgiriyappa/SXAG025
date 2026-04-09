from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    label: str = Field(description="Human readable label for the concept")
    type: str = Field(description="Type of node, e.g. Concept, Paper, Methodology")

class Edge(BaseModel):
    source: str = Field(description="ID of the source node")
    target: str = Field(description="ID of the target node")
    relation: str = Field(description="Relationship label, e.g. uses, critiques, extends")

class Contradiction(BaseModel):
    description: str = Field(description="Description of the contradiction")
    paper_ids: List[str] = Field(description="List of paper IDs that are involved in the contradiction")

class Community(BaseModel):
    name: str = Field(description="Name of the thematic cluster or community")
    node_ids: List[str] = Field(description="List of node IDs that belong to this community")

class GraphExtraction(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    contradictions: List[Contradiction]
    communities: List[Community]
    god_nodes: List[str] = Field(description="List of node IDs representing highly influential anchor concepts")

def process_documents(documents: List[Dict[str, Any]], llm: Any, extract_entities: bool = True, detect_communities: bool = True, find_contradictions: bool = True) -> Dict[str, Any]:
    print("[*] Inside nlp_graphify.process_documents...")
    combined_text = "\n\n".join([d.get("text", "") for d in documents])
    
    prompt = f"""
    You are an expert academic knowledge graph extractor. 
    Analyze the following research documents and extract:
    1. Key concepts or papers as Nodes.
    2. Relationships between these nodes as Edges.
    3. Any contradictions or disagreements between the approaches found in the texts.
    4. Thematic communities or clusters.
    5. 'God nodes' (highly influential concepts/papers that anchor the graph).
    
    TEXTS:
    {combined_text[:500000]} # Trim to fit in standard context just in case it's huge
    """
    
    print("[*] Calling Gemini LLM with structured output schema...")
    structured_llm = llm.with_structured_output(GraphExtraction)
    response = structured_llm.invoke(prompt)
    
  # Dump the LLM output to a dictionary so we can inject our math into it
    graph_dict = response.model_dump()
    
    # --- 2. THE MATHEMATICAL INTEGRATION (NetworkX) ---
    print("[*] Calculating deterministic Centrality and Gap scores...")
    
    # Build a temporary deterministic graph from the LLM's edges
    G = nx.DiGraph()
    for edge in graph_dict.get('edges', []):
        G.add_edge(edge['source'], edge['target'])
        
    if len(G.nodes) > 0:
        # Calculate standard topological metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Flatten contradiction paper IDs for quick lookup
        contradicted_nodes = set()
        for c in graph_dict.get('contradictions', []):
            for pid in c.get('paper_ids', []):
                contradicted_nodes.add(pid)
                
        # Inject the math back into the node dictionaries
        for node in graph_dict.get('nodes', []):
            nid = node['id']
            
            # Centrality: How heavily linked is this node?
            c_score = degree_centrality.get(nid, 0.0)
            b_score = betweenness_centrality.get(nid, 0.0)
            
            # Gap Score Logic: 
            # High betweenness (it connects isolated clusters) * High Sparsity (1 - degree centrality)
            base_gap = b_score * (1.0 - c_score)
            
            # Add a massive penalty/bonus if it is an active contradiction
            conflict_multiplier = 1.5 if nid in contradicted_nodes else 1.0
            
            final_gap_score = base_gap * conflict_multiplier
            
            # Round them for clean JSON output
            node['centrality_score'] = round(c_score, 4)
            node['gap_score'] = round(final_gap_score, 4)

    return graph_dict
