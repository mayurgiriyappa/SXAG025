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
    
    # Return purely as a dictionary to match previous expectations in graphify_node.py
    return response.model_dump()
