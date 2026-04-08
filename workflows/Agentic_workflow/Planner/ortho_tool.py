from google.adk.tools.tool_context import ToolContext
from typing import Dict, Any

def submit_orthogonal_queries(
    technical_query: str,
    constraints_query: str,
    context_query: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Submits exactly 3 distinct, non-overlapping (orthogonal) search queries based on the research topic.
    
    Orthogonal means the queries MUST explore completely different dimensions of the topic to maximize diversity. Do not use direct synonyms.
    
    Dimension Guidelines:
    1. technical_query: Focuses on mechanisms, architecture, chemistry, or 'how it works'.
    2. constraints_query: Focuses on limitations, failures, non-responders, or bottlenecks.
    3. context_query: Focuses on clinical applications, therapeutic use, or historical context.
    
    Example for topic "Pre-biotic potential in human gut microbiome":
    - technical_query: "Microbial metabolomics and short-chain fatty acid pathways of dietary fibers"
    - constraints_query: "Inter-individual variability and clinical limitations in prebiotic responsiveness"
    - context_query: "Therapeutic applications of microbiome modulation in gastrointestinal disease"
    """
    
    # 1. Construct the list from the LLM's arguments
    queries = [technical_query, constraints_query, context_query]

    # 2. Persist structured queries to session state for the Fetcher Agent
    tool_context.state["orthogonal_queries"] = queries
    
    # 3. Return a success signal
    return {"status": "success", "queries": queries}