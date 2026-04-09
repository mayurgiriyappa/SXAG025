# graphify_node.py
import os
import PyPDF2
from nlp_graphify import process_documents
from state import ResearchState
from langchain_google_genai import ChatGoogleGenerativeAI
def graphify_build_and_analyze(state: ResearchState):
    """
    Unified node that reads FULL PDF texts from a local directory, 
    builds the knowledge graph, and runs topological analysis.
    """
    print("--- NODE: GRAPHIFY (Full-Text Extraction & Analysis) ---")
    extraction_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0
    )
    
    curated_papers = state.get("curated_papers", [])
    if not curated_papers:
        return {"critic_feedback": "No papers found to graph."}

    # Define the directory where your PDFs are stored locally
    pdf_directory = "curated_papers" 
    
    # Create the folder if it doesn't exist (prevents crashing)
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"[!] Warning: Directory '{pdf_directory}' was missing and has been created.")

    documents = []

    # 1. Parse the PDFs into raw text
    for paper in curated_papers:
        pdf_filename = paper.get("filename") # e.g., "smith_et_al_2025.pdf"
        
        # Fallback if filename isn't in state: try to use the paper ID
        if not pdf_filename:
            pdf_filename = f"{paper.get('id')}.pdf"
            
        pdf_path = os.path.join(pdf_directory, pdf_filename)

        if not os.path.exists(pdf_path):
            print(f"[!] Warning: PDF not found at {pdf_path}. Skipping.")
            continue

        print(f"[*] Parsing full text from {pdf_filename}...")

        full_text = ""
        try:
            # Open the PDF in read-binary mode
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Loop through every page and extract text
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        full_text += extracted + "\n"
        except Exception as e:
            print(f"[!] Error reading {pdf_filename}: {e}")
            continue

        # 2. Construct the massive payload for Graphify
        documents.append({
            "id": paper.get("id"),
            "text": f"Title: {paper.get('title')}\n\n--- FULL PAPER TEXT ---\n{full_text}"
        })

    if not documents:
        return {"critic_feedback": "Failed to read any PDFs. Graph extraction aborted."}

    print(f"[*] Feeding {len(documents)} full-text papers into Graphify engine...")
    print(f"[*] Note: This is a massive context payload. Execution will take time.")

    # 3. Execute the unified Graphify pipeline on the full text
    graph_results = process_documents(
        documents=documents,
        llm=extraction_llm,
        extract_entities=True,
        detect_communities=True, 
        find_contradictions=True 
    )

    # 4. Map the Graphify output back to our LangGraph State
    nodes = graph_results.get("nodes", [])
    edges = graph_results.get("edges", [])
    
    insights = {
        "contradictions": graph_results.get("contradictions", []),
        "clusters": graph_results.get("communities", []),
        "key_papers": graph_results.get("god_nodes", []), 
        "total_nodes": len(nodes)
    }
    
    print(f"[*] Graphify complete. Found {len(nodes)} nodes and {len(insights['contradictions'])} debates.")

    return {
        "graph_nodes": nodes,
        "graph_edges": edges,
        "networkx_metrics": insights 
    }