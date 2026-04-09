# synthesizer_agent.py
import json
from state import ResearchState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def synthesizer_node(state: ResearchState):
    print("--- NODE: SYNTHESIZER AGENT (Drafting Final Report) ---")
    
    # 1. Extract the data from the LangGraph State
    topic = state.get("topic", [])
    metrics = state.get("networkx_metrics", {})
    nodes = state.get("graph_nodes", [])
    
    # 2. Initialize Gemini 2.5 Pro for deep reasoning and long-form writing
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    
    # 3. The System Prompt: Mapping JSON to Markdown
    system_instruction = """You are an elite Academic Researcher and Synthesizer. 
    Your task is to write a comprehensive, publication-ready Literature Review based STRICTLY on the provided Knowledge Graph JSON data.
    
    You must structure your Markdown report with the following exact sections:
    
    # 1. Executive Summary
    Provide a high-level overview of the research topic and the scope of the analyzed papers.
    
    # 2. Core Research Trends (Thematic Clusters)
    Analyze the 'clusters' data. For each cluster:
    - Name the trend.
    - Explain what concepts and papers define this trend.
    - List the key related papers involved in this cluster.
    
    # 3. Key Debates & Research Gaps
    Analyze the 'contradictions' data. 
    - Detail exactly what the conflicting papers disagree on.
    - Identify this as a "Research Gap" that requires further investigation.
    
    # 4. Foundational Literature (God Nodes)
    Analyze the 'key_papers' data.
    - List these highly influential anchor concepts/papers.
    - Explain why they are central to the graph's topology.
    
    # 5. Conclusion
    Summarize the state of the field.
    
    CRITICAL RULES:
    - Do NOT hallucinate external papers. Rely entirely on the provided JSON insights and nodes.
    - Use professional, academic language.
    - Format beautifully using Markdown (bolding, bullet points, and clear headers).
    """
    
    # 4. Prepare the massive context payload
    # We dump the dictionaries to formatted JSON strings so the LLM reads them cleanly
    user_content = f"""
    Research Topic: {topic}
    
    --- EXTRACTED KNOWLEDGE GRAPH METRICS ---
    {json.dumps(metrics, indent=2)}
    
    --- RAW NODE DICTIONARY (For Reference) ---
    {json.dumps(nodes, indent=2)}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", user_content)
    ])
    
    chain = prompt | llm
    
    print("[*] Gemini 2.5 Pro is analyzing the graph and synthesizing the final literature review...")
    print("[*] This may take 30-60 seconds depending on graph size...")
    
    # 5. Execute the generation
    try:
        response = chain.invoke({})
        final_markdown = response.content
        print("[+] Report generated successfully!")
    except Exception as e:
        print(f"[-] Error generating report: {e}")
        final_markdown = "# Error\nFailed to generate the literature review."
    
    # 6. Update the state with the final report
    return {"final_report": final_markdown}