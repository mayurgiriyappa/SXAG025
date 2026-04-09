# critic_agent.py
from pydantic import BaseModel, Field
from state import ResearchState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the exact decision structure
class CriticDecision(BaseModel):
    is_complete: bool = Field(
        description="True ONLY IF the graph has sufficient contradictions and distinct clusters. False if it is too shallow or one-sided."
    )
    feedback: str = Field(
        description="If false, give strict, specific instructions to the Planner on what to search next (e.g., 'Search for papers critiquing Methodology X'). If true, summarize why the graph is good."
    )

def critic_node(state: ResearchState):
    print("--- NODE: CRITIC AGENT (LLM Evaluation) ---")
    
    topic = state.get("topic", "Unknown Topic")
    metrics = state.get("networkx_metrics", {})
    
    # We use gemini-1.5-flash here because it is incredibly fast and cheap, 
    # making it perfect for rapid routing decisions in a loop.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    system_prompt = """You are the Lead Research Director. 
    Review the metrics of the current knowledge graph. 
    A robust graph MUST have:
    1. At least 1 clear contradiction or debate.
    2. Multiple distinct thematic clusters.
    
    If it meets these criteria, set is_complete to True. 
    If it fails, set is_complete to False and write strict feedback telling the Planner Agent what specific orthogonal search query to run next to fix the graph."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Target Topic: {topic}\nCurrent Graph Metrics: {metrics}")
    ])
    
    # Bind the schema and invoke
    structured_llm = llm.with_structured_output(CriticDecision)
    chain = prompt | structured_llm
    
    print("[*] Critic is analyzing graph topology...")
    decision = chain.invoke({"topic": topic, "metrics": metrics})
    
    if decision.is_complete:
        print(f"[+] Critic Approved: {decision.feedback}")
    else:
        print(f"[-] Critic Rejected: {decision.feedback}")
        
    # Update the LangGraph state
    return {
        "research_complete": decision.is_complete,
        "critic_feedback": decision.feedback
    }