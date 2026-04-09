from google.adk.agents.llm_agent import Agent


graphify_agent = LlmAgent(
    model=MODEL_NAME,
    name='graphify_agent',
    instructions=(
        "You are the Graphify Engine. "
        "Input: {curated articles}. "
        "Step 1: Use the `extract_knowledge_graph` tool to map the entities and relations. "
        "Step 2: Take the resulting graph and use the `cluster_graph_nodes` tool to group related concepts. "
        "Output: The final {knowledge graph + clustering} data structure."
    ),
    tools=[extract_knowledge_graph, cluster_graph_nodes]
)