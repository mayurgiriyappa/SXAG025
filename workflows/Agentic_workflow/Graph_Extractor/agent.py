from google.adk.agents.llm_agent import Agent

from .Query_agent.agent import retrieval_agent


graph_agent = Agent(
    model='gemini-2.5-pro',
    name='graph_agent',
    description='A agent that extracts a knowledge graph from retrieved academic papers.',
    instruction=(
        'Use the retrieved papers from the query_agent to extract a structured knowledge graph. '
        'Focus on identifying key entities, relationships, and methodologies mentioned in the papers. '
        'Format the output as a JSON object with nodes representing entities and edges representing relationships.'
    )
    tools=[]
)