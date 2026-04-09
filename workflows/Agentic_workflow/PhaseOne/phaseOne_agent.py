"""PhaseONE: nested research pipeline.

Input  : research title (user message)
Output : state['curated_papers']  (similar curated research articles)

Outer SequentialAgent (research_pipeline):
  1. planner_agent  - LLM planner cloned from Agentic_workflow.PhaseOne.Planner.
                      Writes state['orthogonal_queries'] via the
                      submit_orthogonal_queries tool.
  2. raw_processor  - inner SequentialAgent (Raw_processing/agent.py) with:
     a. fetcher     - CrossRef async fan-out, one HTTP call per query.
                      Writes state['query_results'] + state['total_fetched'].
     b. mmr         - two-pass MMR (federated top-8 per query + dedup,
                      then final top-15 against the original research title).
                      Writes state['curated_papers'].
"""
from __future__ import annotations

from google.adk.agents.sequential_agent import SequentialAgent

from .Planner.agent import Planner
from .Raw_processing.agent import raw_processor


root_agent = SequentialAgent(
    name="research_pipeline",
    sub_agents=[
        Planner.clone(),
        raw_processor,
    ],
)
