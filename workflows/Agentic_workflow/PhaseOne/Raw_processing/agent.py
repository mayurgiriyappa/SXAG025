"""Raw_processing: fetcher and MMR stages chained as an inner SequentialAgent.

Reads:  state['orthogonal_queries']  (set by the upstream Planner)
Writes: state['curated_papers']      (top-15 MMR-selected papers)
"""
from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

from . import fetcher, mmr


def _strip_embedding(paper: dict) -> dict:
    """Drop the numpy 'embedding' field so the dict survives state serialization."""
    return {k: v for k, v in paper.items() if k != "embedding"}


class FetcherAgent(BaseAgent):
    """Reads state['orthogonal_queries'] and writes state['query_results']."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        queries = ctx.session.state.get("orthogonal_queries", [])
        # Planner tool may store either a list or a {key: query} dict — normalize.
        if isinstance(queries, dict):
            queries = list(queries.values())

        results = await fetcher.fetch_all_queries(queries, limit_per_query=25)
        total = sum(len(v) for v in results.values())

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(
                state_delta={
                    "query_results": results,
                    "total_fetched": total,
                }
            ),
        )


class MmrAgent(BaseAgent):
    """Two-pass Maximal Marginal Relevance filtering.

    Pass 1 (federated):  per-query top 8 with lambda=0.35, deduped by DOI/title.
    Pass 2 (final):      embed the original user query and rank the deduped pool,
                         keeping top 15 with lambda=0.35.

    Output: state['curated_papers']  (embeddings stripped for serialization).
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        query_results = ctx.session.state.get("query_results", {})

        # Pass 1: per-query top-8 + merge
        merged: list[dict] = []
        for q, papers in query_results.items():
            if not papers:
                continue
            mmr.embed_papers(papers)  # mutates each paper with 'embedding'
            q_emb = mmr.embed_query(q)
            top_8 = mmr.apply_mmr(q_emb, papers, top_k=8, lambda_param=0.35)
            merged.extend(top_8)

        # Dedup by DOI then title
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        deduped: list[dict] = []
        for p in merged:
            doi = p.get("doi", "") or ""
            title = (p.get("title", "") or "").lower().strip()
            if doi and doi in seen_dois:
                continue
            if title in seen_titles:
                continue
            if doi:
                seen_dois.add(doi)
            seen_titles.add(title)
            deduped.append(p)  # keep embedding in-process for pass 2

        # Pass 2: final MMR vs original user query (the research title)
        user_query = ""
        if ctx.user_content and ctx.user_content.parts:
            user_query = "".join(
                p.text or "" for p in ctx.user_content.parts
            ).strip()

        curated: list[dict] = []
        if deduped and user_query:
            user_emb = mmr.embed_query(user_query)
            curated = mmr.apply_mmr(
                user_emb, deduped, top_k=15, lambda_param=0.35
            )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(
                state_delta={
                    "curated_papers": [_strip_embedding(p) for p in curated],
                }
            ),
        )


# Inner pipeline: fetcher → mmr
raw_processor = SequentialAgent(
    name="raw_processor",
    sub_agents=[
        FetcherAgent(name="fetcher"),
        MmrAgent(name="mmr"),
    ],
)
