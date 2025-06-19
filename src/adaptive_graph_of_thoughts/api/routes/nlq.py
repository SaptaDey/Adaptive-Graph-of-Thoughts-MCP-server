from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Dict

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from ...domain.services.neo4j_utils import execute_query
from ...services.llm import LLM_QUERY_LOGS, ask_llm

logger = logging.getLogger(__name__)

nlq_router = APIRouter()


def _log_query(prompt: str, response: str) -> None:
    """Append a prompt-response pair to the query log."""
    entry = {"prompt": prompt, "response": response}
    LLM_QUERY_LOGS.append(entry)
    if len(LLM_QUERY_LOGS) > 5:
        LLM_QUERY_LOGS.pop(0)


@nlq_router.post("/nlq")
async def nlq_endpoint(payload: Dict[str, str] = Body(...)) -> StreamingResponse:
    """Translate a natural language question to Cypher and stream results."""
    question = payload.get("question", "")
    cypher_prompt = f"Translate the question to a Cypher query: {question}"
    cypher = await asyncio.to_thread(ask_llm, cypher_prompt)
    _log_query(cypher_prompt, cypher)

    async def event_stream() -> AsyncGenerator[bytes, None]:
        yield json.dumps({"cypher": cypher}).encode() + b"\n"
        try:
            records = await execute_query(cypher)
            rows = [dict(r) for r in records]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            rows = {"error": "Query execution failed"}
        yield json.dumps({"records": rows}).encode() + b"\n"
        summary_prompt = (
            f"Answer the question '{question}' using this data: {rows}."
            " Respond in under 50 words."
        )
        summary = await asyncio.to_thread(ask_llm, summary_prompt)
        _log_query(summary_prompt, summary)
        yield json.dumps({"summary": summary}).encode() + b"\n"

    return StreamingResponse(event_stream(), media_type="application/json")
