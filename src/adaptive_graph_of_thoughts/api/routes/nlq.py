from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Dict
import re
import logging

from fastapi import APIRouter, Body, HTTPException, Depends
from fastapi.responses import StreamingResponse

from ..schemas import NLQPayload
from ...infrastructure.neo4j_utils import execute_query
from ...services.llm import LLM_QUERY_LOGS, ask_llm
from .mcp import verify_token

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

nlq_router = APIRouter()

MALICIOUS_PATTERNS = [
    re.compile(r"(?i)ignore\s+.*instruction"),
    re.compile(r"(?i)forget\s+.*instruction"),
    re.compile(r"(?i)system:"),
    re.compile(r"(?i)assistant:"),
]


def _validate_question(question: str) -> str:
    """Validate question for obvious prompt injection attempts."""
    for pattern in MALICIOUS_PATTERNS:
        if pattern.search(question):
            logger.warning("Potential prompt injection attempt detected: %s", question)
            raise HTTPException(status_code=400, detail="Malicious pattern detected in question")
    return question.strip()


def _armor(text: str) -> str:
    """Simple prompt armoring to escape curly braces."""
    return text.replace("{", "{{").replace("}", "}}").replace("\n", " ")


def _log_query(prompt: str, response: str) -> None:
    """
    Append a prompt-response pair to the query log, maintaining only the five most recent entries.
    """
    entry = {"prompt": prompt, "response": response}
    LLM_QUERY_LOGS.append(entry)
    if len(LLM_QUERY_LOGS) > 5:
        LLM_QUERY_LOGS.pop(0)


@nlq_router.post("/nlq", dependencies=[Depends(verify_token)])
async def nlq_endpoint(payload: Dict[str, str] = Body(...)) -> StreamingResponse:

    """
    Handles POST requests to the /nlq endpoint, translating a natural language question into a Cypher query, executing it, and streaming the Cypher query, results, and a concise summary as a JSON response.
    
    Parameters:
        payload (Dict[str, str]): JSON body containing the "question" key with the user's natural language query.
    
    Returns:
        StreamingResponse: Streams JSON objects for the generated Cypher query, query results, and a summary answer.
    """

    question = _validate_question(payload.get("question", ""))
    safe_question = _armor(question)
    cypher_prompt = f"Translate the question to a Cypher query: {safe_question}"
    cypher = await asyncio.to_thread(ask_llm, cypher_prompt)
    _log_query(cypher_prompt, cypher)

    async def event_stream() -> AsyncGenerator[bytes, None]:
        """
        Asynchronously streams the generated Cypher query and its execution results as JSON-encoded bytes.
        
        Yields:
            bytes: JSON-encoded Cypher query and query results, each separated by a newline.
        """
        yield json.dumps({"cypher": cypher}).encode() + b"\n"
        try:
            records = await execute_query(cypher)
            rows = [dict(r) for r in records]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise HTTPException(status_code=500, detail="Query execution failed")
            rows = {"error": "Query execution failed"}
        yield json.dumps({"records": rows}).encode() + b"\n"
        summary_prompt = (
            f"Answer the question '{safe_question}' using this data: {rows}."
            " Respond in under 50 words."
        )
        summary = await asyncio.to_thread(ask_llm, summary_prompt)
        _log_query(summary_prompt, summary)
        yield json.dumps({"summary": summary}).encode() + b"\n"

    response = StreamingResponse(event_stream(), media_type="application/json")
    event_stream.response = response
    return response
