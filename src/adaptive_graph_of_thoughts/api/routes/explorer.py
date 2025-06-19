from __future__ import annotations

from fastapi import APIRouter
from ...domain.services.neo4j_utils import execute_query

explorer_router = APIRouter()


@explorer_router.get("/graph")
async def graph_explorer_json(limit: int = 50):
    """Return nodes and edges for dashboard graph explorer."""
    query = (
        "MATCH (n)-[r]->(m) RETURN id(n) AS sid, labels(n) AS slabels, "
        "properties(n) AS sprops, id(m) AS tid, labels(m) AS tlabels, "
        "properties(m) AS tprops, id(r) AS rid, type(r) AS rtype, "
        "properties(r) AS rprops LIMIT $limit"
    )
    records = await execute_query(query, {"limit": limit})
    nodes = {}
    edges = []
    for rec in records:
        sid = str(rec["sid"])
        tid = str(rec["tid"])
        nodes[sid] = {"id": sid, "labels": rec["slabels"], "properties": rec["sprops"]}
        nodes[tid] = {"id": tid, "labels": rec["tlabels"], "properties": rec["tprops"]}
        edges.append(
            {
                "id": str(rec["rid"]),
                "type": rec["rtype"],
                "source": sid,
                "target": tid,
                "properties": rec["rprops"],
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}
