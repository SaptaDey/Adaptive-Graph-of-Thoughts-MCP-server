from __future__ import annotations

from fastapi import APIRouter
from ...domain.services.neo4j_utils import execute_query

explorer_router = APIRouter()


@explorer_router.get("/graph")
async def graph_explorer_json(limit: int = 50):
    """
    Retrieve a list of nodes and edges from the Neo4j database for use in a graph explorer dashboard.
    
    Parameters:
        limit (int): Maximum number of relationships to include in the result. Defaults to 50.
    
    Returns:
        dict: A dictionary with two keys:
            - "nodes": List of unique node objects, each containing node ID, labels, and properties.
            - "edges": List of relationship objects, each containing relationship ID, type, source and target node IDs, and properties.
    """
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
