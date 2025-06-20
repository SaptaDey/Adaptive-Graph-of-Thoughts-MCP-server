import json
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from ..models.graph_elements import Node, Edge


def prepare_node_properties_for_neo4j(node_pydantic: Node) -> dict[str, Any]:
    """Convert a ``Node`` model into a property dict for Neo4j."""
    if node_pydantic is None:
        return {}

    props: dict[str, Any] = {"id": node_pydantic.id, "label": node_pydantic.label}

    if node_pydantic.confidence:
        for cv_field, cv_val in node_pydantic.confidence.model_dump().items():
            if cv_val is not None:
                props[f"confidence_{cv_field}"] = cv_val

    if node_pydantic.metadata:
        for meta_field, meta_val in node_pydantic.metadata.model_dump().items():
            if meta_val is None:
                continue
            if isinstance(meta_val, datetime):
                props[f"metadata_{meta_field}"] = meta_val.isoformat()
            elif isinstance(meta_val, Enum):
                props[f"metadata_{meta_field}"] = meta_val.value
            elif isinstance(meta_val, (list, set)):
                if all(isinstance(item, (str, int, float, bool)) for item in meta_val):
                    props[f"metadata_{meta_field}"] = list(meta_val)
                else:
                    try:
                        items_as_dicts = [
                            item.model_dump() if hasattr(item, "model_dump") else item
                            for item in meta_val
                        ]
                        props[f"metadata_{meta_field}_json"] = json.dumps(
                            items_as_dicts
                        )
                    except TypeError as e:
                        logger.warning(
                            f"Could not serialize list/set metadata field {meta_field} to JSON: {e}"
                        )
                        props[f"metadata_{meta_field}_str"] = str(meta_val)
            elif hasattr(meta_val, "model_dump"):
                try:
                    props[f"metadata_{meta_field}_json"] = json.dumps(
                        meta_val.model_dump()
                    )
                except TypeError as e:
                    logger.warning(
                        f"Could not serialize Pydantic metadata field {meta_field} to JSON: {e}"
                    )
                    props[f"metadata_{meta_field}_str"] = str(meta_val)
            else:
                props[f"metadata_{meta_field}"] = meta_val
    return {k: v for k, v in props.items() if v is not None}


def prepare_edge_properties_for_neo4j(edge_pydantic: Edge) -> dict[str, Any]:
    """Convert an ``Edge`` model into a property dict for Neo4j."""
    if edge_pydantic is None:
        return {}

    props: dict[str, Any] = {"id": edge_pydantic.id}

    if hasattr(edge_pydantic, "confidence") and edge_pydantic.confidence is not None:
        if isinstance(edge_pydantic.confidence, (int, float)):
            props["confidence"] = edge_pydantic.confidence
        elif hasattr(edge_pydantic.confidence, "model_dump"):
            props["confidence_json"] = json.dumps(edge_pydantic.confidence.model_dump())

    if edge_pydantic.metadata:
        for meta_field, meta_val in edge_pydantic.metadata.model_dump().items():
            if meta_val is None:
                continue
            if isinstance(meta_val, datetime):
                props[f"metadata_{meta_field}"] = meta_val.isoformat()
            elif isinstance(meta_val, Enum):
                props[f"metadata_{meta_field}"] = meta_val.value
            elif isinstance(meta_val, (list, set, dict)) or hasattr(
                meta_val, "model_dump"
            ):
                try:
                    props[f"metadata_{meta_field}_json"] = json.dumps(
                        meta_val.model_dump()
                        if hasattr(meta_val, "model_dump")
                        else meta_val
                    )
                except TypeError:
                    props[f"metadata_{meta_field}_str"] = str(meta_val)
            else:
                props[f"metadata_{meta_field}"] = meta_val
    return {k: v for k, v in props.items() if v is not None}


__all__ = [
    "prepare_node_properties_for_neo4j",
    "prepare_edge_properties_for_neo4j",
]
