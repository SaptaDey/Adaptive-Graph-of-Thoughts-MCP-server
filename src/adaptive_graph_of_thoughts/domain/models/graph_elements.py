import uuid
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Optional, Set

from pydantic import BaseModel, Field

from .common import (
    CertaintyScore,
    ConfidenceVector,
    EpistemicStatus,
    ImpactScore,
    TimestampedModel,
)


# --- Enumerations ---
class NodeType(str, Enum):
    ROOT = "root"
    TASK_UNDERSTANDING = "task_understanding"
    DECOMPOSITION_DIMENSION = "decomposition_dimension"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    PLACEHOLDER_GAP = "placeholder_gap"
    INTERDISCIPLINARY_BRIDGE = "interdisciplinary_bridge"
    RESEARCH_QUESTION = "research_question"


class EdgeType(str, Enum):
    DECOMPOSITION_OF = "decomposition_of"
    GENERATES_HYPOTHESIS = "generates_hypothesis"
    HAS_SUBQUESTION = "has_subquestion"
    CORRELATIVE = "correlative"
    SUPPORTIVE = "supportive"
    CONTRADICTORY = "contradictory"
    PREREQUISITE = "prerequisite"
    GENERALIZATION = "generalization"
    SPECIALIZATION = "specialization"
    ASSOCIATIVE = "associative"
    EXAMPLE_OF = "example_of"
    RELEVANT_TO = "relevant_to"
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    PREVENTS = "prevents"
    INFLUENCES_POSITIVELY = "influences_positively"
    INFLUENCES_NEGATIVELY = "influences_negatively"
    COUNTERFACTUAL_TO = "counterfactual_to"
    CONFOUNDED_BY = "confounded_by"
    TEMPORAL_PRECEDES = "temporal_precedes"
    TEMPORAL_FOLLOWS = "temporal_follows"
    COOCCURS_WITH = "cooccurs_with"
    OVERLAPS_WITH = "overlaps_with"
    CYCLIC_RELATIONSHIP = "cyclic_relationship"
    DELAYED_EFFECT_OF = "delayed_effect_of"
    SEQUENTIAL_DEPENDENCY = "sequential_dependency"
    IBN_SOURCE_LINK = "ibn_source_link"
    IBN_TARGET_LINK = "ibn_target_link"
    HYPEREDGE_COMPONENT = "hyperedge_component"
    OTHER = "other"


# --- Metadata models ---
class FalsificationCriteria(BaseModel):
    description: str
    testable_conditions: List[str] = []


class BiasFlag(BaseModel):
    bias_type: str
    description: str = ""
    assessment_stage_id: str = ""
    mitigation_suggested: str = ""
    severity: str = "low"


class RevisionRecord(BaseModel):
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    user_or_process: str
    action: str
    changes_made: Dict[str, Any] = {}
    reason: str = ""


class Plan(BaseModel):
    type: str
    description: str
    estimated_cost: float = 0.0
    estimated_duration: float = 0.0
    required_resources: List[str] = []


class InterdisciplinaryInfo(BaseModel):
    source_disciplines: Set[str] = set()
    target_disciplines: Set[str] = set()
    bridging_concept: str = ""


class TemporalMetadata(BaseModel):
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    delay_seconds: float = 0.0
    pattern_type: str = ""


class InformationTheoreticMetrics(BaseModel):
    entropy: float = 0.0
    information_gain: float = 0.0
    kl_divergence_from_prior: float = 0.0


class StatisticalPower(BaseModel):
    value: CertaintyScore = 0.8
    sample_size: int = 0
    effect_size: float = 0.0
    p_value: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    method_description: str = ""


class CausalMetadata(BaseModel):
    """Lightweight causal metadata placeholder."""

    description: str = ""
    causal_strength: float = 0.0
    evidence_source: str = ""


class Attribution(BaseModel):
    source_id: str = ""
    contributor: str = ""
    timestamp: str = ""
    role: str = Field(default="author")


# --- Core graph element models ---
class NodeMetadata(TimestampedModel):
    description: str = ""
    query_context: str = ""
    source_description: str = ""
    epistemic_status: EpistemicStatus = EpistemicStatus.UNKNOWN
    disciplinary_tags: str = ""
    layer_id: str = ""
    impact_score: float = 0.1
    is_knowledge_gap: bool = False
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doi: str = ""
    authors: str = ""
    publication_date: str = ""
    revision_history: List[RevisionRecord] = Field(default_factory=list)


class Node(TimestampedModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    type: NodeType = NodeType.HYPOTHESIS
    confidence: ConfidenceVector = Field(default_factory=ConfidenceVector)
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def update_confidence(self, new_confidence: ConfidenceVector, updated_by: str, reason: Optional[str] = None) -> None:
        old_conf = self.confidence.model_dump()
        self.confidence = new_confidence
        self.metadata.revision_history.append(
            RevisionRecord(
                user_or_process=updated_by,
                action="update_confidence",
                changes_made={"confidence": {"old": old_conf, "new": new_confidence.model_dump()}},
                reason=reason,
            )
        )
        self.touch()


class EdgeMetadata(TimestampedModel):
    description: str = ""
    weight: float = 1.0


class Edge(TimestampedModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    type: EdgeType = EdgeType.SUPPORTIVE
    confidence: float = 0.7
    metadata: EdgeMetadata = Field(default_factory=EdgeMetadata)

    def __hash__(self) -> int:
        return hash((self.id, self.source_id, self.target_id, self.type))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Edge):
            return (
                self.id == other.id
                and self.source_id == other.source_id
                and self.target_id == other.target_id
                and self.type == other.type
            )
        return False


class HyperedgeMetadata(TimestampedModel):
    description: str = ""
    relationship_descriptor: str = ""
    layer_id: str = ""


class Hyperedge(TimestampedModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_ids: Set[str] = set()
    confidence_vector: float = 0.5
    metadata: HyperedgeMetadata = Field(default_factory=HyperedgeMetadata)

    def __hash__(self) -> int:
        return hash((self.id, tuple(sorted(self.node_ids))))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Hyperedge) and self.id == other.id and self.node_ids == other.node_ids


@dataclass(frozen=True)
class GraphElement:
    node_id: uuid.UUID
    label: str = ""
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.node_id, uuid.UUID):
            raise TypeError("node_id must be a UUID instance")
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if len(self.label) > 1000:
            raise ValueError("label cannot exceed 1000 characters")
        if not isinstance(self.weight, (int, float)):
            raise TypeError("weight must be a number")

    def __hash__(self) -> int:
        return hash((self.node_id, self.label, self.weight))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GraphElement):
            return (
                self.node_id == other.node_id
                and self.label == other.label
                and self.weight == other.weight
            )
        return False

    def __repr__(self) -> str:
        return f"GraphElement(node_id={self.node_id!r}, label={self.label!r}, weight={self.weight!r})"

    __str__ = __repr__

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": str(self.node_id), "label": self.label, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphElement":
        return cls(node_id=uuid.UUID(data["node_id"]), label=data["label"], weight=data["weight"])

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "GraphElement":
        import json

        return cls.from_dict(json.loads(data))


class Graph(BaseModel):
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)

    def add_node(self, node_id: str, **kwargs) -> None:
        self.nodes[node_id] = Node(id=node_id, **kwargs)

    def add_edge(self, source: str, target: str, **kwargs) -> None:
        self.edges.append(Edge(source_id=source, target_id=target, **kwargs))

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def has_edge(self, source: str, target: str) -> bool:
        return any(e.source_id == source and e.target_id == target for e in self.edges)
