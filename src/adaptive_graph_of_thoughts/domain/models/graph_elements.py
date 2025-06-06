import datetime
import uuid  # For generating default IDs
from enum import Enum
from typing import Any, Optional  # Ensure Set is NOT imported here

from pydantic import BaseModel, Field, field_serializer, field_validator

from .common import (
    CertaintyScore,
    ConfidenceVector,
    EpistemicStatus,
    ImpactScore,
    TimestampedModel,
)


# --- Enums for Node and Edge Types ---
class NodeType(str, Enum):
    """P1.11 (T=node types) & various P1.x referring to specific node types"""

    ROOT = "root"  # P1.1 (n0)
    TASK_UNDERSTANDING = "task_understanding"  # P1.1 (n0 label)
    DECOMPOSITION_DIMENSION = "decomposition_dimension"  # P1.2
    HYPOTHESIS = "hypothesis"  # P1.3
    EVIDENCE = "evidence"  # P1.4
    PLACEHOLDER_GAP = "placeholder_gap"  # P1.15
    INTERDISCIPLINARY_BRIDGE = "interdisciplinary_bridge"  # P1.8 (IBN)
    # Add more specific types as needed, e.g., Claim, Argument, Question
    RESEARCH_QUESTION = "research_question"


class EdgeType(str, Enum):
    """P1.10, P1.24, P1.25"""

    # Basic Types (P1.10)
    DECOMPOSITION_OF = "decomposition_of"  # Connects dimension to root
    GENERATES_HYPOTHESIS = "generates_hypothesis"  # Connects dimension to hypothesis
    HAS_SUBQUESTION = "has_subquestion"  # Connects node to a research question

    CORRELATIVE = "correlative"  # (⇢)
    SUPPORTIVE = "supportive"  # (↑)
    CONTRADICTORY = "contradictory"  # (⊥)
    PREREQUISITE = "prerequisite"  # (⊢)
    GENERALIZATION = "generalization"  # (⊇)
    SPECIALIZATION = "specialization"  # (⊂)
    ASSOCIATIVE = "associative"  # General association if not one of above
    EXAMPLE_OF = "example_of"
    RELEVANT_TO = "relevant_to"

    # Causal Types (P1.24)
    CAUSES = "causes"  # (→)
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    PREVENTS = "prevents"
    INFLUENCES_POSITIVELY = "influences_positively"
    INFLUENCES_NEGATIVELY = "influences_negatively"
    COUNTERFACTUAL_TO = (
        "counterfactual_to"  # If X had not happened, Y would not have happened
    )
    CONFOUNDED_BY = (
        "confounded_by"  # Edge indicating a confounding relationship to another node
    )

    # Temporal Types (P1.25)
    TEMPORAL_PRECEDES = "temporal_precedes"  # (≺)
    TEMPORAL_FOLLOWS = "temporal_follows"
    COOCCURS_WITH = "cooccurs_with"  # Happens at the same time
    OVERLAPS_WITH = "overlaps_with"  # Temporal overlap
    CYCLIC_RELATIONSHIP = "cyclic_relationship"
    DELAYED_EFFECT_OF = "delayed_effect_of"
    SEQUENTIAL_DEPENDENCY = "sequential_dependency"

    # Interdisciplinary Bridge Node connections (P1.8)
    IBN_SOURCE_LINK = "ibn_source_link"
    IBN_TARGET_LINK = "ibn_target_link"

    # Hyperedge specific (virtual edge type if representing hyperedges in a DiGraph)
    HYPEREDGE_COMPONENT = "hyperedge_component"

    OTHER = "other"


# --- Metadata Sub-Models (aligning with P1.12) ---
class FalsificationCriteria(BaseModel):  # P1.16
    description: str
    testable_conditions: list[str] = Field(default_factory=list)
    # potential_null_results: Optional[str] = None


class BiasFlag(BaseModel):  # P1.17
    bias_type: str  # e.g., "Confirmation Bias", "Selection Bias"
    description: Optional[str] = None
    assessment_stage_id: Optional[str] = None  # Stage where bias was flagged
    mitigation_suggested: Optional[str] = None
    severity: Optional[str] = Field(default="low", examples=["low", "medium", "high"])


class RevisionRecord(BaseModel):
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    user_or_process: str  # Who/what made the change
    action: str  # e.g., "created", "updated_confidence", "merged", "pruned"
    changes_made: dict[
        str, Any
    ]  # e.g., {"confidence.empirical_support": {"old": 0.5, "new": 0.7}}
    reason: Optional[str] = None


class Plan(BaseModel):  # P1.3 (for hypotheses)
    type: str  # e.g., "literature_review", "experiment", "simulation", "data_analysis"
    description: str
    estimated_cost: Optional[float] = Field(default=0.0, ge=0.0)  # Abstract cost unit
    estimated_duration: Optional[float] = Field(
        default=0.0, ge=0.0
    )  # Abstract time unit
    required_resources: list[str] = Field(default_factory=list)
    # status: str = Field(default="pending", examples=["pending", "in_progress", "completed", "failed"])


class InterdisciplinaryInfo(BaseModel):  # P1.8 (metadata for IBNs)
    source_disciplines: set[str] = Field(default_factory=set)
    target_disciplines: set[str] = Field(default_factory=set)
    bridging_concept: Optional[str] = None


class CausalMetadata(BaseModel):  # P1.24 (metadata for causal edges)
    strength: Optional[CertaintyScore] = None  # Strength of causal claim
    mechanism_description: Optional[str] = None
    confounders_identified: list[str] = Field(default_factory=list)
    experimental_support: Optional[bool] = None  # True if supported by experiment
    # counterfactual_reasoning: Optional[str] = None


class TemporalMetadata(BaseModel):  # P1.25 (metadata for temporal edges)
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    duration_seconds: Optional[float] = None
    delay_seconds: Optional[float] = None  # For delayed effects
    pattern_type: Optional[str] = Field(
        default=None, examples=["linear", "cyclic", "event_driven"]
    )
    # frequency: Optional[str] = None # For cyclic patterns


class InformationTheoreticMetrics(BaseModel):  # P1.27
    entropy: Optional[float] = None
    information_gain: Optional[float] = None  # Relative to parent or previous state
    kl_divergence_from_prior: Optional[float] = None
    # mdl_complexity: Optional[float] = None # Minimum Description Length


class StatisticalPower(BaseModel):  # P1.26 (for evidence nodes)
    value: CertaintyScore = Field(default=0.8)  # Default to 80% power if not specified
    sample_size: Optional[int] = None
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    method_description: Optional[str] = None  # How power was calculated/estimated


class Attribution(BaseModel):  # P1.29
    source_id: Optional[str] = None  # ID of the original source, if any
    contributor: Optional[str] = None  # User or process ID
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    role: Optional[str] = Field(
        default="author", examples=["author", "curator", "validator"]
    )


# --- Core Graph Element Models ---


class NodeMetadata(TimestampedModel):  # Aligns with P1.12 for nodes
    description: Optional[str] = None
    query_context: Optional[str] = None  # Verbatim query or context for this node, P1.6
    source_description: Optional[str] = (
        None  # P1.0, P1.1 etc. Source document/rule for this parameter
    )
    epistemic_status: EpistemicStatus = Field(default=EpistemicStatus.UNKNOWN)
    disciplinary_tags: set[str] = Field(default_factory=set)  # P1.8, P1.12
    falsification_criteria: Optional[FalsificationCriteria] = None  # P1.16, P1.12
    bias_flags: list[BiasFlag] = Field(default_factory=list)  # P1.17, P1.12
    revision_history: list[RevisionRecord] = Field(default_factory=list)  # P1.12
    layer_id: Optional[str] = None  # P1.23, P1.12
    # topology_metrics: Optional[dict[str, float]] = None # P1.22, P1.12 (calculated dynamically)
    statistical_power: Optional[StatisticalPower] = (
        None  # P1.26, P1.12 (esp. for Evidence nodes)
    )
    information_metrics: Optional[InformationTheoreticMetrics] = None  # P1.27, P1.12
    impact_score: Optional[ImpactScore] = Field(
        default=0.1
    )  # P1.28, P1.12 default to low impact
    attribution: list[Attribution] = Field(default_factory=list)  # P1.29, P1.12
    plan: Optional[Plan] = None  # P1.3 (for Hypothesis nodes)
    interdisciplinary_info: Optional[InterdisciplinaryInfo] = None  # For IBNs
    # additional_properties: Dict[str, Any] = Field(default_factory=dict) # For extensibility

    # For knowledge gap nodes (P1.15)
    is_knowledge_gap: bool = False
    research_questions_generated: list[str] = Field(default_factory=list)


class Node(TimestampedModel):
    id: str = Field(default_factory=lambda: f"node-{uuid.uuid4()}")
    label: str = Field(..., min_length=1)
    type: NodeType
    confidence: ConfidenceVector = Field(default_factory=ConfidenceVector)
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)

    @field_serializer("confidence")
    def serialize_confidence_to_list(self, v: ConfidenceVector, _info):
        """
        Converts a ConfidenceVector to a list representation for serialization.
        
        Args:
            v: The ConfidenceVector instance to serialize.
            _info: Additional serialization context (unused).
        
        Returns:
            A list representing the confidence vector values.
        """
        return v.to_list()

    # To allow Node instances to be added to sets or used as dict keys
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def update_confidence(
        self,
        new_confidence: ConfidenceVector,
        updated_by: str,
        reason: Optional[str] = None,
    ):
        old_confidence_dict = self.confidence.model_dump()
        self.confidence = new_confidence
        self.metadata.revision_history.append(
            RevisionRecord(
                user_or_process=updated_by,
                action="update_confidence",
                changes_made={
                    "confidence": {
                        "old": old_confidence_dict,
                        "new": new_confidence.model_dump(),
                    }
                },
                reason=reason,
            )
        )
        self.touch()


class EdgeMetadata(TimestampedModel):  # Aligns with P1.12 for edges
    description: Optional[str] = None
    # confidence_on_relationship: Optional[CertaintyScore] = None # If edge has its own certainty distinct from nodes
    weight: Optional[float] = Field(default=1.0)  # For weighted graph algorithms
    causal_metadata: Optional[CausalMetadata] = None  # P1.24
    temporal_metadata: Optional[TemporalMetadata] = None  # P1.25
    attribution: list[Attribution] = Field(default_factory=list)  # P1.29
    revision_history: list[RevisionRecord] = Field(default_factory=list)
    # additional_properties: dict[str, Any] = Field(default_factory=dict)


class Edge(TimestampedModel):
    id: str = Field(default_factory=lambda: f"edge-{uuid.uuid4()}")
    source_id: str
    target_id: str
    type: EdgeType
    # Edge specific confidence separate from node confidences, as per P1.12 for Edges
    # P1.10 also implies edges can have confidence.
    confidence: Optional[CertaintyScore] = Field(default=0.7)
    metadata: EdgeMetadata = Field(
        default_factory=EdgeMetadata
    )  # To allow Edge instances to be added to sets or used as dict keys (e.g. by source, target, type)

    def __hash__(self):
        """
        Returns a hash value based on the edge's ID, source ID, target ID, and type.
        """
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


class HyperedgeMetadata(TimestampedModel):  # Aligns with P1.9 & P1.12
    description: Optional[str] = None
    relationship_descriptor: str  # Describes the N-ary relationship
    attribution: list[Attribution] = Field(default_factory=list)  # P1.29
    revision_history: list[RevisionRecord] = Field(default_factory=list)
    layer_id: Optional[str] = None  # P1.23
    # additional_properties: dict[str, Any] = Field(default_factory=dict)


class Hyperedge(TimestampedModel):  # P1.9
    id: str = Field(default_factory=lambda: f"hyperedge-{uuid.uuid4()}")
    # A hyperedge connects a set of nodes. |E_h| > 2 is typical but can be 2 for typed N-ary.
    node_ids: set[str] = Field(..., min_length=2)
    # Confidence in the hyper-relationship itself
    confidence_vector: ConfidenceVector = Field(
        default_factory=ConfidenceVector
    )  # P1.9: "Assign confidence vector C"
    metadata: HyperedgeMetadata = Field(...)

    @field_validator("node_ids")
    def check_min_nodes(cls, v: set[str]) -> set[str]:  # Ensure this is lowercase 'set'
        """
        Validates that a hyperedge connects at least two nodes.
        
        Raises:
            ValueError: If fewer than two node IDs are provided.
        """
        if (
            len(v) < 2
        ):  # P1.9 stated |Eh| > 2, but often hyperedges also model pairs with specific semantics
            # Let's stick to at least 2 for now, meaning it's at least a binary relation.
            # The "> 2" might refer to "non-additive" aspect implying more than just a sum of pairwise.
            raise ValueError("Hyperedge must connect at least 2 nodes.")
        return v

    def __hash__(self):
        # Order of node_ids should not matter for hash
        return hash((self.id, tuple(sorted(self.node_ids))))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Hyperedge):
            return self.id == other.id and self.node_ids == other.node_ids
        return False
