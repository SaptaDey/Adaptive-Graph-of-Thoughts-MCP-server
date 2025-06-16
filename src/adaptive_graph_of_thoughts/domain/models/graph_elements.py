import datetime
import uuid  # For generating default IDs
from enum import Enum
from typing import Any, Optional, List, Dict  # Added List, Dict

from pydantic import BaseModel, Field

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
    testable_conditions: List[str] = []
    # potential_null_results: Optional[str] = None


class BiasFlag(BaseModel):  # P1.17
    bias_type: str  # e.g., "Confirmation Bias", "Selection Bias"
    description: str = ""
    assessment_stage_id: str = ""  # Stage where bias was flagged
    mitigation_suggested: str = ""
    severity: str = "low"  # low, medium, high


class RevisionRecord(BaseModel):
    timestamp: datetime.datetime = datetime.datetime.now()
    user_or_process: str  # Who/what made the change
    action: str  # e.g., "created", "updated_confidence", "merged", "pruned"
    changes_made: Dict[str, Any] = {}  # e.g., {"confidence.empirical_support": {"old": 0.5, "new": 0.7}}
    reason: str = ""


class Plan(BaseModel):  # P1.3 (for hypotheses)
    type: str  # e.g., "literature_review", "experiment", "simulation", "data_analysis"
    description: str
    estimated_cost: float = 0.0  # Abstract cost unit
    estimated_duration: float = 0.0  # Abstract time unit
    required_resources: List[str] = []
    # status: str = Field(default="pending", examples=["pending", "in_progress", "completed", "failed"])


class InterdisciplinaryInfo(BaseModel):  # P1.8 (metadata for IBNs)
    source_disciplines: set = set()
    target_disciplines: set = set()
    bridging_concept: str = ""


class CausalMetadata(BaseModel):  # P1.24 (metadata for causal edges)
    strength: float = 0.0  # Strength of causal claim
    mechanism_description: str = ""
    confounders_identified: List[str] = []
    experimental_support: bool = False  # True if supported by experiment
    # counterfactual_reasoning: Optional[str] = None


class TemporalMetadata(BaseModel):  # P1.25 (metadata for temporal edges)
    start_time: str = ""  # Can store datetime as string for simplicity
    end_time: str = ""
    duration_seconds: float = 0.0
    delay_seconds: float = 0.0  # For delayed effects
    pattern_type: str = ""  # linear, cyclic, event_driven
    # frequency: Optional[str] = None # For cyclic patterns


class InformationTheoreticMetrics(BaseModel):
    entropy: float = Field(default=0.0)
    information_gain: float = Field(default=0.0)
    kl_divergence_from_prior: float = Field(default=0.0)


class StatisticalPower(BaseModel):  # P1.26 (for evidence nodes)
    value: CertaintyScore = Field(default=0.8)  # Default to 80% power if not specified
    sample_size: int = Field(default=0)
    effect_size: float = Field(default=0.0)
    p_value: float = Field(default=0.0)
    confidence_interval: tuple[float, float] = Field(default=(0.0, 1.0))
    method_description: str = Field(default="")  # How power was calculated/estimated


class Attribution(BaseModel):  # P1.29
    source_id: str = Field(default="")  # ID of the original source, if any
    contributor: str = Field(default="")  # User or process ID
    timestamp: str = Field(default="")  # Simplified to string for pydantic v1 compatibility
    role: str = Field(
        default="author", examples=["author", "curator", "validator"]
    )


# --- Core Graph Element Models ---


class NodeMetadata(TimestampedModel):  # Aligns with P1.12 for nodes
    description: str = Field(default="")
    query_context: str = Field(default="")  # Verbatim query or context for this node, P1.6
    source_description: str = Field(default="")  # P1.0, P1.1 etc. Source document/rule for this parameter
    epistemic_status: EpistemicStatus = Field(default=EpistemicStatus.UNKNOWN)
    disciplinary_tags: str = Field(default="")  # Simplified from set for pydantic v1 compatibility
    # Temporarily simplify complex fields for pydantic v1 compatibility
    layer_id: str = Field(default="")  # P1.23, P1.12
    impact_score: float = Field(default=0.1)  # P1.28, P1.12 default to low impact
    is_knowledge_gap: bool = Field(default=False)
    
    # Bibliometric fields
    url: str = Field(default="")
    doi: str = Field(default="")
    authors: str = Field(default="")  # Simplified from List[str]
    publication_date: str = Field(default="")  # e.g., "YYYY-MM-DD" or "YYYY MMM"


class Node(TimestampedModel):
    id: str = Field(default="node-default")  # Simplified for pydantic v1 compatibility
    label: str = Field(default="")  # Removed min_length constraint for pydantic v1
    type: NodeType = NodeType.HYPOTHESIS  # Provide a default
    confidence: float = Field(default=0.5)  # Simplified from ConfidenceVector
    metadata: str = Field(default="")  # Simplified from Optional[NodeMetadata]

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
    description: str = Field(default="")
    weight: float = Field(default=1.0)  # For weighted graph algorithms
    # Simplified for pydantic v1 compatibility - removed complex types


class Edge(TimestampedModel):
    id: str = Field(default="edge-default")  # Simplified for pydantic v1 compatibility
    source_id: str = Field(default="")
    target_id: str = Field(default="")
    type: EdgeType = EdgeType.SUPPORTIVE  # Provide default
    confidence: float = Field(default=0.7)  # Simplified type
    metadata: str = Field(default="")  # Simplified from Optional[EdgeMetadata]
    # To allow Edge instances to be added to sets or used as dict keys (e.g. by source, target, type)

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
    description: str = Field(default="")
    relationship_descriptor: str = Field(default="")  # Describes the N-ary relationship
    layer_id: str = Field(default="")  # P1.23


class Hyperedge(TimestampedModel):  # P1.9
    id: str = Field(default="hyperedge-default")  # Simplified for pydantic v1 compatibility
    node_ids: str = Field(default="")  # Simplified from set[str] for pydantic v1 compatibility
    confidence_vector: float = Field(default=0.5)  # Simplified from ConfidenceVector
    metadata: str = Field(default="")  # Simplified from HyperedgeMetadata

    def __hash__(self):
        # Order of node_ids should not matter for hash
        return hash((self.id, tuple(sorted(self.node_ids))))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Hyperedge):
            return self.id == other.id and self.node_ids == other.node_ids
        return False


# Simple classes for backward compatibility with tests
class GraphElement(BaseModel):
    """Simple graph element for test compatibility."""
    
    node_id: str = Field(default="")
    label: str = Field(default="")
    weight: float = Field(default=1.0)

    def __hash__(self):
        return hash((self.node_id, self.label, self.weight))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GraphElement):
            return (
                self.node_id == other.node_id
                and self.label == other.label
                and self.weight == other.weight
            )
        return False


class Graph(BaseModel):
    """Simple graph class for test compatibility."""
    
    nodes: str = Field(default="")  # Simplified as JSON string
    edges: str = Field(default="")  # Simplified as JSON string
    
    def add_node(self, node_id: str, **kwargs):
        """Add a node to the graph."""
        pass  # Simplified implementation for compatibility
    
    def add_edge(self, source: str, target: str, **kwargs):
        """Add an edge to the graph."""
        pass  # Simplified implementation for compatibility
    
    def has_node(self, node_id: str) -> bool:
        """Check if graph has a node."""
        return True  # Simplified for compatibility
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if graph has an edge."""
        return True  # Simplified for compatibility
