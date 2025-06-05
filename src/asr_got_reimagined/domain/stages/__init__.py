# Makes 'stages' a sub-package.
from .base_stage import BaseStage, StageOutput

# Import stage classes using a function to prevent circular imports
def import_stages():
    from .stage_1_initialization import InitializationStage
    from .stage_2_decomposition import DecompositionStage
    from .stage_3_hypothesis import HypothesisStage
    from .stage_4_evidence import EvidenceStage
    from .stage_5_pruning_merging import PruningMergingStage
    from .stage_6_subgraph_extraction import SubgraphExtractionStage
    from .stage_7_composition import CompositionStage
    from .stage_8_reflection import ReflectionStage
    
    return {
        "InitializationStage": InitializationStage,
        "DecompositionStage": DecompositionStage,
        "HypothesisStage": HypothesisStage,
        "EvidenceStage": EvidenceStage,
        "PruningMergingStage": PruningMergingStage,
        "SubgraphExtractionStage": SubgraphExtractionStage,
        "CompositionStage": CompositionStage,
        "ReflectionStage": ReflectionStage,
    }

# Export only the base classes at import time
__all__ = [
    "BaseStage",
    "StageOutput",
    "import_stages",
]
