"""Scientific evaluation support for PLUME-Advanced.

Evaluation code consumes canonical stage objects and writes machine-readable
evidence. It is deliberately separate from the production generators.
"""

from plume_advanced.evaluation.provenance import semantic_hash

EXPERIMENT_SCHEMA_VERSION = 1
MODEL_SCHEMA_VERSION = 1

from .continuity import continuity_diagnostics, longitudinal_continuity
from .serialization import diagnostics_to_json

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "MODEL_SCHEMA_VERSION",
    "semantic_hash",
    "continuity_diagnostics",
    "longitudinal_continuity",
    "diagnostics_to_json",
]
