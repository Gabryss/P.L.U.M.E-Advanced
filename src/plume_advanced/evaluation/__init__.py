"""Scientific evaluation support for PLUME-Advanced.

Evaluation code consumes canonical stage objects and writes machine-readable
evidence. It is deliberately separate from the production generators.
"""

from plume_advanced.evaluation.provenance import semantic_hash

EXPERIMENT_SCHEMA_VERSION = 1
MODEL_SCHEMA_VERSION = 1

from .continuity import continuity_diagnostics, longitudinal_continuity
from .network import (
    compute_network_metrics,
    network_diagnostics,
    network_sinuosity_statistics,
    sustained_uphill_diagnostics,
)
from .sections import cross_section_diagnostics, section_diagnostics, section_feature_records
from .serialization import diagnostics_to_json

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "MODEL_SCHEMA_VERSION",
    "semantic_hash",
    "compute_network_metrics",
    "network_diagnostics",
    "network_sinuosity_statistics",
    "sustained_uphill_diagnostics",
    "section_diagnostics",
    "cross_section_diagnostics",
    "section_feature_records",
    "continuity_diagnostics",
    "longitudinal_continuity",
    "diagnostics_to_json",
]
