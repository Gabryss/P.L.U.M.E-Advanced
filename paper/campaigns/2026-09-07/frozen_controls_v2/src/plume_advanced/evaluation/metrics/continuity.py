"""Canonical compatibility exports for longitudinal continuity diagnostics.

The implementation is kept in the evaluation package because it is also used
by report-level consumers; this module is the supported metrics import path.
"""

from plume_advanced.evaluation.continuity import (
    continuity_diagnostics,
    evaluate_continuity,
    longitudinal_continuity,
)

__all__ = ["continuity_diagnostics", "evaluate_continuity", "longitudinal_continuity"]
