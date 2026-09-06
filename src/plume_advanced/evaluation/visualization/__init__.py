"""Diagnostic figures built from saved Stage B/C evaluation artifacts."""

from .dashboard import dashboard_payload, render_diagnostic_dashboard
from .emplacement import render_emplacement_phase_activity

__all__ = ["dashboard_payload", "render_diagnostic_dashboard", "render_emplacement_phase_activity"]
