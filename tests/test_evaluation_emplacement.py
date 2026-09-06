from __future__ import annotations

import numpy as np

from plume_advanced.evaluation.metrics.emplacement import emplacement_metrics
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.evaluation.visualization import render_emplacement_phase_activity
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNode,
    CavePoint,
    CaveSegment,
)


def _point(index: int, x: float, elevation: float, arc: float, slope: float = 10.0) -> CavePoint:
    return CavePoint(index, x, 0.0, elevation, slope, 10.0, 1.0, 0.0, arc, 2.0, flux=1.0)


def _multi_phase_network() -> CaveNetwork:
    nodes = (
        CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"),
        CaveNode(1, 10.0, 0.0, 10.0, 0.0, "junction"),
        CaveNode(2, 20.0, 0.0, 20.0, 0.0, "junction"),
        CaveNode(3, 30.0, 0.0, 30.0, 0.0, "exit"),
        CaveNode(4, 20.0, 5.0, 20.0, 5.0, "terminal"),
    )
    segments = (
        CaveSegment(
            0, 0, 1, "backbone", 0,
            (_point(0, 0.0, 20.0, 0.0), _point(1, 10.0, 19.0, 10.0)),
            {"lobe_path_id": "trunk", "emplacement_phase_count": 4, "birth_phase": 0, "death_phase": 3,
             "formation_state": "persistent_arterial", "branch_order": 0, "new_path": True,
             "initial_flux": 1.0, "parent_flux_before_split": 1.0},
        ),
        CaveSegment(
            1, 1, 2, "anastomosis", 1,
            (_point(0, 10.0, 19.0, 0.0), _point(1, 15.0, 20.0, 5.0), _point(2, 20.0, 18.0, 10.0)),
            {"lobe_path_id": "lobe_0", "emplacement_phase_count": 4, "birth_phase": 1, "death_phase": 2,
             "formation_state": "coalesced", "branch_order": 1, "new_path": True,
             "initial_flux": 0.4, "parent_flux_before_split": 1.0, "coalescence_returned_flux": 0.2,
             "loop_mechanism": "reconnect", "crossing_group_id": "cross_a", "coalescence_id": "coal_a"},
        ),
        CaveSegment(
            2, 2, 3, "abandoned_lobe", -1,
            (_point(0, 20.0, 18.0, 0.0), _point(1, 30.0, 17.0, 10.0)),
            {"lobe_path_id": "lobe_1", "emplacement_phase_count": 4, "birth_phase": 2, "death_phase": 3,
             "formation_state": "thermally_abandoned", "branch_order": 2, "reoccupied_path": True,
             "initial_flux": 0.2, "crossing_group_id": "cross_a"},
        ),
        CaveSegment(
            3, 1, 4, "stalled_lobe", 0,
            (_point(0, 10.0, 19.0, 0.0), _point(1, 20.0, 18.0, 10.0)),
            {"lobe_path_id": "lobe_2", "emplacement_phase_count": 4, "birth_phase": 1, "death_phase": 1,
             "formation_state": "stranded", "branch_order": 1, "reoccupied_path": True,
             "initial_flux": 0.1},
        ),
    )
    return CaveNetwork(
        config=CaveNetworkConfig(), nodes=nodes, segments=segments, junctions=(),
        occupancy=np.zeros((2, 2), dtype=bool), width_field=np.zeros((2, 2)),
        dominant_route_node_ids=(0, 1, 2, 3), slice_along_positions=(),
        slice_channel_counts=(), slice_visible_channel_counts=(),
    )


def test_multi_phase_metrics_cover_flux_outcomes_and_hierarchy() -> None:
    report = emplacement_metrics(_multi_phase_network())

    assert report["available"] is True
    assert report["phase_count"] == 4
    assert report["phase_activity"][1]["active_path_count"] == 3
    assert report["flux_budget"]["initial_flux"] == 1.7
    assert report["flux_budget"]["returned_flux"] == 0.2
    assert report["path_classification"]["new_path_share"] == 0.5
    assert report["path_classification"]["reoccupied_path_share"] == 0.5
    assert report["outcomes"]["counts"] == {"coalesced": 1, "retired": 1, "stalled": 1, "survived": 1}
    assert report["branch_hierarchy"]["order_histogram"] == {"0": 1, "1": 2, "2": 1}
    assert report["trunk_dominance"]["trunk_length_share"] == 0.75
    assert report["loop_diagnostics"]["counts"] == {"reconnect": 1}
    assert report["crossing_coalescence"]["crossing_multi_level_group_count"] == 1
    assert report["crossing_coalescence"]["coalescence_group_count"] == 1
    assert report["longitudinal"]["elevation_reversal_count"] >= 1
    assert report["longitudinal"]["slope_plausibility"] is True


def test_legacy_network_is_safe_and_marks_history_unavailable() -> None:
    network = _multi_phase_network()
    legacy = network.__class__(
        config=network.config,
        nodes=network.nodes,
        segments=tuple(segment.__class__(segment.segment_id, segment.start_node_id, segment.end_node_id,
                                         segment.kind, segment.z_level, segment.points, {}) for segment in network.segments),
        junctions=network.junctions,
        occupancy=network.occupancy,
        width_field=network.width_field,
        dominant_route_node_ids=network.dominant_route_node_ids,
        slice_along_positions=network.slice_along_positions,
        slice_channel_counts=network.slice_channel_counts,
        slice_visible_channel_counts=network.slice_visible_channel_counts,
    )
    report = network_metrics(legacy)["emplacement_diagnostics"]
    assert report["available"] is False
    assert report["phase_metadata_available"] is False
    assert report["phase_activity_available"] is False
    assert report["phase_count"] == 1
    assert report["path_classification"]["new_path_share"] is None
    assert report["loop_diagnostics"]["available"] is False


def test_phase_activity_plot_is_deterministic(tmp_path) -> None:
    network = _multi_phase_network()
    first = render_emplacement_phase_activity(network, tmp_path / "first.png")
    second = render_emplacement_phase_activity(network, tmp_path / "second.png")
    assert first.read_bytes() == second.read_bytes()
