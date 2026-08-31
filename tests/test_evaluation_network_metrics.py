import numpy as np

from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNode,
    CavePoint,
    CaveSegment,
)


def _network(edges: list[tuple[int, int]], node_count: int) -> CaveNetwork:
    nodes = tuple(
        CaveNode(
            index,
            float(index),
            0.0,
            float(index),
            0.0,
            "entry" if index == 0 else "exit" if index == node_count - 1 else "junction",
        )
        for index in range(node_count)
    )
    segments = []
    for segment_id, (start, end) in enumerate(edges):
        segments.append(
            CaveSegment(
                segment_id=segment_id,
                start_node_id=start,
                end_node_id=end,
                kind="backbone",
                z_level=0,
                points=(
                    CavePoint(
                        0,
                        float(start),
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                        1.0,
                        0.0,
                        0.0,
                        2.0,
                        flux=1.0,
                        temperature_k=1400.0,
                        age_s=0.0,
                    ),
                    CavePoint(
                        1,
                        float(end),
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                        1.0,
                        0.0,
                        1.0,
                        2.0,
                        flux=1.0,
                        temperature_k=1399.0,
                        age_s=1.0,
                    ),
                ),
                metadata={},
            )
        )
    return CaveNetwork(
        config=CaveNetworkConfig(),
        nodes=nodes,
        segments=tuple(segments),
        junctions=(),
        occupancy=np.zeros((2, 2), dtype=bool),
        width_field=np.zeros((2, 2)),
        dominant_route_node_ids=tuple(range(node_count)),
        slice_along_positions=(),
        slice_channel_counts=(),
        slice_visible_channel_counts=(),
    )


def test_known_graph_cycle_ranks() -> None:
    assert network_metrics(_network([(0, 1), (1, 2)], 3))["cyclomatic_number"] == 0
    assert network_metrics(_network([(0, 1), (1, 2), (2, 0)], 3))["cyclomatic_number"] == 1
    assert (
        network_metrics(_network([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5))[
            "cyclomatic_number"
        ]
        == 2
    )
    assert network_metrics(_network([(0, 1), (2, 3)], 4))["connected_component_count"] == 2
