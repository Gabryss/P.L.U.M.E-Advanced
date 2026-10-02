"""Connected physical networks can retain multiple acyclic flow sinks."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from plume_advanced.stages.network_connectivity import component_labels, connect_paths


def parallel_corridors(*, blocked=False):
    """Two supplied straight passages; the middle host band may be forbidden."""
    x, y = np.arange(0., 125., 5.), np.arange(0., 65., 5.)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    rows, cols = [], []
    nx = len(x)
    for iy in range(len(y)):
        for ix in range(nx - 1):
            a = iy * nx + ix
            for dy in (-1, 0, 1):
                if not 0 <= iy + dy < len(y):
                    continue
                b = (iy + dy) * nx + ix + 1
                if blocked and (25 <= xy[a, 1] <= 35 or 25 <= xy[b, 1] <= 35):
                    continue
                rows.append(a)
                cols.append(b)
    rows, cols = np.array(rows), np.array(cols)
    weights = np.linalg.norm(xy[rows] - xy[cols], axis=1)
    paths = [list(range(iy * nx, (iy + 1) * nx)) for iy in (2, 10)]
    planner = SimpleNamespace(
        xy=xy, rows=rows, cols=cols, weights=weights,
        graph=csr_matrix((weights, (rows, cols)), shape=(len(xy), len(xy))),
        potential=120. - xy[:, 0], sources=[path[0] for path in paths],
        goals=[path[-1] for path in paths], width=4., step=5., independent_length=10.,
        cost=np.ones(len(xy)), layer_ids=np.zeros(len(xy), dtype=int),
        geometry=SimpleNamespace(seed_x=0., seed_y=0., flow_x=1., flow_y=0.),
        config=SimpleNamespace(
            regional=SimpleNamespace(minimum_branch_length_m=50., maximum_branch_length_m=105.),
            quality=SimpleNamespace(maximum_turn_degrees=55., minimum_bend_radius_widths=.65)),
    )
    return planner, paths


def test_connectivity_uses_undirected_passages_not_reverse_lava_flow():
    labels = component_labels(range(6), [(0, 1), (2, 1), (1, 3), (1, 4)])
    assert len(set(labels.values())) == 2
    assert labels[0] == labels[2] == labels[4]
    assert labels[5] != labels[0]
    assert labels == component_labels(reversed(range(6)), [(1, 4), (1, 3), (2, 1), (0, 1)])


def test_connection_joins_both_systems_and_preserves_both_outlets():
    p, paths = parallel_corridors()
    before = [path[:] for path in paths]
    graph_before = p.graph.copy()
    additions, audit = connect_paths(p, paths)
    assert paths == before
    assert (p.graph != graph_before).nnz == 0
    assert len(additions) == 1
    assert audit['initial_components'] == 2 and audit['final_components'] == 1
    assert audit['expanded_states'] <= audit['state_limit']
    assert audit['searches'] <= 64
    nodes = {cell for path in paths + additions for cell in path}
    edges = {(a, b) for path in paths + additions for a, b in zip(path, path[1:])}
    assert len(set(component_labels(nodes, edges).values())) == 1
    path = additions[0]
    assert path[0] not in p.sources and path[-1] not in p.goals
    assert path[0] in paths[0] and path[-1] in paths[1] or path[0] in paths[1] and path[-1] in paths[0]
    assert all(p.graph[a, b] > 0 and p.potential[a] > p.potential[b] for a, b in edges)
    assert not set(p.goals) & {a for a, b in edges}
    # The new route is a fork and confluence, not a metadata-only join.
    assert sum(a == path[0] for a, b in edges) == 2
    assert sum(b == path[-1] for a, b in edges) == 2


def test_connection_and_work_audit_replay_exactly():
    p, paths = parallel_corridors()
    assert connect_paths(p, paths) == connect_paths(p, paths)


def test_blocked_host_fails_without_an_unvalidated_straight_connector():
    p, paths = parallel_corridors(blocked=True)
    before = [path[:] for path in paths]
    messages = []
    for _ in range(2):
        with pytest.raises(ValueError, match="connectivity repair exhausted") as error:
            connect_paths(p, paths)
        messages.append(str(error.value))
    assert messages[0] == messages[1]
    assert paths == before


def test_already_connected_graph_requires_no_new_connection():
    p, paths = parallel_corridors()
    added, _ = connect_paths(p, paths)
    repeated, audit = connect_paths(p, paths + added)
    assert repeated == []
    assert audit['initial_components'] == audit['final_components'] == 1
    assert audit['searches'] == audit['expanded_states'] == audit['state_limit'] == 0


def test_rejected_geometry_is_replaced_by_another_connection():
    p, paths = parallel_corridors()
    proposed = []

    def validate(path):
        proposed.append(path)
        return len(proposed) > 1

    added, audit = connect_paths(p, paths, validate=validate)
    assert len(proposed) == 2
    assert proposed[0] != proposed[1]
    assert added == [proposed[1]]
    assert audit['geometry_rejections'] == 1
    assert audit['expanded_states'] <= audit['state_limit']


def test_no_connection_is_kept_when_all_metric_checks_fail():
    p, paths = parallel_corridors()
    before = [path[:] for path in paths]
    proposals = []

    def reject(path):
        proposals.append(path)
        return False

    with pytest.raises(ValueError, match="connectivity repair exhausted"):
        connect_paths(p, paths, validate=reject)
    assert 0 < len(proposals) <= 64
    assert paths == before


def test_three_components_are_joined_without_losing_a_terminal():
    p, paths = parallel_corridors()
    paths.insert(1, list(range(6 * 25, 7 * 25)))
    p.sources = [path[0] for path in paths]
    p.goals = [path[-1] for path in paths]
    added, audit = connect_paths(p, paths)
    assert len(added) == 2
    assert audit['initial_components'] == 3 and audit['final_components'] == 1
    assert [c['remaining_components'] for c in audit['connections']] == [2, 1]
    assert audit['expanded_states'] <= audit['state_limit']
    assert not set(p.goals) & {cell for path in paths + added for cell in path[:-1]}


def layered_corridors(*, layers=2, ramps=True):
    """Vertically separated passages with identical plan projections."""
    p, _ = parallel_corridors()
    base_xy, base_rows, base_cols = p.xy, p.rows, p.cols
    n = len(base_xy)
    p.xy = np.tile(base_xy, (layers, 1))
    p.layer_ids = np.repeat(np.arange(layers), n)
    p.depths = 5. + np.arange(layers) * 8.
    p.config.layers = SimpleNamespace(passage_height_m=3., minimum_rock_m=2.)
    p.sample_host = lambda xy: np.zeros((*np.asarray(xy).shape[:-1], 3))
    p.cost_fields = np.stack([np.ones(n) * (layer + 1) for layer in range(layers)])
    rows = [base_rows + layer * n for layer in range(layers)]
    cols = [base_cols + layer * n for layer in range(layers)]
    if ramps:
        for layer in range(layers - 1):
            rows.append(np.array([2 * 25 + ix + layer * n for ix in range(5, 10)]))
            cols.append(rows[-1] + 10 + n)
    p.rows, p.cols = np.concatenate(rows), np.concatenate(cols)
    p.weights = np.linalg.norm(p.xy[p.rows] - p.xy[p.cols], axis=1)
    p.feeder_penalties = lambda _: np.ones_like(p.weights)
    p.graph = csr_matrix((p.weights, (p.rows, p.cols)), shape=(len(p.xy), len(p.xy)))
    p.potential = 120. - p.xy[:, 0] + (layers - 1 - p.layer_ids) * 200.
    paths = [list(range(2 * 25 + layer * n, 3 * 25 + layer * n)) for layer in range(layers)]
    p.sources, p.goals = [q[0] for q in paths], [q[-1] for q in paths]
    return p, paths


@pytest.mark.parametrize('layers', [2, 3])
def test_layer_repair_uses_existing_ramps_without_merging_xy_crossings(layers):
    import json

    p, paths = layered_corridors(layers=layers)
    before = [path[:] for path in paths]
    checked = []

    def validate(path):
        checked.append(path)
        return True

    additions, audit = connect_paths(p, paths, validate=validate)
    assert audit['initial_components'] == layers
    assert audit['final_components'] == 1
    assert additions == checked and len(additions) == layers - 1
    assert paths == before
    assert all(c['end_layer'] == c['start_layer'] + 1 and c['ramp_count'] == 1
               for c in audit['connections'])
    assert audit['expanded_states'] <= audit['state_limit']
    assert connect_paths(p, paths, validate=lambda _: True) == (additions, audit)
    json.dumps(audit)  # Audit values must remain serializable native scalars.
    edges = {edge for path in paths + additions for edge in zip(path, path[1:])}
    assert all(p.graph[a, b] > 0 and p.potential[a] > p.potential[b] for a, b in edges)
    assert not set(p.goals) & {a for a, b in edges}
    assert not set(p.sources) & {b for a, b in edges}


def test_layer_repair_requires_metric_inspection():
    p, paths = layered_corridors()
    with pytest.raises(ValueError, match='requires metric geometry validation'):
        connect_paths(p, paths)


def test_plan_overlap_cannot_replace_a_missing_descending_ramp():
    p, paths = layered_corridors(ramps=False)
    with pytest.raises(ValueError, match='connectivity repair exhausted'):
        connect_paths(p, paths, validate=lambda _: True)


def test_layered_ramp_rejected_by_metric_inspection_is_not_retained():
    p, paths = layered_corridors()
    proposed = []

    def reject(path):
        proposed.append(path)
        return False

    with pytest.raises(ValueError, match='connectivity repair exhausted'):
        connect_paths(p, paths, validate=reject)
    assert 0 < len(proposed) <= 64


def test_clearance_uses_layer_costs_and_swept_ramp_elevation():
    from plume_advanced.stages.network_connectivity_space import connection_space

    p, paths = layered_corridors()
    positions, distance, costs = connection_space(p, [paths[0]])
    np.testing.assert_array_equal(costs, p.cost_fields.ravel())
    assert np.all(distance[paths[0]] == 0)
    assert np.all(distance[paths[1]] > 1.1 * p.width)
    # At the ramp midpoint, both levels are closer to the swept ramp than
    # either is to the ramp's endpoints alone.
    ramp = [paths[0][5], paths[1][15]]
    _, ramp_distance, _ = connection_space(p, [ramp])
    for cell in [paths[0][10], paths[1][10]]:
        endpoint_distance = np.linalg.norm(positions[ramp] - positions[cell], axis=1).min()
        assert ramp_distance[cell] < endpoint_distance / 2


def _cycle_rank(paths):
    edges = {edge for path in paths for edge in zip(path, path[1:])}
    nodes = {cell for path in paths for cell in path}
    components = len(set(component_labels(nodes, edges).values()))
    return len(edges) - len(nodes) + components


def test_optional_connections_create_bounded_alternate_routes_and_replay():
    p, paths = parallel_corridors()
    base, _ = connect_paths(p, paths)
    paths += base
    original = [path[:] for path in paths]
    added, audit = connect_paths(p, paths, extra_connections=2, validate=lambda _: True)
    assert 0 < len(added) <= 2
    assert len(added) == audit['extra_accepted']
    assert audit['initial_components'] == audit['final_components'] == 1
    assert _cycle_rank(paths + added) - _cycle_rank(paths) == len(added)
    assert paths == original
    assert audit['searches'] <= 128 and audit['expanded_states'] <= audit['state_limit'] == 16384
    assert not set(p.goals) & {a for path in paths + added for a in path[:-1]}
    assert connect_paths(p, paths, extra_connections=2, validate=lambda _: True) == (added, audit)


def test_optional_links_are_skipped_without_invalidating_connected_network():
    p, paths = parallel_corridors()
    base, _ = connect_paths(p, paths)
    paths += base
    original = [path[:] for path in paths]
    added, audit = connect_paths(p, paths, extra_connections=2, validate=lambda _: False)
    assert added == [] and paths == original
    assert audit['extra_accepted'] == 0 and audit['final_components'] == 1
    assert len(audit['extra_attempts']) == 2
    assert all(not a['accepted'] for a in audit['extra_attempts'])
    assert audit['geometry_rejections'] > 0


def test_optional_connections_cannot_duplicate_one_unbranched_passage():
    p, paths = parallel_corridors()
    paths = paths[:1]
    p.sources, p.goals = [paths[0][0]], [paths[0][-1]]
    added, audit = connect_paths(p, paths, extra_connections=2, validate=lambda _: True)
    assert not added and not audit['searches']
    assert audit['final_components'] == 1


def test_extra_links_alternate_same_level_and_descending_searches():
    p, paths = layered_corridors()
    base, _ = connect_paths(p, paths, validate=lambda _: True)
    paths += base
    added, audit = connect_paths(p, paths, extra_connections=2, validate=lambda _: True)
    assert [a['kind'] for a in audit['extra_attempts']] == ['descending_ramp', 'same_layer']
    assert not audit['extra_attempts'][1]['accepted']  # Coincident plan views are separate levels.
    assert len(added) == audit['extra_accepted'] == 1
    assert audit['connections'][0]['ramp_count'] == 1
    assert _cycle_rank(paths + added) == _cycle_rank(paths) + 1


@pytest.mark.parametrize('value', [-1, 9, True, 1.5])
def test_invalid_extra_budget_is_rejected_before_search(value):
    p, paths = parallel_corridors()
    with pytest.raises(ValueError, match='extra_connections'):
        connect_paths(p, paths, extra_connections=value)


def test_optional_links_need_geometry_validation_even_in_single_layer():
    p, paths = parallel_corridors()
    with pytest.raises(ValueError, match='requires metric geometry validation'):
        connect_paths(p, paths, extra_connections=1)
