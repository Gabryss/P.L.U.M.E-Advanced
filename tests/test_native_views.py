import numpy as np
import pytest

from plume_advanced.evaluation.native_views import plan_material_views


def inputs():
    arc = np.tile(np.arange(101.), 2)
    ids = np.repeat([3, 7], 101)
    centers = np.c_[arc, ids*10, np.zeros(len(arc))]
    tangents = np.tile([1, 0, 0], (len(arc), 1))
    samples = [dict(point_m=c.tolist(), inside=True, clearance_m=2., floor_distance_m=.5,
                    roof_distance_m=1.5) for c in centers]
    return centers, ids, arc, tangents, samples


def test_views_cover_each_branch_floor_roof_and_ends_reproducibly():
    args = inputs()
    plan = plan_material_views(*args)
    assert plan == plan_material_views(*args)
    for sid in [3, 7]:
        views = [v for v in plan["views"] if v["segment_id"] == sid]
        stations = sorted(set(v["arc_length_m"] for v in views))
        assert stations[0] == 2 and stations[-1] == 98
        assert max(np.diff(stations)) <= 30
        assert {v["angle"] for v in views} == {"floor_forward", "roof_backward"}
        assert all(v["point_m"][2] == .5 for v in views)


def test_insufficient_view_budget_never_silently_truncates_branches():
    with pytest.raises(ValueError, match="Full branch coverage"):
        plan_material_views(*inputs(), max_views=2)


def test_view_outside_cavity_is_not_skipped():
    args = inputs()
    args[-1][2]["inside"] = False
    with pytest.raises(ValueError, match="outside"):
        plan_material_views(*args)


@pytest.mark.parametrize("spacing", [0, -1, float("inf"), float("nan")])
def test_invalid_spacing_rejected(spacing):
    with pytest.raises(ValueError):
        plan_material_views(*inputs(), spacing_m=spacing)
