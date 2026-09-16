import numpy as np

from plume_advanced.evaluation.view_selection import select_passage_target


def samples():
    arc = np.arange(21.)
    return np.c_[arc, np.zeros((21, 2))], arc


def test_view_cone_prefers_open_passage_over_a_narrow_central_slit():
    centers, arc = samples()
    def ray(direction, reach):
        return reach if direction[0] < 0 or np.linalg.norm(direction[1:]) < .01 else .3
    first = select_passage_target(centers, arc, 10, centers[10], ray)
    assert first["target_m"][0] < 10
    assert first == select_passage_target(centers, arc, 10, centers[10], ray)


def test_camera_near_a_cap_looks_back_into_the_passage():
    centers, arc = samples()
    result = select_passage_target(centers, arc, 19, centers[19],
        lambda direction, reach: 1 if direction[0] > 0 else reach)
    assert result["target_m"][0] < 19


def test_no_useful_view_is_reported_as_unavailable():
    centers, arc = samples()
    assert select_passage_target(centers, arc, 10, centers[10], lambda d, r: .5) is None
