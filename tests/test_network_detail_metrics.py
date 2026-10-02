"""Locality diagnostics must distinguish isolated features from background waves."""
import copy

import numpy as np
import pytest

from plume_advanced.evaluation.metrics.network_detail import locality_diagnostics, route_residuals
from plume_advanced.stages.network_detail_features import PassageFeature


def payload(x, y, width, *, sid=0, source=0, target=1, parent=None):
    return {'segments': [dict(segment_id=sid, source_node_id=source, target_node_id=target,
                              metadata={} if parent is None else {'detail_parent_segment_id': parent},
                              centerline=[dict(x=float(a), y=float(b), width=float(c)) for a, b, c in zip(x, y, width)])]}


@pytest.fixture
def straight():
    x = np.linspace(0, 400, 1601)
    return x, payload(x, np.zeros_like(x), np.full_like(x, 5.))


def test_identity_is_quiet_even_if_sampling_changes(straight):
    x, coarse = straight
    detailed = payload(x[::4], np.zeros_like(x[::4]), np.full_like(x[::4], 5.))
    diagnostics = locality_diagnostics(coarse, detailed)
    assert diagnostics['quiet_length_fraction'] == 1
    assert diagnostics['changed_reach_count'] == 0
    assert diagnostics['bend_width_profile_correlation_median'] is None
    assert diagnostics['width_peak_spacing_cv_median'] is None


def test_localization_and_coherence_are_measured_without_rewarding_total_noise(straight):
    x, coarse = straight
    bump = PassageFeature(100, 120, 170, 0., 0., 0., 0., 0., 'pocket').profile(x)
    local = payload(x, bump, 5+.5*bump)
    waves = payload(x, np.sin(x/7), 5+.5*np.cos(x/11))
    a, b = locality_diagnostics(coarse, local), locality_diagnostics(coarse, waves)
    assert a['quiet_length_fraction'] > .8
    assert b['quiet_length_fraction'] < .05
    assert a['changed_reach_count'] == 1
    assert a['bend_width_profile_correlation_median'] > .99
    assert b['width_peak_spacing_cv_median'] < .03
    assert b['routes_with_enough_peaks'] == 1


def test_junction_split_descendants_reconstruct_the_original_route(straight):
    x, coarse = straight
    split = payload(x[:801], x[:801]*0, 5+x[:801]*0, source=0, target=2, parent=0)
    split['segments'] += payload(x[800:], x[800:]*0, 5+x[800:]*0, sid=1, source=2, target=1, parent=0)['segments']
    assert locality_diagnostics(coarse, split)['quiet_length_fraction'] == 1
    broken = copy.deepcopy(split)
    broken['segments'].pop()
    with pytest.raises(ValueError, match='preserved original routes'):
        locality_diagnostics(coarse, broken)


def test_physical_projection_recovers_known_offsets(straight):
    x, coarse = straight
    detailed = payload(x, np.ones_like(x)*2., np.full_like(x, 5.3))
    _, lateral, width = route_residuals(coarse, detailed, coarse['segments'][0])
    np.testing.assert_allclose(lateral, 2.)
    np.testing.assert_allclose(width, .3)
