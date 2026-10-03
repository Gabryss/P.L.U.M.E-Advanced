"""Impossible layer/section cover must fail before expensive candidate search."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from plume_advanced.config import load_project_config
from plume_advanced.pipeline.seed_search import retryable_rejection
from plume_advanced.stages.network import CaveNetworkGenerator


def test_impossible_roof_cover_is_fatal_before_host_or_candidates_are_used():
    config = load_project_config('config/branching-layers.toml')
    generator = CaveNetworkGenerator(config.network)
    with patch('plume_advanced.stages.network_acceptance.generate_accepted_network') as search:
        with pytest.raises(ValueError, match='changing seeds cannot fix') as caught:
            generator.generate(None, section_config=config.section_field)
        search.assert_not_called()
    assert not retryable_rejection(caught.value)


@pytest.mark.parametrize('mode', ['network_only', 'compatible_cover', 'single_layer'])
def test_preflight_does_not_reject_network_only_or_compatible_inputs(mode):
    config = load_project_config('config/branching-layers.toml')
    network, sections = config.network, config.section_field
    if mode == 'network_only':
        sections = None
    elif mode == 'compatible_cover':
        network = replace(network, layers=replace(network.layers, minimum_rock_m=sections.minimum_roof_thickness))
    else:
        network = replace(network, layers=replace(network.layers, enabled=False))
    with patch('plume_advanced.stages.network_acceptance.generate_accepted_network', return_value='accepted') as search:
        assert CaveNetworkGenerator(network).generate(None, section_config=sections) == 'accepted'
        search.assert_called_once()
