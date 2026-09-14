"""Reproduce a real bounded convergence/clearance check without external editors."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from plume_advanced.identity import package_source_hash
from plume_advanced.pipeline.resolution import ResolutionBudgetError, build_with_resolution_checks
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNode,
    CavePoint,
    CaveSegment,
)
from plume_advanced.stages.route_clearance import fit_required_sections
from plume_advanced.stages.section_field import SectionFieldGenerator


def evaluate():
    coordinates = [(float(x), 0.) for x in range(0, 21, 5)]
    nodes = (CaveNode(0, 0, 0, 0, 0, 'entry'), CaveNode(1, 20, 0, 20, 0, 'exit'))
    points = tuple(CavePoint(i, x, y, 100-.01*x, 1, 30, .8, 1, x, 2, 1,
                            1450-.025*x, x/.35) for i, (x, y) in enumerate(coordinates))
    network = CaveNetwork(CaveNetworkConfig(target_route_length_m=20, source_count=1), nodes,
                          (CaveSegment(0, 0, 1, 'backbone', 0, points, {}),), (),
                          np.ones((2, 2), bool), np.ones((2, 2)), (0, 1), (), (), ())
    sections = SectionFieldGenerator().generate(network)
    config = GeometryConfig(voxel_size=.2, density_margin=1, minimum_radius=.2,
        wall_roughness_amplitude=0, cave_diffuse_texture='', cave_normal_texture='',
        cave_roughness_texture='', cave_displacement_texture='', required_route_height_m=.5,
        required_route_width_m=.5, resolution_refinement_attempts=2,
        resolution_min_voxel_size_m=.05, resolution_max_allocated_voxels=8000000)
    sections, envelopes = fit_required_sections(network, sections, config)
    geometry = build_with_resolution_checks(network, sections, config)
    result = dict(source_sha256=package_source_hash(), envelopes=envelopes,
                  positive=dict(geometry.resolution_repair),
                  clearance=dict(geometry.mesh_inspection)['traversal'])
    try:
        build_with_resolution_checks(network, sections, replace(config,
            resolution_refinement_attempts=1, resolution_convergence_m=1e-12))
    except ResolutionBudgetError as error:
        result['unconverged_rejected'] = error.report
    else:
        raise AssertionError('Unconverged candidate incorrectly accepted')
    try:
        build_with_resolution_checks(network, sections, replace(config,
            resolution_max_allocated_voxels=8))
    except ResolutionBudgetError as error:
        result['allocation_rejected'] = error.report
    else:
        raise AssertionError('Allocation budget ignored')
    return result


if __name__ == '__main__':
    import sys
    Path(sys.argv[1]).write_text(json.dumps(evaluate(), indent=2, sort_keys=True)+'\n')
