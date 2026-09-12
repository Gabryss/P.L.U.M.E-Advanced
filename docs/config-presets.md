# Maintained configuration presets

These are explicit, standalone TOML files. Relative texture paths resolve against the configuration location. The manifest records the fully resolved parameters after body, flow, development and resolution policies apply. Editing a derived field in a saved manifest does not change generation.

| Preset | Body | Topology / growth | Systems | Target route (m) | Voxel (m) | Events enabled | Material |
|---|---|---|---:|---:|---:|---|---|
| [earth_full_inspection_seed20260910](../config/earth_full_inspection_seed20260910.toml) | earth | general / layout | 1 | 5000 | 0.2 | no | neutral |
| [earth_independent_gallery](../config/earth_independent_gallery.toml) | earth | trunk_dominated / independent_growth | 3 | 380 | 0.2 | no | neutral |
| [earth_interacting_systems](../config/earth_interacting_systems.toml) | earth | general / layout | 3 | 5000 | 0.2 | no | neutral |
| [earth_long_interconnected](../config/earth_long_interconnected.toml) | earth | interconnected / independent_growth | 3 | 3000 | 0.2 | no | neutral |
| [earth_long_interconnected_full](../config/earth_long_interconnected_full.toml) | earth | interconnected / independent_growth | 3 | 3000 | 0.2 | no | PBR |
| [earth_long_multi](../config/earth_long_multi.toml) | earth | trunk_dominated / independent_growth | 3 | 3000 | 0.2 | no | neutral |
| [earth_long_single](../config/earth_long_single.toml) | earth | trunk_dominated / layout | 1 | 3000 | 0.2 | no | neutral |
| [earth_short_interconnected](../config/earth_short_interconnected.toml) | earth | interconnected / independent_growth | 3 | 400 | 0.2 | no | neutral |
| [earth_short_interconnected_full](../config/earth_short_interconnected_full.toml) | earth | interconnected / independent_growth | 3 | 400 | 0.08 | no | PBR |
| [earth_short_multi](../config/earth_short_multi.toml) | earth | trunk_dominated / independent_growth | 3 | 400 | 0.2 | no | neutral |
| [earth_short_single](../config/earth_short_single.toml) | earth | trunk_dominated / layout | 1 | 400 | 0.2 | no | neutral |
| [earth_tube_only](../config/earth_tube_only.toml) | earth | general / layout | 1 | 1500 | 0.2 | no | PBR |
| [earth_valentine_multi_inspection_seed20260910](../config/earth_valentine_multi_inspection_seed20260910.toml) | earth | trunk_dominated / layout | 3 | 300 | 0.2 | no | neutral |
| [earth_valentine_topology](../config/earth_valentine_topology.toml) | earth | trunk_dominated / layout | 1 | 300 | 0.2 | no | neutral |
| [project](../config/project.toml) | earth | general / layout | 1 | 1500 | 0.6 | yes | PBR |

A route target describes longitudinal extent; combined length across parallel passages can be larger. Event enablement does not imply that rock meshes were requested or that every event kind appears for every seed. Single-network presets use one system, even if the separate candidate/source-point control has a larger value. Use [the reliability guide](reliability.md) to evaluate changed presets across seeds.

The packaged `src/plume_advanced/default_project.toml` is the minimal installation default and does not require Rocky. The project's default `config/project.toml` and the named inspection scenarios have different purposes; inspect the selected resolved configuration before comparing their sizes.

Computed metadata includes host body scales and geometry resolution-policy fields (`resolution_mode`, target sampling and resolved characteristic sizes). Their resolved values describe the chosen body and policy, not independent scientific controls. Explicit metric inspection presets disable host scaling; an Earth-sized fixed domain is not automatically widened by a body override.
