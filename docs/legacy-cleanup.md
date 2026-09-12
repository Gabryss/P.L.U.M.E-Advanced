# Legacy cleanup and review — 11 September 2026

The pre-cleanup version is preserved in commit **`32cbcee`** on
`origin/lava-emplacement-history`. It was pushed before any implementation was
removed. This review covers the maintained Python package, repository utilities,
tests, presets, dependencies, entry points, CI checks and documentation.

## Removed implementation

| Finding | Change | Reason |
|---|---|---|
| `legacy_braid`, `BraidGrammarConfig`, braid zones, ladders and the old spur tracer | Removed the generator branch, its private helpers and public re-export | Every maintained preset uses the current lobe or gallery generators. These recipes were an explicit historical alternative. |
| Braid-only configuration and validation | Removed `network.braid_grammar`, `network.growth_model` and `network.spur_*` settings | They either selected the removed implementation or had no effect on current generation. Current lobe-generated blind branches remain supported. |
| Development limits expressed as braid zones | Replaced `run.dev_max_braid_zones` with `run.dev_max_lobe_paths` | The active development limit now describes lobe paths directly. |
| Length-based loose-debris population | Removed `events.debris_density_basis`, `rock_density_per_100m` and `boulder_density_per_100m` | Maintained presets use floor-area density. Geological modifier density per 100 m remains active and distinct. |
| Configuration migrations for schemas 1 and 2 | Removed implicit migration; require project schema 4 | Unsupported settings now fail explicitly instead of being silently stripped. |
| Export of unwelded chunk concatenations | Removed the fallback; require the assembled Stage-E mesh | Export must consume the actual assembled surface. Missing geometry produces an actionable error. |
| Unequal-profile contour union during lofting | Removed the union fallback; reject mismatched vertex counts | The fallback could widen a passage instead of interpolating its cross-sections. Stage C already supplies matching profiles. |
| Unused GLB primitive writer and evaluation point helper | Removed `_StrictGlbBuilder._primitive` and `_common.point_sequence` | Call-site and syntax-tree inspection found no consumers. The current exporter has its own live primitive construction. |
| Duplicate API names | Removed `CaveGeometry.meshes`, host `routing_cost` properties, `evaluation.metrics.continuity`, and unused continuity/serialization aliases | Use `chunk_meshes`, `growth_cost`, `evaluation.continuity.longitudinal_continuity` and `diagnostics_to_json`. Saved artifact keys such as `routing_cost` remain unchanged. |
| Duplicate command wrappers | Removed `scripts/generate_cave.py` and `scripts/validate_asset.py` | Installed commands and module entry points provide the same functionality. The body-figure caller now invokes the module directly. |
| Duplicate source-order projection | Consolidated it into `network_systems.enforce_source_order` | Both multi-system routing paths now use one implementation without changing the source identities or numeric results. |
| Unused private arguments | Removed unused inputs in routing, debris counts, lineage and plotting helpers | Signatures now reflect the data the functions actually consume. |
| Unused spreadsheet dependency | Removed `openpyxl` from the paper extra and its orphan `et-xmlfile` lock entry | No maintained source, utility or media scene imports it or reads Excel workbooks. The remaining dependencies have live consumers. |

## Current interface

Start with one of the maintained files in `config/`, or the installed packaged
default. All now declare:

```toml
schema_version = 4
```

The **experiment** schema in `paper/experiments.toml` is a separate interface and
remains version **1**. Do not change it to match the project schema.

For an older custom project configuration:

1. Remove the retired keys and the entire `[network.braid_grammar]` table listed
   above. Simply changing the schema number is insufficient.
2. Set `run.dev_max_lobe_paths` if development mode needs a custom lobe budget.
   The default is 6 at the 5 km reference extent. The effective cap is
   `max(1, round(dev_max_lobe_paths * sqrt(target_route_length_m / 5000)))`.
   This replaces the old rounding in groups of three, so custom development
   budgets may need adjustment. Full runs do not use this cap. The packaged
   preview uses a budget of 5 to preserve its previous effective cap of 3.
3. Express loose-rock and boulder density with `rock_density_per_100m2` and
   `boulder_density_per_100m2`. A length-based density cannot be converted to an
   equivalent area density without choosing a representative floor width.
4. For Python callers, use `host.growth_cost`, `sample.growth_cost`,
   `geometry.chunk_meshes`, and imports from `plume_advanced.evaluation.continuity`.
   Use `longitudinal_continuity` instead of `evaluate_continuity` or
   `continuity_diagnostics`, and `diagnostics_to_json` instead of
   `serialize_diagnostics`.
5. Use `plume-generate` / `plume-validate`, or
   `python -m plume_advanced.cli` / `python -m plume_advanced.validation`.

Unknown settings and unsupported project schemas produce an error that points
to this guide. The loader does not infer which removed settings the caller meant.

## Retained deliberately

- **Supported generation modes:** general lobe growth, dominant-gallery layouts,
  independent systems, and persistent interconnected galleries. A control that
  belongs to a different supported mode is not dead code. The mode guides
  describe which controls each mode consumes.
- **Optional stages:** geological modifiers, Rocky props, dense and tiled
  geometry, adaptive and fixed sampling, textures, collision and all export
  targets. A no-rocks inspection preset does not make the rock implementation
  obsolete.
- **Comparison backends:** `downflow_reference` and the external `flowy` adapter
  are reachable, documented and tested. The external executable is still a
  separate optional dependency.
- **Convenience APIs and framework hooks:** `GeometryGenerator.generate()`
  delegates to the current two-pass workflow. Pytest plugin hooks and voxel-query
  protocol methods are dynamically dispatched, even though a simple name scan
  sees no direct calls.
- **Evaluation data adapters:** missing-history handling and input metric-name
  mappings support comparison datasets and modes without phase records. They
  do not invoke an obsolete generator or invent unavailable measurements.
- **Research utilities:** the 28 remaining scripts have distinct generation,
  inspection, validation, plotting or media purposes. Dated comparison scripts
  remain referenced by the corresponding scientific reports.
- **Frozen scientific evidence:** source snapshots, campaign inputs, raw tables,
  measurements and manuscript bundles under `paper/` were not rewritten or
  deleted. They are not imported by the installed package. Their historical code
  and hashes are part of the experiment record.

## Existing caves and reproduction

Existing meshes, Blender files, figures and output directories were preserved.
They can still be inspected in Blender and the target engines.

Python checkpoints are tied to source and configuration fingerprints. The
cleanup changes those fingerprints and removes historical class definitions;
old pickles are not a cross-version interchange format. Use commit `32cbcee` or
the source bundle recorded with a run to resume or re-render that historical
run. Regenerate with a current preset to create a schema-4 run. Do not relabel an
old result as generated by the cleaned source.

## Verification and findings

The original checkpoint was tested from an isolated copy of its source, configs
and tests. It had **nine project test failures and 47 type errors**. An additional
texture-dependent failure in that isolated test environment passed after its
missing texture directory link was restored.

The review corrected stale test fixtures and type errors without disabling the
production morphology gate. Synthetic roof and local-mesh unit fixtures now
explicitly bypass network validation because they have no network graph. The
connectivity smoke test uses a current inspection preset at an appropriate
resolution instead of a coarse general preview. The resume test accepts the
generator's current quality arguments. New regressions cover retired settings,
all maintained presets and rejection of incompatible loft profiles.

Type fixes use explicit list/array variables, typed accumulators, explicit
coordinates and validation of phase metadata. They do not suppress the type
checker or relax the production morphology thresholds.

Four deterministic before/after cases cover the packaged general preview,
short single gallery, independent gallery and short interconnected preset.
All four match in host, network and section semantic hashes. These comparisons
disable candidate acceptance to isolate the generator; they are an output
preservation check, not an additional claim of geological validity. Saved
comparison evidence is in [legacy-cleanup-2026-09-11.json](reviews/legacy-cleanup-2026-09-11.json).

Final validation on Python 3.13.5:

| Check | Result |
|---|---|
| Full test suite | 410 passed; 59 subtests passed; 1 optional Manim module skipped |
| Line coverage | 84.05%, exceeding the CI requirement of 70% |
| Ruff and mypy | Passed; all 84 package source files type-check |
| Compilation and internal import audit | Passed for maintained source, scripts, tests and the media scene |
| Utility entry points | All 27 non-Blender commands passed `--help`; the Blender utility compiled |
| Dependency checks | Installed dependencies compatible; updated lock passes the offline consistency check |
| Distribution build | Wheel and source package built; installed-wheel defaults load successfully |
| Documentation links and whitespace checks | Passed |
| Generation preservation | All four host/network/section comparisons match the checkpoint |

The optional `video` dependency is absent in this environment, so the Manim
typography module was not executed. Native Blender execution and the external
Flowy program were not rerun; their maintained interfaces were reviewed, and
export contracts and the Flowy adapter are covered by automated tests. The CI
Python 3.12 job was not run locally.

The cleanup remains as local changes for review; the pushed checkpoint precedes
the cleanup. Detailed counts, hashes and environment versions are recorded in
the evidence file linked above.
