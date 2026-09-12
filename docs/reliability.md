# Reliability, testing and simulation budgets

The [preset catalog](config-presets.md) lists the maintained scenarios and their effective sizes.

The [12 September implementation and evaluation report](reviews/reliability-cleanup-2026-09-12.md) records the tested cases, actual failures and fixes, exported sizes, and verification limits.

PLUME screens a bounded sequence of network candidates before expensive meshing. A seed identifies a repeatable candidate sequence, not a promise that any physically impossible configuration can succeed. Acceptance uses the shared host field and the generated cross sections. It retains the first passing candidate, records rejected candidates and repairs, and never silently disables quality checks.

Repairs enforce the configured width-gradient bound using the repaired physical distances, so smoothing cannot create arbitrarily abrupt changes in tube width. Terminal tapering uses an absolute envelope rather than compounding at every repair.

Gallery routing bounds correlated lateral variation smoothly, so increasing route length does not introduce arbitrarily large random excursions. Blind breakouts screen other passages and host bounds before being added. The trunk-dominated style forms local split/rejoin islands after source confluence; the interconnected style retains compound parallel interactions. Empty island labels do not count as islands. Local gallery junction transitions span at most four base widths, avoiding long premature fusions between arms.

When a route already passes network screening but its sections overlap, repair preserves that route and reduces conflicting width envelopes within the configured minimum passage radius. It then repeats the section, topology and flow checks. This avoids smoothing a valid island closed while trying to repair its clearance. Rejected candidates and earlier failed campaigns remain evidence; thresholds are not relaxed to make a seed pass.

A bad candidate caused by numerical or geometric degeneracy can advance to the next deterministically derived seed. A host that cannot fit the requested tube scale fails immediately with a domain-size explanation. Programming errors, missing inputs and resource exhaustion remain errors; retrying them as random morphology would conceal defects.

## Routine verification

Install the test dependencies, including those used by the paper evaluation tests:

```bash
uv sync --locked --group dev --extra rocks --extra paper
uv run --no-sync ruff check src scripts tests
uv run --no-sync mypy src/plume_advanced
uv run --no-sync pytest -q --cov=plume_advanced --cov-report=term-missing
```

The suite includes corrupt/stale checkpoint rejection, invalid numeric input, bounded retries and exhaustion, deterministic repairs, conserved flow and graph topology, dense/sparse mesh agreement, stability and floor checks, export round trips, texture bindings, shading-frame reference comparisons, provenance, and file/triangle budgets. Performance tests use broad limits to catch large regressions rather than rank machines.

## Seed campaigns

The campaign runs each case in a fresh process. A timeout or native crash is retained as a failed case, and later cases still run. Successful cases are replayed with a different `PYTHONHASHSEED`. Stages A–C compare semantic identities; full cases additionally compare the GLB bytes.

A network and section sweep across short and long Earth configurations:

```bash
uv run --no-sync python -m plume_advanced.evaluation.reliability \
  --configs config/earth_short_single.toml config/earth_short_multi.toml \
    config/earth_short_interconnected.toml config/earth_long_single.toml \
    config/earth_long_multi.toml config/earth_long_interconnected.toml \
  --output outputs/seed_campaign --timeout 600
```

The default seed list includes zero, small integers, a dated inspection seed, and the maximum unsigned 32-bit seed. Each root seed derives all stage seeds, so the host is varied as well. Use `--seeds` for a different list. `--no-replay` reduces runtime but explicitly omits the cross-process repeatability check.

A compact full pipeline campaign:

```bash
uv run --no-sync python -m plume_advanced.evaluation.reliability \
  --configs config/earth_short_single.toml config/earth_short_interconnected.toml \
  --scope full --seeds 0 17 --voxel-size 0.22 --timeout 600 \
  --output outputs/full_seed_campaign
```

`--voxel-size` is an explicit study override, recorded in each resolved configuration. Coarser meshes cost less, but their measurements need separate resolution/convergence checks. Full cases use each preset's event and material settings, perform both floor-map passes, generate the mesh, export it, and validate the portable asset. They do not imply that rocks were enabled or that every application imported the asset.

For other bodies, use a host configuration that scales with the body:

```bash
uv run --no-sync python -m plume_advanced.evaluation.reliability \
  --configs config/project.toml --bodies mars moon --seeds 0 1 17 4294967295 \
  --output outputs/body_campaign --timeout 600
```

The explicitly dimensioned `earth_short_*` presets retain their Earth-sized host when the body is overridden. Larger planetary tube scales may not fit that field. Widen the host or choose a body-scaled preset; do not treat a failed domain constraint as evidence that more seeds will fix it.

Every campaign creates a new directory and records `summary.json`, per-case configuration, quality report, progress trace, timings, and worker log. Full cases retain their exports. The process address-space ceiling defaults to 8192 MiB on Linux (`--memory-limit-mib`); other platforms record that this limit is unavailable. The wall-time limit applies on all platforms. A finite passing sample is evidence about the tested cases, not proof for every integer seed or geological validation.

Keep source and dependencies unchanged during a campaign. The summary includes their identities and fails its overall pass flag if production source changes during the run. Timings from concurrent campaigns are not controlled performance benchmarks.

## Detailed progress

Normal generation shows eleven overall stages, plus the current work unit. Long steps report host layers, candidate/repair decisions, section segments, voxel tiles, relief, mesh chunks, floor-raycast/revalidation work, surface orientation, UV-chart batches, shading-frame triangle batches, collision preparation, serialization and checkpoint work. Operations with unknown work counts show elapsed time without a fabricated percentage or ETA.

`progress.jsonl` beside the run outputs preserves the events for diagnosing a slow or failed run. It is an operational trace, not part of the geometry's deterministic identity. Checkpoints use schema v2 and verify configuration, inputs, executing source, lockfile where available, dependency versions, Python version and payload integrity. Historical pickles must be reopened with their historical code.

## Simulation exports

Exports include `export_size_report.json`, covering visual triangle count, vertices after UV seams, mesh-buffer bytes, collision triangles, individual file sizes and the package total. Engine allocations, decoded textures and mipmaps add runtime memory beyond those figures.

Optional limits fail before publishing an oversized package:

```toml
[export]
target = "unity"
format = "glb"
generate_collision = false
max_visual_triangles = 5000000
max_asset_bytes = 350000000
```

Zero disables a limit. Triangle limits are checked before UV and material preparation; file-size limits are checked inside the atomic staging directory. Disabling collision also skips the collision-generation work. PLUME does not automatically decimate a narrow cave merely to meet a file-size budget: that could change traversability. Choose a suitable extent, texture size and verified mesh resolution.

Shading calculations run in bounded triangle batches. Material revisions reuse the existing UV buffer rather than retaining a second unused copy. Neutral tube-only exports use the same surface preparation as normal generation. Diagnostic mesh figures cap their displayed triangle sample without modifying the exported mesh.

## Native Blender regression and inspection

```bash
PLUME_BLENDER_BINARY=/path/to/blender uv run --no-sync pytest -q tests/test_blender_material.py
blender --background --python scripts/create_blender_inspection.py -- outputs/my_run --quality standard
```

The optional native test imports embedded maps, checks shader connections and linear data-map color spaces, packs the images, and renders a controlled two-color fixture with Cycles. It catches grey/missing-map regressions but does not prove that every cave is free of UV seams or shading defects. It is skipped when Blender is not configured. Unity and Unreal native imports still require their respective application checks.

Inspection presets offer `preview` (32 samples), `standard` (256) and `high` (1024), with denoising enabled. Material Preview remains the saved textured startup view. Existing user-edited Blender scenes are not rewritten by the test suite.

To validate an intentionally untextured file, use `plume-validate ASSET --material-profile neutral`. The default `textured` profile requires the PBR maps. Both profiles check geometry, normals, UVs and provenance. The selected asset must be an exact recorded output or a verified material revision linked to that output; a nearby successful run is insufficient.
