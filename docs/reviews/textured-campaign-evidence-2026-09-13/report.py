"""Build the dated report from audited campaign outcomes, without changing evidence."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = json.loads((root / "audit.json").read_text())
    native = json.loads((root / "native_campaign.json").read_text())
    assert native["complete"], "Wait for native evaluation to finish"
    assert all(
        json.loads((root / g / "summary.json").read_text())["complete"]
        for g in ["single_250", "multi"]
    )
    cases = sorted(audit["cases"], key=lambda c: (c["group"] == "multi", c["case"]["seed"]))
    passed = [c for c in cases if c["runs"] and "metrics" in c["runs"][0]]
    replays = sum(
        len(c["runs"]) == 2
        and all("identity" in r for r in c["runs"])
        and c["runs"][0]["identity"] == c["runs"][1]["identity"]
        for c in cases
    )
    originals = [c["runs"][0] for c in passed]
    checked = sum(x["passed"] for x in audit["checks"])
    native_passed = sum(c["passed"] for c in native["cases"])
    table = []
    resources = []
    links = []
    native_table = []
    for c in cases:
        name = "Multi" if c["group"] == "multi" else "Single"
        seed = c["case"]["seed"]
        r = c["runs"][0]
        if "metrics" not in r:
            table.append(f"| {name} / {seed} | Rejected | — | — | — | — |")
            continue
        m = r["metrics"]
        replay = c["runs"][1] if len(c["runs"]) > 1 else {}
        table.append(
            f"| {name} / {seed} | {m['combined_length_m']:.1f} | {m['recovery_outcome']} | {m['surface_attempts']} / {r['relief']['relief_scale']:g} | {m['triangles']:,} | {m['asset_bytes'] / 1e6:.1f} |"
        )
        resources.append(
            f"| {name} / {seed} | {r['elapsed_s'] / 60:.1f} / {replay.get('elapsed_s', 0) / 60:.1f} | {max(r.get('peak_rss_mib') or 0, replay.get('peak_rss_mib') or 0):.0f} | {r['under_resolved']}/{r['section_count']} | {r['minimum_sampled_clearance_m']:.3f} | {r['package_bytes'] / 1e6:.1f} |"
        )
        run_rel = Path(r["directory"])
        input_base = f"../../outputs/textured_campaign_20260913/{run_rel}"
        n = next(
            x for x in native["cases"] if x["group"] == c["group"] and x["index"] == c["index"]
        )
        if n["passed"]:
            p = f"../../outputs/textured_campaign_20260913/{Path(n['output']).relative_to(root)}"
            links.append(
                f"| {name} / {seed} | [GLB]({input_base}/export/plume_cave.glb) | [Unity scene]({p}/unity_project/Assets/PLUME_Inspection.unity) | [Unreal project]({p}/unreal_project/PLUMENative.uproject) | [Quality report]({input_base}/pipeline_quality_report.json) |"
            )
            u = n["summary"]["checks"]["unity"]["native"]
            e = n["summary"]["checks"]["unreal"]["native"]
            native_table.append(
                f"| {name} / {seed} | {u['passagePassed']}/{u['passageSamples']} | {u['maximumClearanceErrorM'] * 1000:.4f} | {e['passage_passed']}/{e['passage_samples']} | {e['maximum_clearance_error_cm'] * 10:.4f} |"
            )
        else:
            native_table.append(f"| {name} / {seed} | Native failure | — | Native failure | — |")
    routes_repaired = sum(r["metrics"]["repair_pass"] > 0 for r in originals)
    reduced = sum(r["relief"]["relief_scale"] < 1 for r in originals)
    zero = sum(r["relief"]["relief_scale"] == 0 for r in originals)
    regenerated = sum(r["metrics"]["recovery_outcome"] == "regenerated" for r in originals)
    local = sum(r["metrics"]["recovery_outcome"] == "locally_repaired" for r in originals)
    texture = sum(r["metrics"]["texture_map_repairs"] > 0 for r in originals)
    package = sum(r["metrics"]["texture_package_attempts"] > 1 for r in originals)
    uv = sum(r["uv_repair_observed"] for r in originals)
    visual = sum(r["metrics"]["visual_attempts"] > 1 for r in originals)
    thin = sum(r["under_resolved"] > 0 for r in originals)
    collider = sum(r["metrics"]["collision_raw_fallback"] for r in originals)
    multi_loops = [c["runs"][0]["topology"]["genus"] for c in passed if c["group"] == "multi"]
    loop_limit = (
        "None of the full multi-source cases contains a closed split-and-rejoin loop; the single-source cases and regression fixtures provide loop coverage."
        if multi_loops and not any(multi_loops)
        else "Closed split-and-rejoin topology coverage is limited to the accepted graphs shown above."
    )
    evidence = "textured-campaign-evidence-2026-09-13"
    text = f"""# Textured evaluation campaign — 13–14 September 2026

**{len(passed)}/{len(cases)} originals passed, {replays}/{len(cases)} cold replays matched exactly, and {native_passed}/{len(cases)} originals passed both Unity and Unreal inspection.** The final independent audit passed **{checked}/{len(audit["checks"])} checks**, verifying **{audit["verified_artifacts"]} generated artifact receipts**. These are six designs, each built twice; the native engine checks use the six originals.

The campaign started on 13 September and completed on 14 September; paths retain the start date.

This campaign found an **inspection-lighting acceptance gap**: a high-contrast image could pass while its floor was badly overexposed. The native inspection now keeps its light at a verified air sample, uses bounded intensity adjustments, retains exposure trials, and rejects excessive clipping. No generation algorithm or source material was changed during this campaign.

## Protocol and retained failures

Both groups use Earth, a 250 m dominant-route target, fixed 12 cm voxels, no rocks/events, and three reusable **4096×4096** PBR maps at a **4 m tile size** with normal strength **1**. Baked height-map displacement is disabled; surface relief comes from the geometry stages and normal mapping. Each group uses seeds **1, 42 and 4294967295**. Multi-source cases grow three systems in the same host field. Seed 1 deliberately revisits the difficult multi-source regression; the sample is not six previously unseen seeds.

Successful originals are regenerated from scratch under Python hash seeds 11 and 37. Host, network, section, raw-mesh, GLB, upstream-recovery and texture-recovery identities must all match. Original/replay pairs use no shared checkpoints. Generation source, recipe content and dependency fingerprints remain fixed. Two full workers run concurrently with per-worker limits of 3600 seconds and 8192 MiB address space. Native editor jobs run sequentially on an RTX 3070.

The initial single-source trial shortened the existing recipe to 150 m without reducing its required loops and side branches. All three seeds rejected that incompatible configuration before meshing. The corrected protocol restores 250 m and keeps all seeds. These three setup rejections are retained separately in `single/`; they are not counted as repaired 250 m cases or silently removed. See the [initial protocol]({evidence}/protocol.json), [final protocol]({evidence}/protocol_v2.json), and [execution record]({evidence}/execution.md).

## Generation and repair results

| Case / seed | Summed length (m) | Upstream recovery | Surface attempts / accepted relief scale | Triangles | GLB MB |
| --- | ---: | --- | ---: | ---: | ---: |
{chr(10).join(table)}

The summed length includes all branches; it can exceed the 250 m route target. The surface-attempt count describes the accepted upstream realization; the complete recovery journal also preserves earlier rejected realizations.

Natural runs accepted a refined network candidate in **{routes_repaired}/{len(cases)} cases**, normal-vector repair in **{texture}/{len(cases)}**, surface-detail reduction in **{reduced}/{len(cases)}**, and upstream network replacement in **{regenerated}/{len(cases)}**. Accepted local upstream repair occurred in **{local}/{len(cases)}**, UV repair in **{uv}/{len(cases)}**, visual export retry in **{visual}/{len(cases)}**, and texture-package rebuilding in **{package}/{len(cases)}**. A zero count means that path was not naturally needed in this sample; the fault-injection tests exercise those paths separately.

For the shared normal map, preparation repaired **472,092 nonunit vectors** and reported zero remaining vectors outside its tolerance after quantization. Color and roughness were retained without a repair. Both the prepared-map and repair-journal identities participate in cold replay. Missing or corrupt source images remain errors; texture repair does not invent a replacement tile or change the network seed.

![Accepted network centrelines]({evidence}/networks.png)

These panels show accepted centrelines, not mesh silhouettes. Y is horizontal, X vertical, and each panel uses equal metre scales. Orange dots mark sources. Distinct colours indicate source contributions and dark lines indicate shared passages. The audit checks source counts, parallel coverage, graph/mesh topology and preserved host identity.

## Native texture inspection

Every accepted original was imported into **Unity 6000.6.0f1 / URP 17.6 / glTFast 6.20** and **Unreal 5.8.2**, using Vulkan and the shipped continuous projection material. Each editor produced two interior captures plus a normal-disabled control. Both retained three 4K maps with the expected color spaces and passed the normal-map influence check. Unity additionally compared all imported vertices and rendered again after deliberately corrupting the UVs. Unreal checked mesh bounds and compiled shaders. Floor/roof measurements were compared with the final exported visual surface.

| Case / seed | Unity samples passed | Max ray-distance error (mm) | Unreal samples passed | Max ray-distance error (mm) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(native_table)}

The initial seed-1 single-source view clipped **28.81%** of pixels in Unity and **16.92%** in Unreal. The old blank/contrast check accepted them. The new fixture halves light intensity for at most eight trials, targets no more than 0.5% white-clipped pixels, and rejects final captures above 1%. It preserves the rejected trial images and records the chosen intensity. The first view now clips approximately **0.00016%** in Unity and **0.0249%** in Unreal. A regression test verifies that a half-white, half-black image fails despite high contrast. This is inspection exposure control, not a change to the rock material or generated mesh.

![Same cave before and after inspection-light adjustment]({evidence}/exposure_comparison.png)

The original native receipt is retained as [version 1 evidence]({evidence}/native_campaign_v1.json). All final native checks were repeated with the [version 2 inspection protocol]({evidence}/native_protocol_v2.json); the final audit rejects changed fixture hashes.

![Unity interior contact sheet]({evidence}/unity_interiors.png)

![Unreal interior contact sheet]({evidence}/unreal_interiors.png)

Manual review found the rock tile present on floor and roof without obvious triangular chart boundaries in the captured views. Several views show very low, broad passages, and Unreal is noticeably brighter and paler than Unity. [Image fingerprints and observations]({evidence}/visual_review.json) identify the reviewed captures.

The two engines use separate light and tone-mapping implementations; these images are not a calibrated photometric comparison. The ordinary GLB retains its UV material. The continuous native material must be applied explicitly; a passing continuous-material test does not prove the portable UV material is seam-free.

## Tests and resource limits

The full suite passed **787 tests and 59 subtests**, with 20 optional skips and three performance/paper tests deselected. Seventeen of those optional tests were subsequently executed successfully: two native Blender material/import tests and 15 Unity shader-compiler variants. After adding the clipping guard, all **13 native-runner regression tests** passed. The remaining unavailable checks are two glslang compiler tests and the optional Manim module. Lint passed and mypy checked 104 source files.

[Material-test inventory and results]({evidence}/test_coverage.json) include 40 texture-recovery cases, four UV-repair cases, projection tests and native-runner failure handling. They cover missing/stale maps, damaged bindings/settings/shaders, normal conventions, bounded retries, preservation of previous exports, and exact repair replay. This inventory is not a code-coverage percentage.

| Case / seed | Original / replay min | Larger peak RSS MiB | Under-resolved profiles | Smallest sampled clearance (m) | Whole export package MB |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(resources)}

All primary assets remain subject to the unchanged five-million-triangle and 350 MB limits. Package sizes include alternative formats, collision and material support files; they exclude checkpoints and native editor project caches. Timings include concurrent evaluation load and are not isolated performance benchmarks.

**Remaining limitations:** {thin}/{len(cases)} cases retain profiles below the eight-voxel sampling recommendation; {collider}/{len(cases)} retain the full collider because simplified collision failed inspection. In {zero}/{len(cases)} cases the added accretion layer was omitted to preserve topology. Very low passages can pass numerical inspection: acceptance does not guarantee clearance for a person or a particular robot. The engine checks verify import and sampled collision, not simulator frame rate, complete navigation, every triangle intersection, every camera location, or geological realism. {loop_limit} The finite sample covers one body, one resolution and one shared texture tile. Unreal validation disables Nanite and uses full visual-mesh collision; production Nanite, streaming and simplified-collider performance require separate evaluation.

## Inspect and reproduce

| Case / seed | Portable asset | Unity | Unreal | Pipeline evidence |
| --- | --- | --- | --- | --- |
{chr(10).join(links)}

Open each Unity project through the editor/Hub and load its saved scene. Each Unreal project starts at `/Game/PLUME_Run01/PLUME_Inspection`. These isolated projects contain the continuous material. The [execution record]({evidence}/execution.md) contains the generation and native-inspection commands. The [independent audit]({evidence}/audit.json), [native summary]({evidence}/native_campaign.json), [test logs]({evidence}/tests.log), and [archive manifest]({evidence}/archive_manifest.json) preserve compact evidence if bulky outputs are later removed.

Frozen production source: `{audit["source_sha256"]}`. The inspection-runner and fixture revisions are tracked separately in the native protocol.
"""
    Path(__file__).resolve().parent.parent.joinpath("textured-campaign-2026-09-13.md").write_text(
        text
    )


if __name__ == "__main__":
    main()
