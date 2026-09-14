"""Read-only evidence audit for this campaign; write only a separate audit.json."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.reliability import ReliabilityCase
from plume_advanced.evaluation.reliability_reports import verify_result
from plume_advanced.evaluation.reliability_state import plan_identity
from plume_advanced.identity import package_source_hash, sha256_file
from plume_advanced.pipeline.recovery import SEED_DOMAIN
from plume_advanced.procedural import derive_subseed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol_v2.json")
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    protocol = json.loads(args.protocol.read_text())
    checks, rows = [], []

    def check(label, passed):
        checks.append({"name": label, "passed": bool(passed)})

    source = package_source_hash()
    for filename, digest in protocol["configs_sha256"].items():
        check(f"Frozen recipe: {Path(filename).name}", sha256_file(filename) == digest)
    check("Frozen production source", source == protocol["source_sha256"])
    for group, cases in protocol["groups"].items():
        base = root / group
        summary = json.loads((base / "summary.json").read_text())
        check(f"{group}: complete and accepted", summary["complete"] and summary["passed"])
        check(f"{group}: every planned case retained", len(summary["cases"]) == len(cases))
        plan = json.loads((base / "plan.json").read_text())
        check(
            f"{group}: source, inputs, runtime and recipe unchanged",
            plan
            == plan_identity([ReliabilityCase(**c) for c in cases], source=source, replay=True),
        )
        for index, expected in enumerate(cases):
            label = f"{group}/{index}: seed {expected['seed']}"
            original, replay = None, None
            row = dict(group=group, index=index, case=expected, runs=[])
            for prefix, hash_seed in (("case", 11), ("replay", 37)):
                # No automatic replacement of missing/failed evidence by this auditor.
                path = base / f"{prefix}_{index:04d}" / "attempt_0000"
                result, error = verify_result(path)
                check(f"{label}: {prefix} artifact receipts", result is not None and error is None)
                if result is None or error or result.get("status") != "passed":
                    check(f"{label}: {prefix} generation accepted", False)
                    row["runs"].append(
                        dict(directory=str(path.relative_to(root)), error=error, result=result)
                    )
                    continue
                check(f"{label}: {prefix} requested seed/recipe", result["case"] == expected)
                check(f"{label}: {prefix} hash seed", result["python_hash_seed"] == hash_seed)
                check(f"{label}: {prefix} source version", result["source_sha256"] == source)
                if prefix == "replay":
                    check(
                        f"{label}: replay has no checkpoints",
                        not (path.parent / "checkpoints").exists(),
                    )
                    replay = result
                else:
                    original = result
                record = dict(
                    directory=str(path.relative_to(root)),
                    metrics=result["metrics"],
                    warnings=result["warnings"],
                    elapsed_s=result["elapsed_s"],
                    peak_rss_mib=result.get("peak_rss_mib"),
                    artifact_count=len(result["artifacts"]),
                    identity=result["identity"],
                )
                network = json.loads((path / "stage_b_network.json").read_text())
                sections = json.loads((path / "stage_c_sections.json").read_text())
                check(
                    f"{label}: {prefix} published B/C identities",
                    (
                        network["semantic_sha256"] == result["identity"]["network"]
                        and sections["semantic_sha256"] == result["identity"]["sections"]
                    ),
                )
                quality = json.loads((path / "network_quality.json").read_text())
                check(f"{label}: {prefix} network quality accepted", quality["accepted"])
                multi = Path(expected["config"]).name.startswith("multi_")
                check(
                    f"{label}: {prefix} requested source count retained",
                    network["summary_invariants"]["source_count"] == (3 if multi else 1),
                )
                if multi:
                    interaction = network["interconnection"]
                    record["interconnection"] = {
                        k: v
                        for k, v in interaction["metrics"].items()
                        if isinstance(v, (float, int, str))
                    }
                    check(
                        f"{label}: {prefix} parallel passage requirements",
                        all(
                            interaction["metrics"][name] >= interaction["controls"][minimum]
                            for name, minimum in (
                                ("parallel_fraction", "minimum_parallel_fraction"),
                                (
                                    "minimum_window_parallel_fraction",
                                    "minimum_window_parallel_fraction",
                                ),
                            )
                        ),
                    )
                record["rejected_network_candidates"] = sum(
                    not a["accepted"] for a in quality["attempts"]
                )
                if expected["scope"] == "full":
                    configured = load_project_config(
                        expected["config"],
                        world_body=expected["body"],
                        seed_override=expected["seed"],
                    )
                    recovery = json.loads((path / "pipeline_recovery.json").read_text())
                    check(
                        f"{label}: {prefix} accepted recovery context",
                        recovery["accepted"]
                        and all(
                            recovery["accepted_identity"][k] == result["identity"][k]
                            for k in ("network", "sections")
                        ),
                    )
                    check(
                        f"{label}: {prefix} original host preserved",
                        recovery["host_unchanged"]
                        and recovery["host_semantic_sha256"] == result["identity"]["host"],
                    )
                    check(
                        f"{label}: {prefix} root seed preserved",
                        recovery["root_seed"] == expected["seed"],
                    )
                    budgets = recovery["budgets"]
                    check(
                        f"{label}: {prefix} upstream attempts bounded",
                        len(recovery["attempts"])
                        <= 1 + budgets["local_attempts"] + budgets["network_attempts"],
                    )
                    replacements = [
                        a for a in recovery["attempts"] if a["kind"] == "network_regeneration"
                    ]
                    check(
                        f"{label}: {prefix} replacement sequence matches policy",
                        len(replacements) <= budgets["network_attempts"]
                        and all(
                            a["network_seed"]
                            == derive_subseed(
                                recovery["original_network_seed"], SEED_DOMAIN, a["index"]
                            )
                            for a in replacements
                        ),
                    )
                    check(
                        f"{label}: {prefix} surface attempts bounded",
                        all(
                            len(
                                a.get(
                                    "surface_attempts", a.get("inspection", {}).get("attempts", [])
                                )
                            )
                            <= 6
                            for a in recovery["attempts"]
                        ),
                    )
                    inspection = json.loads((path / "pipeline_quality_report.json").read_text())
                    export = inspection["export_inspection"]
                    texture = inspection["texture_recovery"]
                    check(
                        f"{label}: {prefix} final inspection",
                        inspection["passed"]
                        and export["passed"]
                        and export["serialized"]["passed"],
                    )
                    check(
                        f"{label}: {prefix} sampled passages preserved",
                        not export["visual"]["outside_centers"],
                    )
                    check(
                        f"{label}: {prefix} texture inspection",
                        texture["passed"] and texture["package_attempts"][-1]["passed"],
                    )
                    check(
                        f"{label}: {prefix} actual map resolution matches recipe",
                        all(
                            asset["prepared_size"]
                            == [
                                min(size, configured.geometry.embedded_texture_max_size)
                                for size in asset["source_size"]
                            ]
                            for asset in texture["assets"]
                            if asset["role"] != "displacement"
                        ),
                    )
                    check(
                        f"{label}: {prefix} texture package attempts bounded",
                        len(texture["package_attempts"]) <= 2,
                    )
                    asset_checks = json.loads((path / "asset_checks.json").read_text())
                    check(
                        f"{label}: {prefix} all portable checks",
                        all(c["passed"] for c in asset_checks),
                    )
                    topology = json.loads((path / "export_topology.json").read_text())
                    graph_loops = len(network["segments"]) - len(network["nodes"]) + 1
                    check(
                        f"{label}: {prefix} exported topology matches graph",
                        topology["components"] == 1 and topology["genus"] == graph_loops,
                    )
                    record.update(
                        portable_checks=len(asset_checks),
                        topology=topology,
                        recovery=recovery,
                        relief=inspection["surface_repairs"][-1],
                        texture_assets=texture["assets"],
                        visual_attempts=export["visual_attempts"],
                        under_resolved=inspection["resolution"]["under_resolved_count"],
                        section_count=inspection["resolution"]["section_count"],
                        package_bytes=sum(
                            p.stat().st_size for p in (path / "export").rglob("*") if p.is_file()
                        ),
                    )
                    values = [
                        m["clearance_m"]
                        for m in export["visual"]["measurements"]
                        if m["clearance_m"] is not None
                    ]
                    record["minimum_sampled_clearance_m"] = min(values) if values else None
                    check(
                        f"{label}: {prefix} simulation budgets",
                        result["metrics"]["triangles"] <= 5_000_000
                        and result["metrics"]["asset_bytes"] <= 350_000_000,
                    )
                    record["texture_recovery"] = texture
                    record["uv_repair_observed"] = (
                        '"UV repair"' in (path / "progress.jsonl").read_text()
                    )
                row["runs"].append(record)
            check(
                f"{label}: exact cold replay",
                original is not None
                and replay is not None
                and original["identity"] == replay["identity"],
            )
            rows.append(row)
        group_rows = [row for row in rows if row["group"] == group]
        for identity in ("network", "glb") if cases[0]["scope"] == "full" else ("network",):
            values = [row["runs"][0].get("identity", {}).get(identity) for row in group_rows]
            check(
                f"{group}: distinct accepted {identity} identities",
                None not in values and len(set(values)) == len(cases),
            )
    native_protocol = json.loads((root / "native_protocol_v2.json").read_text())
    for filename, digest in native_protocol["fixtures_sha256"].items():
        check(f"Frozen native inspection: {filename}", sha256_file(filename) == digest)
    native_path = root / "native_campaign.json"
    native = json.loads(native_path.read_text())
    check("Native cohort complete", native["complete"])
    check("Every planned original has a native outcome", len(native["cases"]) == len(rows))
    native_receipts = {}
    for row in rows:
        label = f"{row['group']}: seed {row['case']['seed']}"
        matches = [
            r for r in native["cases"] if r["group"] == row["group"] and r["index"] == row["index"]
        ]
        check(f"{label}: one native outcome", len(matches) == 1)
        if len(matches) != 1:
            continue
        result = matches[0]
        row["native"] = result
        check(f"{label}: native engines accepted", result["passed"])
        if not result["passed"]:
            continue
        folder = Path(result["output"])
        receipt = json.loads((folder / "native_input_receipt.json").read_text())
        run = root / row["group"] / f"case_{row['index']:04d}" / "attempt_0000"
        check(
            f"{label}: native input is this exact accepted mesh",
            receipt["glb_sha256"]
            == sha256_file(run / "export/plume_cave.glb")
            == result["input_glb_sha256"],
        )
        check(
            f"{label}: native inspection uses final export",
            receipt["quality_sha256"] == sha256_file(run / "pipeline_quality_report.json")
            and receipt["measurements_source"] == "export_inspection.visual",
        )
        check(f"{label}: native adapter version", receipt["adapter_source"] == source)
        for fixture, digest in receipt["fixture_sha256"].items():
            check(
                f"{label}: frozen fixture {fixture}",
                digest == sha256_file(Path("tests/fixtures") / fixture),
            )
        checks_native = result["summary"]["checks"]
        check(f"{label}: both engines inspected", set(checks_native) == {"unity", "unreal"})
        for engine, evidence in checks_native.items():
            report = evidence["native"]
            unity = engine == "unity"
            results_folder = folder / ("unity_project" if unity else "unreal_run_01")
            receipt_files = [
                results_folder / "native_result.json",
                folder / "native_summary.json",
                folder / "native_input_receipt.json",
            ]
            for filename, expected_stats in evidence["images"].items():
                capture = results_folder / filename
                with Image.open(capture) as picture:
                    check(
                        f"{label}: {engine} capture dimensions {filename}",
                        picture.size == (960, 640),
                    )
                    rgb = np.asarray(picture.convert("RGB"), dtype=np.float64) / 255
                actual = {
                    "mean": float(rgb.mean()),
                    "standard_deviation": float(rgb.std()),
                    "clipped_fraction": float(np.all(rgb >= 254 / 255, axis=2).mean()),
                }
                check(
                    f"{label}: {engine} image measurements unchanged {filename}",
                    all(
                        np.isclose(actual[k], v, rtol=0, atol=1e-12)
                        for k, v in expected_stats.items()
                    ),
                )
                receipt_files.append(capture)
            receipt_files.append(results_folder / "normal_off_control.png")
            if unity:
                receipt_files.extend(
                    [
                        results_folder / "uv_control.png",
                        results_folder / "Assets/PLUME_Inspection.unity",
                    ]
                )
            else:
                receipt_files.append(
                    folder / "unreal_project/Content/PLUME_Run01/PLUME_Inspection.umap"
                )
            for artifact in receipt_files:
                native_receipts[str(artifact.relative_to(root))] = sha256_file(artifact)
            count = report["passagePassed" if unity else "passage_passed"]
            check(f"{label}: {engine} all passage rays", count == receipt["samples"])
            check(
                f"{label}: {engine} exact triangle count",
                report["triangles"] == receipt["triangles"],
            )
            check(
                f"{label}: {engine} maps remain 4K",
                len(report["textures"]) == 3
                and all(t["width"] == t["height"] == 4096 for t in report["textures"]),
            )
            check(
                f"{label}: {engine} normal map changes shading",
                report["normalRenderDifference" if unity else "normal_render_difference"] > 0.0001,
            )
            check(f"{label}: {engine} two actual captures", len(evidence["images"]) == 2)
            check(
                f"{label}: {engine} bounded clipping",
                all(image["clipped_fraction"] <= 0.01 for image in evidence["images"].values()),
            )
            if unity:
                check(
                    f"{label}: Unity independent of UV charts", report["uvRenderDifference"] < 0.001
                )
                check(f"{label}: Unity no shader errors", not report["shaderErrors"])
            else:
                check(f"{label}: Unreal shader actually compiled", report["shader_count"] > 0)
    (root / "native_output_receipts.json").write_text(json.dumps(native_receipts, indent=2) + "\n")
    audit = dict(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        cases=rows,
        source_sha256=source,
        native_artifact_count=len(native_receipts),
        protocol_sha256=sha256_file(args.protocol),
        verified_artifacts=sum(r.get("artifact_count", 0) for row in rows for r in row["runs"]),
        scope="Frozen numerical, repair, cold-replay and native import/material evidence; visual review and tests recorded separately",
    )
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("passed", "verified_artifacts", "source_sha256")}))
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} audit checks passed")
    for item in checks:
        if not item["passed"]:
            print("FAILED:", item["name"])
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
