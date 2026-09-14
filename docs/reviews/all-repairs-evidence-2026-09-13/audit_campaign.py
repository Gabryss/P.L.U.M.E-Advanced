"""Read-only evidence audit for this campaign; write only a separate audit.json."""

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("protocol_final.json"))
    args = parser.parse_args()
    root = args.directory.resolve()
    protocol = json.loads(args.protocol.read_text())
    checks, rows = [], []

    def check(label, passed):
        checks.append({"name": label, "passed": bool(passed)})

    source = package_source_hash()
    check("Frozen production source", source == protocol["source_sha256"])
    for group, cases in protocol["groups"].items():
        base = root / group
        summary = json.loads((base / "summary.json").read_text())
        check(f"{group}: complete and accepted", summary["complete"] and summary["passed"])
        check(f"{group}: every planned case retained", len(summary["cases"]) == len(cases))
        plan = json.loads((base / "plan.json").read_text())
        check(f"{group}: source, inputs, runtime and recipe unchanged", plan == plan_identity(
            [ReliabilityCase(**c) for c in cases], source=source, replay=True
        ))
        for index, expected in enumerate(cases):
            label = f"{group}/{index}: seed {expected['seed']}"
            original, replay = None, None
            row = dict(group=group, index=index, case=expected, runs=[])
            for prefix, hash_seed in (("case", 11), ("replay", 37)):
                # No automatic replacement of missing/failed evidence by this auditor.
                path = base / f"{prefix}_{index:04d}" / "attempt_0000"
                result, error = verify_result(path)
                check(f"{label}: {prefix} artifact receipts", result is not None and error is None)
                if result is None:
                    row["runs"].append(dict(directory=str(path.relative_to(root)), error=error))
                    continue
                check(f"{label}: {prefix} requested seed/recipe", result["case"] == expected)
                check(f"{label}: {prefix} hash seed", result["python_hash_seed"] == hash_seed)
                check(f"{label}: {prefix} source version", result["source_sha256"] == source)
                if prefix == "replay":
                    check(f"{label}: replay has no checkpoints", not (path.parent / "checkpoints").exists())
                    replay = result
                else:
                    original = result
                record = dict(directory=str(path.relative_to(root)), metrics=result["metrics"],
                              warnings=result["warnings"], elapsed_s=result["elapsed_s"],
                              peak_rss_mib=result.get("peak_rss_mib"),
                              artifact_count=len(result["artifacts"]), identity=result["identity"])
                network = json.loads((path / "stage_b_network.json").read_text())
                sections = json.loads((path / "stage_c_sections.json").read_text())
                check(f"{label}: {prefix} published B/C identities", (
                    network["semantic_sha256"] == result["identity"]["network"]
                    and sections["semantic_sha256"] == result["identity"]["sections"]
                ))
                quality = json.loads((path / "network_quality.json").read_text())
                check(f"{label}: {prefix} network quality accepted", quality["accepted"])
                multi = Path(expected["config"]).name.startswith("multi_")
                check(f"{label}: {prefix} requested source count retained",
                      network["summary_invariants"]["source_count"] == (3 if multi else 1))
                if multi:
                    interaction = network["interconnection"]
                    record["interconnection"] = {k: v for k, v in interaction["metrics"].items()
                                                 if isinstance(v, (float, int, str))}
                    check(f"{label}: {prefix} parallel passage requirements", all(
                        interaction["metrics"][name] >= interaction["controls"][minimum]
                        for name, minimum in (("parallel_fraction", "minimum_parallel_fraction"),
                                              ("minimum_window_parallel_fraction", "minimum_window_parallel_fraction"))
                    ))
                record["rejected_network_candidates"] = sum(not a["accepted"] for a in quality["attempts"])
                if expected["scope"] == "full":
                    configured = load_project_config(expected["config"], world_body=expected["body"],
                                                     seed_override=expected["seed"])
                    recovery = json.loads((path / "pipeline_recovery.json").read_text())
                    check(f"{label}: {prefix} accepted recovery context", recovery["accepted"] and all(
                        recovery["accepted_identity"][k] == result["identity"][k] for k in ("network", "sections")
                    ))
                    check(f"{label}: {prefix} original host preserved", recovery["host_unchanged"] and
                          recovery["host_semantic_sha256"] == result["identity"]["host"])
                    check(f"{label}: {prefix} root seed preserved", recovery["root_seed"] == expected["seed"])
                    budgets = recovery["budgets"]
                    check(f"{label}: {prefix} upstream attempts bounded", len(recovery["attempts"]) <=
                          1 + budgets["local_attempts"] + budgets["network_attempts"])
                    replacements = [a for a in recovery["attempts"] if a["kind"] == "network_regeneration"]
                    check(f"{label}: {prefix} replacement sequence matches policy",
                          len(replacements) <= budgets["network_attempts"] and all(
                              a["network_seed"] == derive_subseed(recovery["original_network_seed"], SEED_DOMAIN, a["index"])
                              for a in replacements))
                    check(f"{label}: {prefix} surface attempts bounded", all(
                        len(a.get("surface_attempts", a.get("inspection", {}).get("attempts", []))) <= 6
                        for a in recovery["attempts"]
                    ))
                    inspection = json.loads((path / "pipeline_quality_report.json").read_text())
                    export = inspection["export_inspection"]
                    texture = inspection["texture_recovery"]
                    check(f"{label}: {prefix} final inspection", inspection["passed"] and
                          export["passed"] and export["serialized"]["passed"])
                    check(f"{label}: {prefix} sampled passages preserved", not export["visual"]["outside_centers"])
                    check(f"{label}: {prefix} texture inspection", texture["passed"] and
                          texture["package_attempts"][-1]["passed"])
                    check(f"{label}: {prefix} actual map resolution matches recipe", all(
                        asset["prepared_size"] == [min(size, configured.geometry.embedded_texture_max_size)
                                                   for size in asset["source_size"]]
                        for asset in texture["assets"] if asset["role"] != "displacement"
                    ))
                    check(f"{label}: {prefix} texture package attempts bounded", len(texture["package_attempts"]) <= 2)
                    asset_checks = json.loads((path / "asset_checks.json").read_text())
                    check(f"{label}: {prefix} all portable checks", all(c["passed"] for c in asset_checks))
                    topology = json.loads((path / "export_topology.json").read_text())
                    graph_loops = len(network["segments"]) - len(network["nodes"]) + 1
                    check(f"{label}: {prefix} exported topology matches graph",
                          topology["components"] == 1 and topology["genus"] == graph_loops)
                    record.update(portable_checks=len(asset_checks), topology=topology,
                                  recovery=recovery, relief=inspection["surface_repairs"][-1],
                                  texture_assets=texture["assets"],
                                  visual_attempts=export["visual_attempts"],
                                  under_resolved=inspection["resolution"]["under_resolved_count"],
                                  section_count=inspection["resolution"]["section_count"],
                                  package_bytes=sum(p.stat().st_size for p in (path / "export").rglob("*") if p.is_file()))
                    values = [m["clearance_m"] for m in export["visual"]["measurements"] if m["clearance_m"] is not None]
                    record["minimum_sampled_clearance_m"] = min(values) if values else None
                    check(f"{label}: {prefix} simulation budgets", result["metrics"]["triangles"] <= 5_000_000
                          and result["metrics"]["asset_bytes"] <= 350_000_000)
                    record["uv_repair_observed"] = '"UV repair"' in (path / "progress.jsonl").read_text()
                row["runs"].append(record)
            check(f"{label}: exact cold replay", original is not None and replay is not None and
                  original["identity"] == replay["identity"])
            rows.append(row)
        group_rows = [row for row in rows if row["group"] == group]
        for identity in (("network", "glb") if cases[0]["scope"] == "full" else ("network",)):
            values = [row["runs"][0].get("identity", {}).get(identity) for row in group_rows]
            check(f"{group}: distinct accepted {identity} identities",
                  None not in values and len(set(values)) == len(cases))
    audit = dict(passed=all(c["passed"] for c in checks), checks=checks, cases=rows,
                 source_sha256=source, protocol_sha256=sha256_file(args.protocol),
                 verified_artifacts=sum(r.get("artifact_count", 0) for row in rows for r in row["runs"]),
                 scope="Numerical repair/replay and artifact audit; tests and native rendering are separate evidence")
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("passed", "verified_artifacts", "source_sha256")}))
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} audit checks passed")
    for item in checks:
        if not item["passed"]:
            print("FAILED:", item["name"])
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
