"""A byte-limit failure retains checked work without relaxing geometry acceptance."""

import json
from pathlib import Path

import pytest
from test_embedded_inspection import geometry

from plume_advanced.exporters import export_target_asset
from plume_advanced.exporters.errors import ExportBudgetError
from plume_advanced.exporters.package_recovery import recover_size_rejected_package
from plume_advanced.identity import sha256_file
from plume_advanced.reexport import main
from plume_advanced.world import ExportConfig


@pytest.fixture
def retained(tmp_path):
    with pytest.raises(ExportBudgetError) as caught:
        export_target_asset(geometry(smoothing=0),
            ExportConfig(target="all", file_format="auto", max_asset_bytes=16), tmp_path / "export")
    assert not (tmp_path / "export").exists()
    return Path(caught.value.report["preserved_package"])


def test_recovery_reuses_every_asset_without_preparing_or_exporting_geometry(retained, tmp_path, monkeypatch):
    import plume_advanced.exporters.targets as targets

    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must not repeat geometry preparation or export")

    monkeypatch.setattr(targets, "prepare_export_scene", forbidden)
    monkeypatch.setattr(targets, "_export_target_asset_in_place", forbidden)
    original = {path.relative_to(retained): sha256_file(path) for path in retained.rglob("*") if path.is_file()}
    output = tmp_path / "recovered"
    assert main(["--recover-package", str(retained), "--output", str(output),
                 "--max-asset-bytes", "10000000"]) == 0
    for relative, digest in original.items():
        assert sha256_file(retained / relative) == digest
        if relative.suffix in (".obj", ".glb", ".usd", ".png", ".mtl"):
            assert sha256_file(output / relative) == digest
    report = json.loads((output / "export_size_report.json").read_text())
    assert report["passed"] and report["limits"]["asset_bytes"] == 10000000
    assert not report["oversized_files"]
    for name, size in report["files"].items():
        assert (output / name).stat().st_size == size
    assert report["total_file_bytes"] == sum(report["files"].values())
    assert not json.loads((output / "robot_qualification.json").read_text())["qualified"]
    manifest = json.loads((output / "plume_cave.all_exports.json").read_text())
    assert "export_recovery.json" in manifest["shared_files"]
    assert not (output / "size_recovery.json").exists()


@pytest.mark.parametrize("damage", ["asset", "inspection", "missing", "extra", "symlink", "receipt", "path"])
def test_damaged_or_incomplete_packages_cannot_be_recovered(retained, tmp_path, damage):
    asset = retained / "blender/plume_cave.glb"
    if damage == "asset":
        payload = bytearray(asset.read_bytes())
        payload[-1] ^= 1
        asset.write_bytes(payload)
    elif damage == "inspection":
        report = retained / "pipeline_inspection.json"
        report.write_text(report.read_text().replace('"passed": true', '"passed":false', 1))
    elif damage == "missing":
        asset.unlink()
    elif damage == "extra":
        (retained / "unexpected").write_text("extra")
    elif damage == "symlink":
        asset.unlink()
        asset.symlink_to(tmp_path / "outside")
    else:
        path = retained / "size_recovery.json"
        receipt = json.loads(path.read_text())
        if damage == "receipt":
            receipt["status"] = "incomplete"
        else:
            receipt["files"]["../outside"] = dict(bytes=1, sha256="invalid")
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        recover_size_rejected_package(retained, tmp_path / "rejected", max_asset_bytes=10000000)
    assert not (tmp_path / "rejected").exists()
    assert retained.exists()


def test_insufficient_new_budget_keeps_retained_package(retained, tmp_path):
    with pytest.raises(ExportBudgetError, match="still exceeded"):
        recover_size_rejected_package(retained, tmp_path / "rejected", max_asset_bytes=16)
    assert retained.is_dir() and not (tmp_path / "rejected").exists()


def test_size_failure_preserves_previous_published_output(tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    (output / "existing").write_text("previous complete result")
    with pytest.raises(ExportBudgetError):
        export_target_asset(geometry(smoothing=0), ExportConfig(max_asset_bytes=16), output)
    assert (output / "existing").read_text() == "previous complete result"
    assert len(list(tmp_path.glob("export.size-rejected-*"))) == 1


@pytest.mark.parametrize("args", [[], ["--max-asset-bytes", "10000000", "--max-visual-triangles", "100"]])
def test_cli_recovery_requires_explicit_size_only_review(retained, tmp_path, args):
    with pytest.raises(SystemExit) as caught:
        main(["--recover-package", str(retained), "--output", str(tmp_path / "rejected"), *args])
    assert caught.value.code == 2
    assert not (tmp_path / "rejected").exists()


def test_recovery_cannot_replace_or_nest_in_source(retained):
    for destination in (retained, retained / "nested", retained.parent):
        with pytest.raises(ValueError, match="new output"):
            recover_size_rejected_package(retained, destination, max_asset_bytes=10000000)


def test_checksum_interruption_keeps_assets_but_no_recoverable_receipt(tmp_path, monkeypatch):
    import plume_advanced.exporters.package_recovery as recovery

    def interrupted(*args):
        raise OSError("Synthetic checksum interruption")

    monkeypatch.setattr(recovery, "sha256_file", interrupted)
    with pytest.raises(OSError, match="Synthetic"):
        export_target_asset(geometry(smoothing=0), ExportConfig(max_asset_bytes=16), tmp_path / "export")
    package = next(tmp_path.glob("export.size-rejected-*"))
    assert list(package.glob("*.glb"))
    assert not (package / "size_recovery.json").exists()


def test_recovery_preserves_finite_budget_requirement_and_updates_evidence(tmp_path):
    from plume_advanced.acceptance import AcceptancePolicy

    with pytest.raises(ExportBudgetError) as caught:
        export_target_asset(geometry(smoothing=0),
            ExportConfig(max_visual_triangles=10000, max_asset_bytes=16), tmp_path / "export",
            acceptance=AcceptancePolicy(require_export_budgets=True))
    package = Path(caught.value.report["preserved_package"])
    with pytest.raises(ValueError, match="finite"):
        recover_size_rejected_package(package, tmp_path / "unlimited", max_asset_bytes=0)
    output = tmp_path / "recovered"
    recover_size_rejected_package(package, output, max_asset_bytes=10000000)
    inspection = json.loads((output / "pipeline_inspection.json").read_text())
    assert inspection["acceptance"]["checks"]["export_budgets"]["limits"] == dict(
        visual_triangles=10000, asset_bytes=10000000)


def test_failed_copy_does_not_publish_or_damage_retained_package(retained, tmp_path, monkeypatch):
    import plume_advanced.exporters.package_recovery as recovery

    copy = recovery.shutil.copyfile

    def corrupt(source, destination):
        copy(source, destination)
        Path(destination).write_bytes(b"damaged during copy")

    monkeypatch.setattr(recovery.shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="while copying"):
        recover_size_rejected_package(retained, tmp_path / "rejected", max_asset_bytes=10000000)
    assert not (tmp_path / "rejected").exists()
    assert (retained / "size_recovery.json").is_file()
    assert not list(tmp_path.glob(".rejected.staging-*"))
