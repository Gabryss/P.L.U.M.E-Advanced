"""Reject unrelated, modified or unverifiable assets even near valid manifests."""

import json

import pytest

from plume_advanced.identity import sha256_file
from plume_advanced.validation import PortableAssetValidator


def validator_case(tmp_path):
    source = tmp_path / "original.glb"
    source.write_bytes(b"original")
    validator = object.__new__(PortableAssetValidator)
    validator.asset_path = source
    validator.run_manifest_path = tmp_path / "run_manifest.json"
    validator.run_manifest = {
        "status": "complete",
        "outputs": [
            {"path": source.name, "sha256": sha256_file(source)},
        ],
    }
    return validator, source


def selected_passes(validator):
    return next(
        c.passed
        for c in validator._reproducibility_checks()
        if c.name == "Selected asset provenance"
    )


def test_exact_generated_asset_is_verified(tmp_path):
    validator, source = validator_case(tmp_path)
    assert selected_passes(validator)
    source.write_bytes(b"modified")
    assert not selected_passes(validator)


def test_unlisted_asset_is_not_attested_by_neighbor_manifest(tmp_path):
    validator, source = validator_case(tmp_path)
    other = tmp_path / "other.glb"
    other.write_bytes(source.read_bytes())
    validator.asset_path = other
    assert not selected_passes(validator)


@pytest.mark.parametrize(
    "damage", [None, "asset", "source", "hash", "unlisted_source", "malformed"]
)
def test_material_revision_chain(tmp_path, damage):
    validator, source = validator_case(tmp_path)
    directory = tmp_path / "revision"
    directory.mkdir()
    asset = directory / "revised.glb"
    asset.write_bytes(b"revised")
    validator.asset_path = asset
    revision = {
        "schema": "plume.material-revision.v1",
        "source_asset": str(source),
        "source_sha256": sha256_file(source),
        "asset_sha256": sha256_file(asset),
    }
    if damage == "asset":
        asset.write_bytes(b"changed")
    elif damage == "source":
        source.write_bytes(b"changed")
    elif damage == "hash":
        revision["asset_sha256"] = "incorrect"
    elif damage == "unlisted_source":
        copy = tmp_path / "copy.glb"
        copy.write_bytes(source.read_bytes())
        revision["source_asset"] = str(copy)
    (directory / "material_revision.json").write_text(
        "{" if damage == "malformed" else json.dumps(revision)
    )
    assert selected_passes(validator) is (damage is None)
