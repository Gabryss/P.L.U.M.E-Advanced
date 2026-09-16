"""Keep the single guide, recipe examples and installable resources usable."""

import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from plume_advanced.config import load_project_config

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_links_and_sections():
    text = (ROOT / "README.md").read_text()
    assert re.findall(r"^## (.+)$", text, re.MULTILINE) == [
        "Introduction",
        "Generation examples",
        "Installation & usage",
        "Simulators",
        "Config file",
        "Architecture",
        "Limits",
    ]
    headings = re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE)
    anchors = {re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-") for heading in headings}
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        url = urlsplit(target)
        if url.scheme:
            continue
        if url.path:
            assert (ROOT / unquote(url.path)).exists(), target
        if url.fragment and not url.path:
            assert url.fragment in anchors, target


def test_readme_toml_examples_resolve(tmp_path):
    blocks = re.findall(r"```toml\n(.*?)```", (ROOT / "README.md").read_text(), re.DOTALL)
    assert blocks
    for index, block in enumerate(blocks):
        raw = tomllib.loads(block)
        if not {"recipe_version", "schema_version"} & raw.keys():
            block = 'recipe_version = 1\npreset = "preview"\n' + block
        path = tmp_path / f"example-{index}.toml"
        path.write_text(block)
        load_project_config(path)


def test_maintained_sources_have_one_markdown_guide():
    for directory in ("src", "scripts", "config", "tests", "docs"):
        assert not list((ROOT / directory).rglob("*.md")), directory


def test_repository_excludes_retired_evaluation_archives():
    for directory in ("docs/reviews", "docs/maintenance", "paper/campaigns", "paper/overleaf"):
        assert not (ROOT / directory).exists(), directory


def test_removed_animation_workflow_has_no_advertised_entrypoints():
    for name in ("assemble_manim_video.py", "prepare_manim_assets.py", "render_manim_video.py"):
        assert not (ROOT / "scripts" / name).exists()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "video" not in project["project"]["optional-dependencies"]
    assert not any(p.name.startswith("manim") for p in (ROOT / "src").rglob("*.py"))
    assert "--extra video" not in (ROOT / "README.md").read_text()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert "manim" not in {package["name"] for package in lock["package"]}


def test_packaged_recipe_resources_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    data = project["tool"]["setuptools"]["package-data"]["plume_advanced"]
    assert "presets.json" in data and "default_project.toml" in data
    assert "material_assets/*.txt" in data
    assert "evaluation/resources/*.toml" in data
    assert "evaluation/resources/seeds/*.txt" in data
    assert "evaluation/resources/splits/*.txt" in data
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["plume_advanced*"]


def test_simulator_gallery_is_archived_and_matches_receipts():
    """Clearing generation outputs must not remove or silently replace README captures."""
    directory = ROOT / "docs/simulators"
    gallery = json.loads((directory / "gallery.json").read_text())
    assert {row["application"] for row in gallery["images"]} == {
        "Blender", "Unity", "Unreal Engine", "Gazebo", "Isaac Sim",
    }
    readme = (ROOT / "README.md").read_text()
    for row in gallery["images"]:
        for role in ("image", "receipt", "source_receipt", "generation_receipt"):
            if role not in row:
                continue
            path = (ROOT / row[role]["path"]).resolve()
            assert path.is_relative_to(directory.resolve())
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[role]["sha256"]
        assert f']({row["image"]["path"]})' in readme
        assert row["captured_on"] and row["application_version"] and row["scope"]
