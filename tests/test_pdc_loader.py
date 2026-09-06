import json
from pathlib import Path

from plume_advanced.evaluation.datasets.pdc import audit_pdc, load_pdc


def test_pdc_loader_preserves_ids_and_records_every_rejection(tmp_path: Path) -> None:
    cave = tmp_path / "Cave_Alpha"
    cave.mkdir()
    (cave / "section_closed.txt").write_text("x y\n0 0\n2, 0\n2; 1\n0 1\n0 0\n", encoding="utf-8")
    (cave / "section_open.txt").write_text("0 0\n2 0\n2 1\n0 1\n", encoding="utf-8")
    (cave / "degenerate.txt").write_text("0 0\n1 0\n2 0\n", encoding="utf-8")

    sections, rejections = load_pdc(tmp_path)
    assert len(sections) == 1
    assert sections[0].reference_cave_id == "Cave_Alpha"
    assert sections[0].reference_section_id == "section_closed"
    assert {rejection.reason for rejection in rejections} == {"open_contour", "degenerate"}

    audit = audit_pdc(tmp_path, tmp_path / "outputs")
    assert audit["total_candidate_files"] == 3
    assert audit["accepted_contours"] == 1
    assert audit["rejected_contours"] == 2
    assert json.loads((tmp_path / "outputs" / "pdc_audit.json").read_text())["unique_cave_ids"] == 1


def test_pdc_implicit_closure_is_explicit_opt_in(tmp_path: Path) -> None:
    (tmp_path / "open.txt").write_text("0 0\n1 0\n1 1\n0 1\n", encoding="utf-8")
    strict, _ = load_pdc(tmp_path)
    permissive, _ = load_pdc(tmp_path, implicit_closure=True)
    assert strict == []
    assert len(permissive) == 1


def test_pdc_repeated_prefix_marks_one_explicit_cycle(tmp_path: Path) -> None:
    path = tmp_path / "cyclic_export.txt"
    path.write_text(
        "X,Y\n0,0\n2,0\n2,1\n0,1\n0,0\n2,0\n2,1\n",
        encoding="utf-8",
    )

    sections, rejections = load_pdc(tmp_path)

    assert rejections == []
    assert len(sections) == 1
    assert sections[0].was_open is False
    assert sections[0].trailing_cycle_point_count == 2
    assert len(sections[0].raw_points) == 7
    assert len(sections[0].contour) == 5


def test_pdc_loader_preserves_numeric_station_order(tmp_path: Path) -> None:
    cave = tmp_path / "Cave_Stations"
    cave.mkdir()
    contour = "0 0\n2 0\n2 1\n0 1\n0 0\n"
    for station in (10, 2, 1):
        (cave / f"cross-section_{station}.txt").write_text(contour, encoding="utf-8")

    sections, rejections = load_pdc(tmp_path)

    assert rejections == []
    assert [section.reference_section_id for section in sections] == [
        "cross-section_1",
        "cross-section_2",
        "cross-section_10",
    ]
