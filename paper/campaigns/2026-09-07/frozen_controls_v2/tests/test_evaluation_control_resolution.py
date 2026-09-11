"""Experimental interventions must behave like actual project-file edits."""
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from plume_advanced.config import load_project_config


@pytest.mark.parametrize("key,value", [
    ("distributary_tendency", value) for value in (.2, .5, .8)
] + [("duration_scale", value) for value in (.5, 1., 1.5)]
  + [("inflation", value) for value in (.2, .5, .8)]
  + [("supply_rate_scale", value) for value in (.7, 1., 1.3)])
def test_control_override_matches_project_file_edit(tmp_path, key, value):
    source = Path(__file__).resolve().parents[1] / "config/project.toml"
    original = source.read_text()
    edited, count = re.subn(rf"^{key}\s*=\s*[^\n]+", f"{key} = {value}",
                             original, count=1, flags=re.MULTILINE)
    assert count == 1
    variant = tmp_path / "project.toml"
    variant.write_text(edited)
    expected = load_project_config(variant, dev_mode=False)
    actual = load_project_config(source, dev_mode=False, flow_regime_overrides={key: value})
    # Asset paths differ because the edited fixture lives in a temporary folder;
    # all stages consumed by this experiment must resolve identically.
    for field in ("world", "flow_regime", "host_field", "network", "section_field"):
        assert asdict(getattr(actual, field)) == asdict(getattr(expected, field))
    x, y = actual.host_field.seed_point
    assert abs(x) < actual.host_field.grid.width / 2
    assert abs(y) < actual.host_field.grid.height / 2


def test_distributary_intervention_reaches_current_lobe_generator():
    source = Path(__file__).resolve().parents[1] / "config/project.toml"
    low = load_project_config(source, dev_mode=False,
                              flow_regime_overrides={"distributary_tendency": .2})
    high = load_project_config(source, dev_mode=False,
                               flow_regime_overrides={"distributary_tendency": .8})
    assert all(a < b for a, b in zip(low.network.lobe_growth.path_count,
                                    high.network.lobe_growth.path_count))


def test_unknown_flow_control_is_rejected():
    source = Path(__file__).resolve().parents[1] / "config/project.toml"
    with pytest.raises(ValueError, match="Unknown"):
        load_project_config(source, flow_regime_overrides={"made_up_control": 1.})
