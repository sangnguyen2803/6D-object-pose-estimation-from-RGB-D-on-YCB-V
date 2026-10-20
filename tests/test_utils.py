"""Check errors and file handling in shared utilities."""

import pytest

from clutterpick.utils.common import load_config


@pytest.mark.parametrize("content", ["", "# empty config", "null", "- item", "42"])
def test_config_rejects_non_mapping_yaml(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration must be a YAML mapping") as error:
        load_config(path)
    assert str(path) in str(error.value)


def test_config_accepts_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\ndata:\n  name: ycbv\n", encoding="utf-8")
    assert load_config(path) == {"seed": 42, "data": {"name": "ycbv"}}
