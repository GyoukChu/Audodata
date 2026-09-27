import json

import pytest
from pydantic import ValidationError

from autodata.config import AppConfig, ModelEndpoint, load_config


@pytest.mark.parametrize("overrides", [
    {"weak_avg_max": -0.1}, {"weak_attempt_max": 1.1}, {"strong_avg_min": 2},
    {"strong_avg_max": -1}, {"gap_min": 1.01}, {"gap_min": float("nan")},
    {"strong_avg_min": float("inf")}, {"weak_avg_max": "not a number"}, {"typo": 0.5},
])
def test_bad_acceptance_overrides_fail_at_load(tmp_path, overrides):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {}, "acceptance_overrides": overrides}))
    with pytest.raises(ValidationError):
        load_config(path)


def test_valid_overrides_are_validated_and_do_not_change_presets(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {}, "acceptance_overrides": {"gap_min": "0.25"}}))
    config = load_config(path)
    assert config.acceptance.gap_min == 0.25
    assert AppConfig(models={}).acceptance.gap_min == 0.2


def test_unknown_preset_fails_at_load(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {}, "acceptance_preset": "typo"}))
    with pytest.raises(ValidationError, match="unknown acceptance preset"):
        load_config(path)


def test_run_seed_is_default_and_endpoint_seed_wins():
    endpoint = ModelEndpoint(base_url="http://fake/v1", model="m")
    config = AppConfig(models={"default": endpoint, "explicit": endpoint.model_copy(update={"seed": 0})},
                       run={"seed": 123})
    assert config.endpoint("default").seed == 123
    assert config.endpoint("explicit").seed == 0
    assert endpoint.seed is None
