import json

import numpy as np
import polars as pl
import pytest

from fresh_demand.active.experiment import simulate_policy
from fresh_demand.active.service import plan_next


def files(tmp_path):
    p = np.array([[.1, .2, .3, .2, .1, .05, .05], [.1, .1, .1, .2, .2, .2, .1]])
    pmfs = np.repeat(p[None], 12, axis=0)
    rng = np.random.default_rng(10)
    warmup = rng.integers(0, 7, (10, 2))
    demand = rng.integers(0, 7, (12, 2))
    config = {"underage": 3, "overage": 1, "extra_budget": 2, "initial_order": 2,
              "delta": .1, "max_demand": 6, "n_contexts": 2, "horizon": 12, "warmup": 10}
    _, _, observations = simulate_policy(pmfs, demand, warmup, "decision", config, 11, 16, True)
    history = tmp_path / "history.csv"
    observations.filter(pl.col("round") < 6).write_csv(history)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"planning_samples": 16}))
    return history, config_path, selection, observations


def test_serving_reconstructs_exact_experiment_state_without_demand(tmp_path):
    history, config, selection, observations = files(tmp_path)
    result = plan_next(history, config, selection, tmp_path / "plan.json", "decision", 11)
    expected = observations.filter(pl.col("round") == 6).sort("context")["order"].to_list()
    assert [row["order"] for row in result["contexts"]] == expected
    assert result["next_round"] == 6 and result["oracle_data_used"] is False
    assert "demand" not in pl.read_csv(history).columns


def test_incomplete_or_missing_capacity_history_is_rejected(tmp_path):
    history, config, selection, _ = files(tmp_path)
    data = pl.read_csv(history)
    data.slice(1).write_csv(history)
    with pytest.raises(ValueError, match="complete"):
        plan_next(history, config, selection, tmp_path / "plan.json")
    data.drop("order").write_csv(history)
    with pytest.raises(ValueError, match="columns"):
        plan_next(history, config, selection, tmp_path / "plan.json")
