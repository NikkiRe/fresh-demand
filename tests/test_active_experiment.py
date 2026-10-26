import numpy as np
import polars as pl

from fresh_demand.active.experiment import paired_comparison, simulate_policy


def fixture():
    p = np.array([[.1, .2, .3, .2, .1, .05, .05], [.1, .1, .1, .2, .2, .2, .1]])
    pmfs = np.repeat(p[None], 12, axis=0)
    rng = np.random.default_rng(10)
    warmup = rng.integers(0, 7, (10, 2))
    demand = rng.integers(0, 7, (12, 2))
    config = {"underage": 3, "overage": 1, "extra_budget": 2, "initial_order": 2, "delta": .1}
    return pmfs, demand, warmup, config


def test_future_and_oracle_pmf_do_not_affect_learner_actions():
    pmfs, demand, warmup, config = fixture()
    for policy in ["sales", "risk_uniform", "geometric_ts", "uniform", "uncertainty", "decision", "adaptive_cost", "myopic_cost"]:
        _, _, original = simulate_policy(pmfs, demand, warmup, policy, config, 11, 16, True)
        future = demand.copy()
        future[6:] = 6 - future[6:]
        changed_pmfs = pmfs[..., ::-1].copy()
        _, _, modified = simulate_policy(changed_pmfs, future, warmup, policy, config, 11, 16, True)
        a = original.filter((pl.col("phase") == "online") & (pl.col("round") <= 6))["order"].to_numpy()
        b = modified.filter((pl.col("phase") == "online") & (pl.col("round") <= 6))["order"].to_numpy()
        np.testing.assert_array_equal(a, b)


def test_expected_regret_is_nonnegative_and_budget_is_exact():
    pmfs, demand, warmup, config = fixture()
    for policy in ["uniform", "uncertainty", "decision"]:
        summary, rounds, _ = simulate_policy(pmfs, demand, warmup, policy, config, 3, 16)
        assert summary["cumulative_expected_regret"] >= -1e-10
        expected = np.minimum(2, rounds["available_extra_capacity"].to_numpy())
        np.testing.assert_array_equal(rounds["extra"].to_numpy(), expected)
        assert summary["extra_units"] == int(expected.sum())


def test_paired_bootstrap_preserves_pairing():
    frame = pl.DataFrame({
        "scenario": ["x"] * 6, "seed": [0, 1, 2, 0, 1, 2],
        "policy": ["a"] * 3 + ["b"] * 3,
        "cumulative_expected_regret": [8., 98., 998., 10., 100., 1000.],
    })
    result = paired_comparison(frame, "x", "a", "b", n_boot=100)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == -2


def test_context_paths_and_jobs_use_callers_directory_after_installation(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from fresh_demand.active.experiment import _job, _resolve_context_path

    monkeypatch.chdir(tmp_path)
    context = tmp_path / "inputs" / "contexts.json"
    context.parent.mkdir()
    context.write_text(json.dumps({"pmfs": [[.2] * 5], "metadata": {"fixture": True}}))
    relative = Path("inputs/contexts.json")
    assert _resolve_context_path(relative) == context
    assert _resolve_context_path(context) == context
    config = {
        "horizon": 4, "max_demand": 4, "n_contexts": 1,
        "warmup": 2, "initial_order": 1, "underage": 3, "overage": 1,
        "extra_budget": 1, "delta": .1, "policies": ["uniform", "decision"],
        "seeds": [7], "scenarios": ["fresh_empirical"], "context_path": str(relative),
    }
    relative_results = _job((config, "fresh_empirical", 7, 1))
    absolute_results = _job(({**config, "context_path": str(context)}, "fresh_empirical", 7, 1))
    assert relative_results[0] == absolute_results[0]
    assert relative_results[1].equals(absolute_results[1])


def test_source_provenance_hashes_executing_installed_package(tmp_path, monkeypatch):
    import hashlib

    from fresh_demand.active import experiment

    installed = tmp_path / "site-packages" / "fresh_demand" / "active"
    installed.mkdir(parents=True)
    files = {"experiment.py": b"# installed experiment\n", "statistics.py": b"# installed statistics\n"}
    for name, contents in files.items():
        (installed / name).write_bytes(contents)
    monkeypatch.setattr(experiment, "__file__", str(installed / "experiment.py"))
    assert experiment._source_hashes() == {
        f"src/fresh_demand/active/{name}": hashlib.sha256(contents).hexdigest()
        for name, contents in files.items()
    }
