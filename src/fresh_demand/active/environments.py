"""Known, bounded demand distributions for controlled inventory experiments.

The full PMF and demand draws are evaluator/oracle inputs. A policy should see
only its own censored observations, produced by the experiment runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _validate_pmfs(pmfs: np.ndarray) -> np.ndarray:
    values = np.asarray(pmfs, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or values.shape[-1] < 2:
        raise ValueError("pmfs must have shape [rounds, contexts, max_demand + 1]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("PMFs must be finite and nonnegative")
    if not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-10):
        raise ValueError("Each demand PMF must sum to one")
    return values


def _poisson(mean: float, max_demand: int) -> np.ndarray:
    """Poisson law conditioned on the declared finite support, without floors."""
    support = np.arange(max_demand + 1, dtype=float)
    log_mass = support * math.log(max(mean, 1e-8)) - mean - np.asarray([
        math.lgamma(value + 1) for value in support
    ])
    mass = np.exp(log_mass - log_mass.max())
    return mass / mass.sum()


def _point(value: int, max_demand: int) -> np.ndarray:
    mass = np.zeros(max_demand + 1)
    mass[int(np.clip(value, 0, max_demand))] = 1
    return mass


def _heterogeneous(n_contexts: int, max_demand: int) -> tuple[np.ndarray, list[dict]]:
    distributions, specifications = [], []
    for context in range(n_contexts):
        kind = context % 4
        if kind == 0:
            mean, tail_weight = max_demand * 0.10, 0.18
            mass = ((1 - tail_weight) * _poisson(mean, max_demand)
                    + tail_weight * _point(round(max_demand * 0.82), max_demand))
            description = {"type": "low_mean_rare_spike", "poisson_mean": mean,
                           "spike_weight": tail_weight, "spike_at": round(max_demand * 0.82)}
        elif kind == 1:
            low, high, low_weight = max_demand * 0.20, max_demand * 0.55, 0.60
            mass = low_weight * _poisson(low, max_demand) + (1 - low_weight) * _poisson(high, max_demand)
            description = {"type": "bimodal", "low_mean": low, "high_mean": high,
                           "low_weight": low_weight}
        elif kind == 2:
            mean, tail_weight = max_demand * 0.35, 0.05
            mass = ((1 - tail_weight) * _poisson(mean, max_demand)
                    + tail_weight * _point(max_demand, max_demand))
            description = {"type": "medium_mean_rare_endpoint", "poisson_mean": mean,
                           "endpoint_weight": tail_weight}
        else:
            mean, zero_weight = max_demand * 0.65, 0.25
            mass = ((1 - zero_weight) * _poisson(mean, max_demand)
                    + zero_weight * _point(0, max_demand))
            description = {"type": "high_mean_zero_inflated", "poisson_mean": mean,
                           "zero_weight": zero_weight}
        distributions.append(mass / mass.sum())
        specifications.append({"context": context, **description})
    return np.asarray(distributions), specifications


def make_environment(
    name: str,
    seed: int,
    horizon: int = 500,
    max_demand: int = 40,
    n_contexts: int = 4,
    context_path: Path | None = None,
) -> dict:
    """Construct [T,K,M+1] PMFs; seeds never change scenario parameters.

    ``fresh_empirical`` is iid resampling from fitted fully-stocked train sales,
    not replay of actual chronology or identifiable counterfactual demand.
    ``shift`` and ``fresh_shift`` are explicitly nonstationary stress tests.
    """
    if horizon < 2 or max_demand < 2 or n_contexts < 1:
        raise ValueError("Require horizon >= 2, max_demand >= 2, n_contexts >= 1")
    metadata = {
        "name": name, "seed": int(seed), "horizon": int(horizon),
        "max_demand": int(max_demand), "n_contexts": int(n_contexts),
        "support": [0, int(max_demand)], "parameters_depend_on_seed": False,
        "oracle_knows_true_pmf": True, "policy_access_to_true_pmf": False,
        "probability_floor": None, "stationary": True,
        "interpretation": "Controlled bounded synthetic demand, not a business counterfactual.",
    }
    if name in {"heterogeneous", "shift"}:
        base, specifications = _heterogeneous(n_contexts, max_demand)
        metadata["context_parameters"] = specifications
    elif name == "geometric":
        probabilities = (
            np.asarray([0.10, 0.18, 0.30, 0.45])
            if n_contexts == 4 else np.linspace(0.10, 0.45, n_contexts)
        )
        support = np.arange(max_demand + 1)
        base = np.asarray([
            probability * (1 - probability) ** support
            for probability in probabilities
        ])
        base[:, -1] = (1 - probabilities) ** max_demand
        metadata.update({
            "context_parameters": [{"geometric_success_probability": float(value)} for value in probabilities],
            "interpretation": "Correctly specified conjugate control: min(G, M), where G counts failures before the first geometric success.",
            "tail_mass_at_support_endpoint": "P(G >= max_demand), not P(G == max_demand)",
        })
    elif name == "identified":
        base = np.zeros((n_contexts, max_demand + 1))
        for context in range(n_contexts):
            probability_one = 0.10 + 0.65 * context / max(n_contexts - 1, 1)
            base[context, :2] = [1 - probability_one, probability_one]
        metadata.update({
            "context_parameters": [{"probability_demand_one": float(row[1])} for row in base],
            "actual_demand_support": [0, 1], "minimum_fully_observing_stock": 2,
            "interpretation": "Negative control: every demand is below an initial stock of at least two.",
        })
    elif name == "heavy_tail":
        support = np.arange(max_demand + 1, dtype=float)
        tail_start = max(2, round(max_demand * 0.25))
        tail = np.where(support >= tail_start, (support + 1) ** -1.2, 0.0)
        tail /= tail.sum()
        rows, specifications = [], []
        for context in range(n_contexts):
            tail_weight = 0.12 + 0.45 * context / max(n_contexts - 1, 1)
            center = max(1, round(max_demand * (0.06 + 0.02 * (context % 3))))
            core = (0.25 * _point(0, max_demand)
                    + 0.50 * _point(center, max_demand)
                    + 0.25 * _point(center + 1, max_demand))
            rows.append((1 - tail_weight) * core + tail_weight * tail)
            specifications.append({"context": context, "tail_weight": tail_weight,
                                   "tail_start": tail_start, "tail_power": 1.2, "core_center": center})
        base = np.asarray(rows)
        metadata.update({
            "context_parameters": specifications,
            "interpretation": "Bounded long-tail mixture; this is not an unbounded heavy-tail guarantee.",
        })
    elif name in {"fresh_empirical", "fresh_shift"}:
        if context_path is None:
            raise ValueError("context_path is required for an empirical environment")
        source_bytes = Path(context_path).read_bytes()
        context_model = json.loads(source_bytes)
        base = np.asarray(context_model["pmfs"], dtype=float)
        if base.shape != (n_contexts, max_demand + 1):
            raise ValueError("Empirical context model shape does not match n_contexts/max_demand")
        metadata.update({
            "context_model_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "context_model_metadata": context_model["metadata"],
            "interpretation": (
                "Semi-synthetic iid bootstrap of train-only, fully-stocked scaled sales. "
                "No observed natural-stockout demand or real policy counterfactual is supplied."
            ),
        })
    else:
        raise ValueError(f"Unknown environment: {name}")

    base = _validate_pmfs(base[None])[0]
    pmfs = np.repeat(base[None, :, :], horizon, axis=0)
    if name in {"shift", "fresh_shift"}:
        change = horizon // 2
        if name == "fresh_shift":
            permutation = np.roll(np.arange(n_contexts), 1)
            shifted = base[permutation]
            if n_contexts == 1:
                shifted = 0.70 * shifted + 0.30 * _point(max_demand, max_demand)[None]
        else:
            permutation = np.arange(n_contexts)[::-1]
            shifted = 0.70 * base[permutation] + 0.30 * _point(max_demand, max_demand)[None]
        pmfs[change:] = shifted
        metadata.update({
            "stationary": False, "change_round": change,
            "context_distribution_permutation": permutation.tolist(),
            "stationary_confidence_bounds_apply": False,
            "shift_interpretation": "A constructed regime change, not an empirical chronology.",
        })
    _validate_pmfs(pmfs)
    metadata["pmf_sha256"] = hashlib.sha256(pmfs.tobytes()).hexdigest()
    metadata["context_means_before_shift"] = (base @ np.arange(max_demand + 1)).tolist()
    metadata["context_q75_before_shift"] = np.argmax(np.cumsum(base, axis=-1) >= 0.75, axis=-1).tolist()
    return {"pmfs": pmfs, "metadata": metadata}


def sample_demands(pmfs: np.ndarray, seed: int) -> np.ndarray:
    """Inverse-CDF sampling using one uniform per round/context.

    Reuse this draw matrix across policies to obtain common random numbers.
    Identical seeds give identical uniforms even when the PMFs differ.
    """
    values = _validate_pmfs(pmfs)
    cumulative = np.cumsum(values, axis=-1)
    cumulative[..., -1] = 1.0
    uniforms = np.random.default_rng(seed).random(values.shape[:2])
    return (uniforms[..., None] >= cumulative).sum(axis=-1).astype(np.int64)
