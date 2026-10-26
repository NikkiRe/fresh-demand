import numpy as np
import pytest

from fresh_demand.active.baselines import kaplan_meier_pmf


@pytest.mark.parametrize("tail", ["uniform", "lower", "upper"])
def test_full_support_orders_recover_empirical_distribution_including_endpoint(tail):
    demand = [0, 1, 1, 2, 4, 5, 5]
    history = [(5, d, d >= 5) for d in demand]
    estimate = kaplan_meier_pmf(history, 5, tail)
    np.testing.assert_allclose(estimate, np.bincount(demand, minlength=6) / len(demand))


def test_censored_tail_has_explicit_uniform_lower_and_upper_completions():
    history = [(8, 5, False), (8, 8, True)]
    uniform = kaplan_meier_pmf(history, 12)
    expected = np.zeros(13)
    expected[5] = 0.5
    expected[8:] = 0.1
    np.testing.assert_allclose(uniform, expected)
    lower, upper = kaplan_meier_pmf(history, 12, "lower"), kaplan_meier_pmf(history, 12, "upper")
    np.testing.assert_allclose(lower[[5, 8]], [0.5, 0.5])
    np.testing.assert_allclose(upper[[5, 12]], [0.5, 0.5])
    assert lower.sum() == pytest.approx(1)
    assert upper.sum() == pytest.approx(1)


def test_censoring_is_removed_before_events_at_the_same_value():
    history = [(1, 1, True), (2, 1, False), (2, 2, True)]
    # At d=1: the censor at Q=1 is removed before the uncensored event at 1.
    # Risk=2 and deaths=1, rather than the events-first risk of 3.
    np.testing.assert_allclose(kaplan_meier_pmf(history, 3), [0, 0.5, 0.25, 0.25])


def test_censoring_at_zero_supplies_no_hazard_information():
    history = [(3, 0, False), (3, 1, False), (3, 3, True)]
    np.testing.assert_allclose(
        kaplan_meier_pmf(history, 5),
        kaplan_meier_pmf(history + [(0, 0, True)] * 50, 5),
    )


def test_empty_history_and_only_zero_orders_are_uninformative():
    for history in [[], [(0, 0, True)] * 3]:
        np.testing.assert_allclose(kaplan_meier_pmf(history, 3), [0.25] * 4)
        np.testing.assert_allclose(kaplan_meier_pmf(history, 3, "lower"), [1, 0, 0, 0])
        np.testing.assert_allclose(kaplan_meier_pmf(history, 3, "upper"), [0, 0, 0, 1])


@pytest.mark.parametrize("history", [
    [(2, 2, False)], [(2, 1, True)], [(0, 0, False)],
    [(5, 2, False)], [(2, -1, False)], [(2, 3, False)],
    [(2, 1, 0)], [(1.2, 1, False)],
])
def test_invalid_feedback_is_rejected(history):
    with pytest.raises(ValueError):
        kaplan_meier_pmf(history, 4)


def test_unknown_tail_mode_is_rejected():
    with pytest.raises(ValueError, match="tail"):
        kaplan_meier_pmf([], 4, "unidentified")
