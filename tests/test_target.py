"""Reweighting first-passage times onto experimental rates with a Poisson target."""
import csv
import os

import numpy as np
import pytest
from scipy import stats

from maxcal import target as mt


# ------------------------------------------------------------------------ weights
def test_target_weights_match_the_target():
    rng = np.random.default_rng(0)
    t = rng.exponential(100.0, 400)
    tau = 60.0                                        # ask for a 40% faster process
    w = mt.target_weights(t, tau)
    assert w.sum() == pytest.approx(1.0)
    assert np.sum(w * t) == pytest.approx(tau, rel=0.05)
    d = mt.diagnostics(t, w, tau)
    assert d["ks_D"] < 0.05                           # weighted CDF sits on the target
    assert 0.3 * t.size < d["neff"] <= t.size


def test_weights_are_flat_when_the_target_equals_the_model():
    rng = np.random.default_rng(1)
    t = rng.exponential(10.0, 500)
    w = mt.target_weights(t, t.mean())                # nothing to correct
    assert mt.diagnostics(t, w, t.mean())["neff"] > 0.95 * t.size
    assert w.max() / w.min() < 1.5


def test_weights_grow_with_time_for_a_slower_target():
    rng = np.random.default_rng(5)
    t = rng.exponential(10.0, 300)
    w = mt.target_weights(t, 1.8 * t.mean(), match_mean=False)
    assert stats.spearmanr(t, w).statistic > 0.99
    d = mt.diagnostics(t, w, 1.8 * t.mean())
    assert d["tail_gap"] < 0.6 and d["neff"] > 0.4 * t.size


def test_mean_weights_are_an_exponential_tilt():
    rng = np.random.default_rng(2)
    t = rng.exponential(5.0, 500)
    w = mt.mean_weights(t, 3.0)
    assert np.sum(w * t) == pytest.approx(3.0, rel=1e-6)
    # w ∝ exp(-θ t): log-weights linear in t
    p = np.polyfit(t, np.log(w), 1)
    assert np.corrcoef(np.polyval(p, t), np.log(w))[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert mt.mean_weights(t, 10 * t.max()) is None    # unreachable target


def test_neff_fraction_and_clock_modes():
    assert mt.neff_fraction(1.0) == pytest.approx(1.0)
    assert mt.neff_fraction(0.5) == pytest.approx(0.75)
    assert mt.neff_fraction(2.0) == pytest.approx(0.0)  # hard limit for an exponential
    A = {"a": (1.0, 4.0), "b": (2.0, 8.0)}
    assert mt.choose_clock(A, "geometric") == pytest.approx((1 * 4 * 2 * 8) ** 0.25)
    assert mt.choose_clock(A, "decrease-only") == 8.0
    assert mt.choose_clock(A, "3.5") == 3.5
    c = mt.choose_clock(A, "balanced")
    worst = lambda cc: min(mt.neff_fraction(v / cc) for pair in A.values() for v in pair)
    assert worst(c) >= max(worst(cc) for cc in (1.0, 2.0, 4.0, 8.0)) - 1e-9


def test_model_rates_binding_and_first_order():
    s = dict(t_bind=np.full(10, 400.0), t_unbind=np.full(10, 1e6))
    kon, koff = mt.model_rates(s, 0.017, "binding")
    assert kon == pytest.approx(1 / (400.0 * 17000.0))   # per uM per model time unit
    assert koff == pytest.approx(1e-6)
    kf, ku = mt.model_rates(s, 0.017, "first-order")
    assert kf == pytest.approx(1 / 400.0)


# -------------------------------------------------------- does it correct observables?
def test_reweighting_recovers_a_structural_observable():
    """A descriptor correlated with the first-passage time: reweighting the model sample
    onto the (known) target must recover the target's average of that descriptor."""
    rng = np.random.default_rng(3)
    tau_true = 120.0
    t_model = rng.lognormal(3.9, 0.55, 600)             # model: wrong mean and shape
                                                        # (not the family fitted internally)
    descr = lambda x: 1.0 / (1.0 + 60.0 / x)            # e.g. "fraction of contacts formed"
    d_model = descr(t_model) + rng.normal(0, 0.02, t_model.size)
    t_truth = rng.exponential(tau_true, 20000)
    d_truth = descr(t_truth).mean()
    w = mt.target_weights(t_model, tau_true)
    assert np.sum(w * d_model) == pytest.approx(d_truth, abs=0.02)
    assert abs(d_model.mean() - d_truth) > 0.05          # the unweighted model is off


# ---------------------------------------------------------------------- end-to-end
@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for name, kon, koff, tb, tu in [("S1", 2.6, 22.0, 400.0, 9e5), ("S2", 1.2, 50.0, 530.0, 5e5)]:
        for direction, tau in (("bind", tb), ("unbind", tu)):
            x = rng.exponential(tau, 50)
            x *= tau / x.mean()
            np.savetxt(tmp_path / f"{name}_{direction}.dat", x)
        rows.append(dict(system=name, kon_exp=kon, koff_exp=koff,
                         bind_times=f"{name}_bind.dat", unbind_times=f"{name}_unbind.dat"))
    p = tmp_path / "systems.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return p, tmp_path


def test_end_to_end(dataset):
    p, tmp = dataset
    out = tmp / "out"
    mt.main([str(p), "--conc", "0.017", "--clock", "balanced", "--out", str(out)])
    with open(out / "summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    for r in rows:
        assert float(r["weighted_mean"]) == pytest.approx(float(r["tau_target"]), rel=0.05)
        assert 0 < float(r["N_eff"]) <= 50
    for f in ["calibration.png", "cdf_bind.png", "cdf_unbind.png",
              "weights_S1_bind.csv", "weights_S2_unbind.csv"]:
        assert (out / f).exists(), f
    with open(out / "weights_S1_bind.csv") as fh:
        w = [float(x["weight"]) for x in csv.DictReader(fh)]
    assert len(w) == 50 and sum(w) == pytest.approx(1.0, abs=1e-6)


def test_ratios_are_independent_of_the_clock(dataset):
    """Only relative rates are physical: the needed factors must scale with 1/c."""
    p, tmp = dataset
    got = []
    for clock in ("geometric", "decrease-only"):
        out = tmp / f"o_{clock}"
        mt.main([str(p), "--clock", clock, "--out", str(out)])
        with open(out / "summary.csv") as fh:
            got.append({(r["system"], r["direction"]): float(r["needed_factor"])
                        for r in csv.DictReader(fh)})
    ratios = [got[0][k] / got[1][k] for k in got[0]]
    assert np.allclose(ratios, ratios[0], rtol=1e-3)
