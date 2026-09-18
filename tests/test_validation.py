"""Validation against a known answer (slow; run with: pytest --runslow tests/test_validation.py).

Ground truth: overdamped Langevin on the same double well, once with the low barrier
(the "multi-eGO-like" model) and once with an extra Gaussian bump of height dB on the
barrier top.  The bump leaves the basins unchanged, which is exactly the assumption
behind the MaxCal crossing tilt: correct sampling, underestimated barrier.

Checks:
  * the high-barrier ground truth is Poissonian;
  * stitching the LOW-barrier cycles with the HIGH-barrier success probability
    reproduces the high-barrier folding-time distribution (mean and shape),
    provided the attempt surface Q‡ sits at the foot of the missing barrier;
  * the blind Poisson threshold found from low-barrier data alone is a lower bound
    on the true tilt;
  * known limitation: with Q‡ on the barrier top the construction fails, because
    the missing barrier then also changes how often Q‡ is reached.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import maxcal_poisson as m
from langevin import run_many

QU, QF = 0.3, 0.8
Q_FOOT, Q_TOP = 0.40, 0.55
KW = dict(dt=0.002, sigma=0.06, stride=10)
N = 200

pytestmark = pytest.mark.slow


def as_trajs(Qs, dt):
    out = []
    for i, q in enumerate(Qs):
        fold, fails = m.parse_traj(q, QU, QF)
        out.append(dict(path=str(i), q=q, dt=dt, n=q.size, fold=fold, fails=fails))
    return out


def logit(p):
    return np.log(p / (1 - p))


@pytest.fixture(scope="module")
def low():
    Qs, dt = run_many(N, seed=1, dB=0.0, **KW)
    return as_trajs(Qs, dt)


@pytest.fixture(scope="module", params=[2.0, 3.0], ids=["dB2kT", "dB3kT"])
def high(request):
    Qs, dt = run_many(N, seed=2, dB=request.param, **KW)
    return request.param, as_trajs(Qs, dt)


def stitched_vs_truth(low, high, qts, seed=0):
    Al = m.attempts(low, qts, 0.0, False)
    Ah = m.attempts(high, qts, 0.0, False)
    pl, ph = m.success_prob(Al["k"], Al["done"]), m.success_prob(Ah["k"], Ah["done"])
    lam_true = logit(pl) - logit(ph)
    assert m.tilted_p(pl, lam_true) == pytest.approx(ph)
    ts = m.stitch(ph, Al["fail_pool"], Al["succ_pool"], 20000, np.random.default_rng(seed))
    Th = Ah["T"][Ah["done"]]
    return ts, Th, lam_true, Al, pl


def test_ground_truth_is_poisson(high):
    dB, th = high
    T = np.array([t["fold"] * t["dt"] for t in th if t["fold"] is not None])
    assert T.size == N
    assert T.std() / T.mean() == pytest.approx(1.0, abs=0.15)
    assert m.lilliefors_p(m.D_rows(T[None, :])[0], T.size) > 0.01


def test_stitching_reproduces_high_barrier(low, high):
    dB, th = high
    ts, Th, lam_true, _, _ = stitched_vs_truth(low, th, Q_FOOT)
    assert lam_true > 0, "the bump must lower the success odds per attempt"
    assert ts.mean() / Th.mean() == pytest.approx(1.0, abs=0.15)
    assert ts.std() / ts.mean() == pytest.approx(Th.std() / Th.mean(), abs=0.2)  # SE of CV ~0.07 at N=200
    assert stats.ks_2samp(ts, Th).pvalue > 0.01


def test_poisson_threshold_is_lower_bound(low, high):
    dB, th = high
    _, _, lam_true, Al, pl = stitched_vs_truth(low, th, Q_FOOT)
    args = SimpleNamespace(nstitch=5000, nsub=200, alpha=0.05, cv_tol=0.1)
    lam_blind, _ = m.stitch_scan(np.linspace(0, 10, 201), pl, Al, N, args,
                                 np.random.default_rng(0))
    assert np.isfinite(lam_blind)
    assert lam_blind <= lam_true + 0.2


def test_known_limitation_surface_on_barrier_top(low, high):
    dB, th = high
    ts, Th, _, _, _ = stitched_vs_truth(low, th, Q_TOP)
    assert ts.mean() / Th.mean() < 0.7
