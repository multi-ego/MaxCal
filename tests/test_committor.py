"""Committor correction and TS location (README §2.10).

Fast: exact 1D committors of the Langevin test potential, where a Gaussian bump on
the barrier top is the "missing barrier".  Slow (--runslow): the same check with
committors estimated from simulated trajectories, as the script does."""
import numpy as np
import pytest

import maxcal as m
import langevin as L

QU, QF, QTS, M_BUMP, SIG = 0.3, 0.8, 0.40, 0.55, 0.06


def exact_committor(x, dB):
    F = L.F_total(x, B=0.8, dB=dB, sigma=SIG)
    e = np.exp(F - F.max())
    c = np.cumsum(e)
    return c / c[-1]


@pytest.mark.parametrize("dB", [2.0, 3.0])
def test_located_barrier_formula_exact_1d(dB):
    x = np.linspace(QU, QF, 200001)
    qm, qt = exact_committor(x, 0.0), exact_committor(x, dB)
    i = np.argmin(abs(x - QTS))
    r = qt[i] / qm[i]                                  # p'/p as measured at the attempt interface
    qstar = qm[np.argmin(abs(x - M_BUMP))]
    pred = m.corrected_committor(qm, r, qstar)
    outside = np.abs(x - M_BUMP) > 4 * SIG
    assert np.max(np.abs(pred - qt)[outside]) < 1e-3
    assert x[np.argmin(abs(pred - 0.5))] == pytest.approx(x[np.argmin(abs(qt - 0.5))], abs=2e-3)


def test_one_sided_logit_rule_misplaces_ts():
    """The naive rule logit q' = logit q - lambda (barrier implicitly at the folded
    boundary) moves the TS far from the true one: documents why q* is needed."""
    x = np.linspace(QU, QF, 200001)
    qm, qt = exact_committor(x, 0.0), exact_committor(x, 3.0)
    i = np.argmin(abs(x - QTS))
    lam = np.log(qm[i] / (1 - qm[i])) - np.log(qt[i] / (1 - qt[i]))
    naive_ts = x[np.argmin(abs(qm - 1 / (1 + np.exp(-lam))))]
    assert abs(naive_ts - M_BUMP) > 0.1


def trajs(Qs, dt):
    out = []
    for i, q in enumerate(Qs):
        f, fa = m.parse_traj(q, QU, QF)
        out.append(dict(path=str(i), q=q, dt=dt, n=q.size, fold=f, fails=fa))
    return out


@pytest.fixture(scope="module")
def sampled():
    KW = dict(dt=0.002, sigma=SIG, stride=10)
    Qm, dt = L.run_many(200, 1, dB=0.0, **KW)
    Qt, _ = L.run_many(200, 2, dB=3.0, **KW)
    return trajs(Qm, dt), trajs(Qt, dt)


def profile(tr):
    _, _, qv, oc = m.committor_frames(tr, QU, QF)
    c, n, raw, iso = m.committor_profile(qv, oc, QU, QF, 20)
    return c, iso


@pytest.mark.slow
def test_sampled_ts_prediction(sampled):
    model, truth = sampled
    c, iso_m = profile(model)
    ct, iso_t = profile(truth)
    Am, At = m.attempts(model, QTS, 0.0, False), m.attempts(truth, QTS, 0.0, False)
    r = m.success_prob(At["k"], At["done"]) / m.success_prob(Am["k"], Am["done"])
    qstar = float(np.interp(M_BUMP, c, iso_m))          # barrier on the bump, in model committor
    qts_pred, where = m.ts_location(c, iso_m, r, qstar)
    qts_true = m.q_at(ct, iso_t, 0.5)
    assert where == "barrier"
    assert qts_pred == pytest.approx(qts_true, abs=0.03)
    # a wrongly assumed location gives a wrong TS: the location is an input, not an output
    qts_wrong, _ = m.ts_location(c, iso_m, r, 0.85)
    assert abs(qts_wrong - qts_true) > 0.05
