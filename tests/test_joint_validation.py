"""Validation of the joint mode against a known answer (slow: pytest --runslow).

Asymmetric double well (DeltaG ≈ -0.6 kT, B more stable), overdamped Langevin.
  model : low barrier                       (stand-in for multi-eGO)
  truth : same potential + 2 kT Gaussian bump on the barrier top
  hot   : model potential, backward runs at 1.5x the temperature

Checks:
  * kinetic DeltaG from forward/backward renewal means equals the equilibrium
    (committor-split) DeltaG of the potential, for model and truth;
  * detailed-balance coupling: tilting the forward direction to the true success
    probability and fixing the backward one by DeltaG predicts the true backward
    success probability and folding-time distribution;
  * transition paths are time-reversal symmetric at the same temperature, and the
    symmetry test detects a backward set run at a different temperature;
  * misuse: declaring 'same' for data at different temperatures is flagged.
"""
import numpy as np
import pytest

import maxcal_joint as mj
import maxcal_poisson as mp
import langevin as L
from langevin import run_many

pytestmark = pytest.mark.slow

QA, QB, QTF, QTB, QTSE = 0.3, 0.8, 0.4, 0.7, 0.55
POT = dict(B=1.0, tilt=-1.0, sigma=0.06)
KW = dict(dt=0.002, stride=10, **POT)
N = 200
ARGS = type("a", (), dict(t0=0.0, include_initial=False))()


def pair(dB, seed, Tr_bwd=1.0):
    f, dt = run_many(N, seed, dB=dB, q0=0.2, **KW)
    b, _ = run_many(N, seed + 50, dB=dB, q0=0.9, stop_below=0.25, Tr=Tr_bwd, **KW)
    F = mj.direction(mj.build(mj_raw(f, dt), QA, QB, 1), QA, QB, QTF, QTSE, ARGS)
    B = mj.direction(mj.build(mj_raw(b, dt), -QB, -QA, -1), -QB, -QA, -QTB, -QTSE, ARGS)
    return F, B, (f, b, dt)


def mj_raw(qs, dt):
    return [(str(i), q, dt, None) for i, q in enumerate(qs)]


@pytest.fixture(scope="module")
def model():
    return pair(0.0, 1)


@pytest.fixture(scope="module")
def truth():
    return pair(2.0, 101)


@pytest.fixture(scope="module")
def hot():
    return pair(0.0, 201, Tr_bwd=1.5)


def dG_kin(F, B):
    return -np.log(B["M0"] / F["M0"])


def test_kinetic_dG_equals_equilibrium(model, truth):
    assert dG_kin(*model[:2]) == pytest.approx(L.delta_G_eq(QA, QB, **POT), abs=0.3)
    assert dG_kin(*truth[:2]) == pytest.approx(L.delta_G_eq(QA, QB, dB=2.0, **POT), abs=0.3)


def test_detailed_balance_coupling_predicts_backward(model, truth):
    Fm, Bm, _ = model
    Ft, Bt, _ = truth
    R = np.exp(-L.delta_G_eq(QA, QB, dB=2.0, **POT))       # "experimental" DeltaG
    Mf = mj.renewal_mean(Ft["p"], Fm["muC"], Fm["muS"])    # forward tilted to the truth
    pb = mj.solve_p(R * Mf, Bm["muC"], Bm["muS"])           # backward from detailed balance
    assert pb == pytest.approx(Bt["p"], rel=0.25)
    ts = mp.stitch(pb, Bm["A"]["fail_pool"], Bm["A"]["succ_pool"], 20000,
                   np.random.default_rng(0))
    Tb = Bt["A"]["T"]
    assert ts.mean() / Tb.mean() == pytest.approx(1.0, abs=0.2)
    assert ts.std() / ts.mean() == pytest.approx(Tb.std() / Tb.mean(), abs=0.2)


def test_transition_paths_time_reversal_symmetric(model, truth):
    from scipy import stats
    for F, B, _ in (model, truth):
        assert stats.ks_2samp(F["tp_dur"], B["tp_dur"]).pvalue > 0.01


def test_different_temperature_is_detected(hot):
    from scipy import stats
    F, B, _ = hot
    assert stats.ks_2samp(F["tp_dur"], B["tp_dur"]).pvalue < 0.01


def test_same_declared_for_different_T_is_flagged(hot, tmp_path):
    f, b, dt = hot[2]
    L.write_set(str(tmp_path / "f"), f, dt)
    L.write_set(str(tmp_path / "b"), b, dt)
    a = mj.parse_args(["--fwd", str(tmp_path / "f" / "*.xvg"), "--bwd", str(tmp_path / "b" / "*.xvg"),
                       "--qa", str(QA), "--qb", str(QB), "--qts-fwd", str(QTF),
                       "--qts-bwd", str(QTB), "--qtse", str(QTSE),
                       "--temperature-relation", "same", "--nstitch", "1000", "--nsub", "50",
                       "--out", str(tmp_path / "o")])
    res = mj.analyse(a)[0]
    assert any("transition-path durations differ" in n for n in res["notes"])
