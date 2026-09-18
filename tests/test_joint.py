"""Unit tests and fast end-to-end checks for maxcal_joint.py."""
import numpy as np
import pytest

import maxcal_joint as mj
import maxcal_poisson as mp
from langevin import run_many

QA, QB, QTSE = 0.3, 0.8, 0.55


def raw(qs, dt=1.0, feats=None):
    return [(f"t{i}", np.asarray(q, float), dt, None if feats is None else feats[i])
            for i, q in enumerate(qs)]


# ------------------------------------------------------------------ orientation
def test_orient():
    s, lv = mj.orient(0.3, 0.8, 0.4, None)
    assert s == 1 and lv == [0.3, 0.8, 0.4, None]
    s, lv = mj.orient(2.0, 0.6, 1.4)
    assert s == -1 and lv == [-2.0, -0.6, -1.4]


def test_backward_mirror_parsing():
    # starts in B (>=0.8), one failed attempt past the B-side interface 0.7 (dips to 0.6,
    # returns above 0.8), then reaches A (<=0.3)
    q = [0.9, 0.6, 0.85, 0.9, 0.5, 0.2]
    tr = mj.build(raw([q]), -QB, -QA, -1)
    assert tr[0]["fold"] == 5
    A = mp.attempts(tr, qts=-0.7, t0=0.0, include_initial=True)
    assert A["k"][0] == 1


# ------------------------------------------------------------- transition paths
def test_transition_path_and_time_reversal_symmetry():
    q = [0.1, 0.2, 0.52, 0.61, 0.50, 0.57, 0.9]
    F = mj.build(raw([q]), QA, QB, 1)
    dur, nc, fr = mj.transition_paths(F, QA, QB, QTSE)
    assert dur[0] == pytest.approx(5.0) and nc[0] == 3
    assert sorted(f for _, f, _ in fr) == [2, 4, 5]
    assert sum(w for *_, w in fr) == pytest.approx(1.0)
    # the time-reversed trajectory, analysed as a backward run, gives the same path
    Bk = mj.build(raw([q[::-1]]), -QB, -QA, -1)
    durb, ncb, frb = mj.transition_paths(Bk, -QB, -QA, -QTSE)
    assert durb[0] == pytest.approx(dur[0]) and ncb[0] == nc[0]
    assert sorted(len(q) - 1 - f for _, f, _ in frb) == [2, 4, 5]


def test_transition_path_needs_start_basin():
    F = mj.build(raw([[0.5, 0.6, 0.9]]), QA, QB, 1)
    dur, nc, fr = mj.transition_paths(F, QA, QB, QTSE)
    assert dur.size == 0 and fr == []


# ------------------------------------------------------------ kinetics helpers
def test_renewal_mean_and_inverse():
    muC, muS = 2.0, 0.5
    for p in [0.9, 0.3, 0.01]:
        assert mj.solve_p(mj.renewal_mean(p, muC, muS), muC, muS) == pytest.approx(p)
    assert np.isnan(mj.solve_p(0.4, muC, muS))


def test_binding_standard_state_conversion():
    V = 125.0
    assert mj.dG_box_to_standard(0.0, V) == pytest.approx(-np.log(V * mj.C_STANDARD))
    assert mj.dG_standard_to_box(mj.dG_box_to_standard(-3.2, V), V) == pytest.approx(-3.2)
    # a box of exactly 1/C° (1.66 nm^3) is the standard state
    assert mj.dG_box_to_standard(-2.0, 1 / mj.C_STANDARD) == pytest.approx(-2.0)


def test_joint_scan_enforces_detailed_balance():
    rng = np.random.default_rng(0)
    A = dict(fail_pool=rng.exponential(1.0, 300), succ_pool=rng.exponential(0.2, 100))
    F = dict(p=0.3, muC=A["fail_pool"].mean(), muS=A["succ_pool"].mean(), A=A, N=100)
    Bd = dict(p=0.2, muC=A["fail_pool"].mean(), muS=A["succ_pool"].mean(), A=A, N=100)
    args = type("a", (), dict(nstitch=2000, nsub=50, alpha=0.05, cv_tol=0.1))()
    R = 1.7
    _, rows = mj.joint_scan(F, Bd, R, np.linspace(0, 3, 13), args, rng)
    ok = [r for r in rows if np.isfinite(r["p_b"])]
    assert ok
    for r in ok:
        Mf = mj.renewal_mean(r["p_f"], F["muC"], F["muS"])
        Mb = mj.renewal_mean(r["p_b"], Bd["muC"], Bd["muS"])
        assert Mb / Mf == pytest.approx(R)                    # k_f/k_b fixed along the scan
    assert all(np.diff([r["p_f"] for r in ok]) < 0)             # barrier raised in both directions
    assert all(np.diff([r["p_b"] for r in ok]) < 0)


def test_tse_features_path_weighting():
    q = [0.1, 0.2, 0.52, 0.61, 0.9]           # crossings: frame 2 (nearer), then fold
    feats = [np.array([[0.], [0.], [1.], [3.], [5.]])]
    F = mj.build(raw([q], feats=feats), QA, QB, 1)
    D = dict(tse=mj.transition_paths(F, QA, QB, QTSE)[2], trajs=F)
    mean, se, n = mj.tse_features(D, 20, np.random.default_rng(0))
    assert n == 1 and mean[0] == pytest.approx(1.0)


def test_cli_requires_temperature_relation():
    with pytest.raises(SystemExit):
        mj.parse_args(["--fwd", "a", "--bwd", "b", "--qa", "0.3", "--qb", "0.8",
                       "--qts-fwd", "0.4", "--qts-bwd", "0.7"])


# --------------------------------------------------------- fast end-to-end runs
POT = dict(B=1.0, tilt=-1.0, sigma=0.06, dt=0.002, stride=10)


@pytest.fixture(scope="module")
def small_sets(tmp_path_factory):
    import langevin as L
    d = tmp_path_factory.mktemp("joint")
    f, dt, ff = run_many(60, 1, q0=0.2, n_feat=2, **POT)
    b, _, fb = run_many(60, 2, q0=0.9, stop_below=0.25, n_feat=2, **POT)
    L.write_set(str(d / "fwd"), f, dt, ff)
    L.write_set(str(d / "bwd"), b, dt, fb)
    # same data expressed as a "distance": d = 1.5 - Q (A = high d)
    L.write_set(str(d / "fwd_d"), [1.5 - x for x in f], dt, ff)
    L.write_set(str(d / "bwd_d"), [1.5 - x for x in b], dt, fb)
    return d


def run(d, fwd, bwd, rel, extra=(), levels=("0.3", "0.8", "0.4", "0.7", "0.55")):
    qa, qb, qf, qbw, qt = levels
    a = mj.parse_args(["--fwd", str(d / fwd / "*.xvg"), "--bwd", str(d / bwd / "*.xvg"),
                       "--qa", qa, "--qb", qb, "--qts-fwd", qf, "--qts-bwd", qbw,
                       "--qtse", qt, "--temperature-relation", rel, "--nstitch", "1000",
                       "--nsub", "50", "--boot", "20", "--out", str(d / f"out_{rel}"),
                       *extra])
    res, F, B, ft, pp, rng = mj.analyse(a)
    mj.write_outputs(a, res, F, B, ft, pp, rng)
    return res


def test_same_temperature_run(small_sets):
    res = run(small_sets, "fwd", "bwd", "same")
    assert np.isfinite(res["dG_kin_box_renewal"])
    assert "lambda_joint_f" in res and "tse_features" in res
    for f in ["joint_summary.json", "tse_forward.csv", "tse_backward.csv",
              "survival_joint.png", "tp_durations.png", "tse_features.csv"]:
        assert (small_sets / "out_same" / f).exists(), f


def test_different_temperature_run_decouples(small_sets):
    res = run(small_sets, "fwd", "bwd", "different", extra=("--dG", "-0.6"))
    assert res["dG_kin_box_renewal"] is None
    assert "lambda_fwd_independent" in res and "lambda_bwd_independent" in res
    assert any("--dG ignored" in n for n in res["notes"])


def test_orientation_invariance(small_sets):
    """Analysing Q or the reversed CV d = 1.5 - Q must give identical kinetics."""
    rq = run(small_sets, "fwd", "bwd", "same")
    rd = run(small_sets, "fwd_d", "bwd_d", "same",
             levels=("1.2", "0.7", "1.1", "0.8", "0.95"))
    assert rd["orientation"].startswith("A high")
    for key in ["p", "sum_k", "M0", "mean_tp_duration"]:
        assert rd["fwd"][key] == pytest.approx(rq["fwd"][key])
        assert rd["bwd"][key] == pytest.approx(rq["bwd"][key])
    assert rd["dG_kin_box_renewal"] == pytest.approx(rq["dG_kin_box_renewal"])


def test_binding_requires_volume_with_dG(small_sets):
    with pytest.raises(SystemExit):
        run(small_sets, "fwd", "bwd", "same", extra=("--system", "binding", "--dG", "-5"))
