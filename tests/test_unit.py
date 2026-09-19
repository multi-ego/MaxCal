"""Unit tests for the shared maxcal machinery (run with: pytest -q tests/)."""
import numpy as np
import pytest

import maxcal as m

QU, QF = 0.3, 0.8


def traj(q, dt=1.0):
    q = np.asarray(q, float)
    fold, fails = m.parse_traj(q, QU, QF)
    return dict(path="x", q=q, dt=dt, n=q.size, fold=fold, fails=fails)


# ------------------------------------------------------------------- parse_traj
def test_parse_counts_failures_and_fold():
    #        U    exc1(max .5)  U    exc2(max .7)   U   success
    q = [0.1, 0.4, 0.5, 0.2, 0.35, 0.7, 0.6, 0.1, 0.5, 0.9, 0.2]
    fold, F = m.parse_traj(np.array(q), QU, QF)
    assert fold == 9
    np.testing.assert_array_equal(F[:, :2], [[1, 3], [4, 7]])   # [start, first frame back in U]
    np.testing.assert_allclose(F[:, 2], [0.5, 0.7])


def test_parse_ignores_initial_run_above_qu():
    # starts in the transition region, falls into U without it being an attempt
    fold, F = m.parse_traj(np.array([0.5, 0.6, 0.2, 0.1, 0.9]), QU, QF)
    assert fold == 4 and F.shape == (0, 3)


def test_parse_censored_and_trailing_excursion():
    # never folds; last excursion is still open -> not a failure
    fold, F = m.parse_traj(np.array([0.1, 0.5, 0.1, 0.6, 0.7]), QU, QF)
    assert fold is None
    np.testing.assert_array_equal(F[:, :2], [[1, 2]])


def test_parse_fold_at_first_frame():
    fold, F = m.parse_traj(np.array([0.95, 0.1]), QU, QF)
    assert fold == 0 and F.shape == (0, 3)


# --------------------------------------------------------------------- attempts
def test_attempts_threshold_times_segments_and_tse():
    q = [0.1, 0.4, 0.5, 0.2, 0.35, 0.7, 0.6, 0.1, 0.5, 0.9, 0.2]
    t = [traj(q, dt=2.0)]
    A = m.attempts(t, qts=0.45, t0=0.0, include_initial=True)
    assert A["k"][0] == 2 and A["done"][0] and A["T"][0] == pytest.approx(18.0)
    A = m.attempts(t, qts=0.6, t0=0.0, include_initial=True)
    assert A["k"][0] == 1                               # only the max-0.7 excursion counts
    # segments: [0 -> 7] ends in failure (initial), [7 -> 9] success
    np.testing.assert_allclose(A["fail_pool"], [14.0])
    np.testing.assert_allclose(A["succ_pool"], [4.0])
    # last upward crossing of Q‡=0.6 before folding: frame 9 (q goes 0.5 -> 0.9)
    assert A["tse"] == [(0, 9)]
    # separate TSE surface: last upward crossing of 0.4 before folding is frame 8 (0.1 -> 0.5)
    assert m.attempts(t, qts=0.6, t0=0.0, include_initial=True, qtse=0.4)["tse"] == [(0, 8)]


def test_attempts_t0_discards_early_events_and_early_folders():
    q = [0.1, 0.4, 0.5, 0.2, 0.35, 0.7, 0.6, 0.1, 0.5, 0.9, 0.2]
    A = m.attempts([traj(q)], qts=0.45, t0=4.0, include_initial=True)
    assert A["k"][0] == 1 and A["T"][0] == pytest.approx(5.0)
    A = m.attempts([traj([0.1, 0.9])], qts=0.45, t0=4.0, include_initial=True)
    assert A["k"].size == 0


# ------------------------------------------------------------- tilt / MaxCal
def test_tilted_p_limits_and_odds():
    p = 0.3
    assert m.tilted_p(p, 0.0) == pytest.approx(p)
    for lam in [0.5, 2.0, 5.0]:
        pp = m.tilted_p(p, lam)
        assert pp / (1 - pp) == pytest.approx(np.exp(-lam) * p / (1 - p))


def test_log_weights_normalized_uniform_at_zero():
    k = np.array([0, 1, 2, 5]); done = np.array([1, 1, 0, 1], bool)
    lw = m.log_weights(np.array([0.0, 1.0, 3.0]), k, done, 0.4)
    np.testing.assert_allclose(np.exp(lw).sum(axis=1), 1.0)
    np.testing.assert_allclose(np.exp(lw[0]), 0.25)


def test_log_weights_ratios():
    p, lam = 0.4, 1.3
    c = 1.0 / (1.0 - p * (1.0 - np.exp(-lam)))
    k = np.array([0, 1, 1]); done = np.array([True, True, False])
    w = np.exp(m.log_weights(lam, k, done, p)[0])
    assert w[1] / w[0] == pytest.approx(c)                    # one more failure
    assert w[1] / w[2] == pytest.approx(c * np.exp(-lam))     # completed vs censored


def test_log_weights_saturate():
    p = 0.3; k = np.array([0, 3]); done = np.array([True, True])
    w = np.exp(m.log_weights(60.0, k, done, p)[0])
    assert w[1] / w[0] == pytest.approx((1 - p) ** -3, rel=1e-8)


def test_maxcal_identity_geometric_to_geometric():
    """Tilting an exactly geometric(p) attempt distribution gives geometric(p')."""
    p, lam, K = 0.4, 1.5, 200
    kk = np.arange(K)
    pop = p * (1 - p) ** kk                              # population of each k
    lw = m.log_weights(lam, kk, np.ones(K, bool), p)[0]
    tilted = pop * np.exp(lw); tilted /= tilted.sum()
    pp = m.tilted_p(p, lam)
    np.testing.assert_allclose(tilted, pp * (1 - pp) ** kk, rtol=1e-6, atol=1e-14)


def test_success_prob():
    assert m.success_prob(np.array([0, 2, 1]), np.array([1, 1, 0], bool)) == pytest.approx(2 / 5)


# ---------------------------------------------------------------- cv / crossing
def test_cv_curve_at_zero_matches_plain_cv():
    rng = np.random.default_rng(0)
    T = rng.exponential(size=50); k = rng.integers(0, 4, 50); done = np.ones(50, bool)
    cv, neff = m.cv_curve(np.array([0.0]), k, T, done, 0.5)
    assert cv[0] == pytest.approx(T.std() / T.mean())
    assert neff[0] == pytest.approx(50)


def test_first_crossing():
    x = np.array([0.0, 1.0, 2.0]); 
    assert m.first_crossing(x, np.array([0.5, 0.9, 1.3])) == pytest.approx(1.25)
    assert m.first_crossing(x, np.array([1.2, 0.9, 1.3])) == 0.0
    assert np.isnan(m.first_crossing(x, np.array([0.1, 0.2, 0.3])))


# ------------------------------------------------------------ survival / KS
def test_weighted_km_uniform_no_censoring_is_empirical():
    t = np.array([3.0, 1.0, 2.0, 4.0]); d = np.ones(4, bool)
    ts, S, _ = m.weighted_km(t, d, np.full(4, 0.25))
    np.testing.assert_allclose(ts, [1, 2, 3, 4]); np.testing.assert_allclose(S, [0.75, 0.5, 0.25, 0.0])


def test_weighted_km_censoring_textbook():
    # events at 1, 3; censored at 2 -> S(1)=3/4, S(3)=3/4*(1-1/2)
    t = np.array([1.0, 2.0, 3.0, 4.0]); d = np.array([1, 0, 1, 0], bool)
    _, S, _ = m.weighted_km(t, d, np.ones(4))
    np.testing.assert_allclose(S, [0.75, 0.75, 0.375, 0.375])


def test_weighted_km_weight_equals_duplication():
    t = np.array([1.0, 2.0, 3.0]); d = np.array([1, 0, 1], bool)
    _, Sw, _ = m.weighted_km(t, d, np.array([2.0, 1.0, 1.0]))
    _, Sd, _ = m.weighted_km(np.r_[1.0, t], np.r_[True, d], np.ones(4))
    assert Sw[-1] == pytest.approx(Sd[-1])


def test_ks_exp_weighted_tau_mle_with_censoring():
    t = np.array([1.0, 2.0, 3.0]); d = np.array([1, 0, 1], bool)
    _, tau = m.ks_exp_weighted(t, d, np.ones(3))
    assert tau == pytest.approx(6.0 / 2.0)


def test_D_rows_matches_scipy_with_fitted_scale():
    from scipy import stats
    x = np.random.default_rng(1).exponential(2.0, size=40)
    D = m.D_rows(x[None, :])[0]
    assert D == pytest.approx(stats.kstest(x, "expon", args=(0, x.mean())).statistic)


def test_lilliefors_size_and_power():
    rng = np.random.default_rng(7)
    n = 50
    p_null = [m.lilliefors_p(m.D_rows(rng.exponential(size=(1, n)))[0], n) for _ in range(400)]
    assert 0.02 < np.mean(np.array(p_null) < 0.05) < 0.09          # correct size
    gam = rng.gamma(4.0, size=(1, n))                                   # CV = 0.5
    assert m.lilliefors_p(m.D_rows(gam)[0], n) < 0.05                  # power


# ----------------------------------------------------------- independence test
def test_geometric_test_accepts_geometric_rejects_memory():
    rng = np.random.default_rng(3)
    k = rng.geometric(0.35, size=300) - 1
    assert m.geometric_test(k)[0] > 0.01
    k_mem = np.r_[np.zeros(150, int), np.full(150, 4)]                 # bimodal: memory
    assert m.geometric_test(k_mem)[0] < 1e-3
    assert np.isnan(m.geometric_test(np.zeros(50, int))[0])            # untestable


# ------------------------------------------------------------------- stitching
def test_stitch_mean_and_limits():
    rng = np.random.default_rng(5)
    cyc = rng.exponential(1.0, 500); succ = rng.exponential(0.2, 200)
    pp = 0.05
    t = m.stitch(pp, cyc, succ, 40000, rng)
    assert t.mean() == pytest.approx(succ.mean() + (1 - pp) / pp * cyc.mean(), rel=0.03)
    assert t.std() / t.mean() == pytest.approx(1.0, abs=0.06)          # thinning -> Poisson
    t1 = m.stitch(1.0, cyc, succ, 1000, rng)
    assert np.all(np.isin(t1, succ))                                    # no failures


def test_stitch_clt_branch_mean():
    rng = np.random.default_rng(6)
    cyc = rng.exponential(1.0, 500); succ = np.array([0.0])
    t = m.stitch(1e-4, cyc, succ, 4000, rng, kexact=10)
    assert t.mean() == pytest.approx((1 - 1e-4) / 1e-4 * cyc.mean(), rel=0.05)


# ------------------------------------------------------------------------ I/O
def test_load_xvg_two_columns(tmp_path):
    f = tmp_path / "a.xvg"
    f.write_text("# c\n@ title \"x\"\n0.0 0.1\n0.5 0.2\n1.0 0.9\n")
    q, dt = m.load_traj(str(f), None)
    np.testing.assert_allclose(q, [0.1, 0.2, 0.9]); assert dt == pytest.approx(0.5)


def test_load_one_column_needs_dt(tmp_path):
    f = tmp_path / "b.dat"; f.write_text("0.1\n0.2\n")
    with pytest.raises(SystemExit):
        m.load_traj(str(f), None)
    assert m.load_traj(str(f), 2.0)[1] == 2.0


# ---------------------------------------------------------------- status flags
def test_status_flags():
    assert m.geom_status(5, 0.3) == "too_few_trajectories"
    assert m.geom_status(50, np.nan) == "too_few_bins"
    assert m.geom_status(50, 0.3) == "ok"
    assert m.lag1_status(np.zeros((3, 2))) == "too_few_pairs"
    assert m.lag1_status(np.ones((20, 2))) == "constant_cycles"
    assert m.lag1_status(np.random.default_rng(0).random((20, 2))) == "ok"
    assert m.reweight_status(1.2, 0.0, 100, 30) == "cv_ge_1_at_lambda0"
    assert m.reweight_status(0.8, np.nan, np.nan, 30) == "no_root"
    assert m.reweight_status(0.8, 1.0, 10, 30) == "low_neff"
    assert m.reweight_status(0.8, 1.0, 100, 30) == "ok"
    assert m.stitch_status(0.5, 0, 10, "") == "empty_pool"
    assert m.stitch_status(np.nan, 10, 10, "") == "no_pass"
    assert m.stitch_status(0.5, 10, 10, "success pool <10") == "ok_pool_fallback"
    assert m.stitch_status(0.5, 10, 10, "") == "ok"


# ------------------------------------------------------ committor / TS location
def test_isotonic_pav():
    y = np.array([0.1, 0.3, 0.2, 0.4]); w = np.array([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(m.isotonic(y, w), [0.1, 0.25, 0.25, 0.4])
    np.testing.assert_allclose(m.isotonic(np.array([0.5, 0.1]), np.array([3.0, 1.0])), [0.4, 0.4])


def test_corrected_committor_formula():
    qm = np.array([0.1, 0.4, 0.6, 0.9])
    np.testing.assert_allclose(m.corrected_committor(qm, 1.0, 0.5), qm)       # no tilt
    q = m.corrected_committor(qm, 0.2, 0.5)
    np.testing.assert_allclose(q[:2], 0.2 * qm[:2])                          # U side scaled
    np.testing.assert_allclose(1 - q[2:], 0.2 * (1 - qm[2:]))                # F side: return prob scaled


def test_ts_location_cases():
    c = np.linspace(0.3, 0.8, 11); qm = np.linspace(0.0, 1.0, 11)           # linear model committor
    for r in (1.0, 0.5, 0.1):                                                 # barrier on the model TS:
        qts, where = m.ts_location(c, qm, r, 0.5)                             # TS never moves
        assert where == "barrier" and qts == pytest.approx(0.55)
    qts, where = m.ts_location(c, qm, 0.95, 0.9)                             # weak tilt, far barrier
    assert where == "U_side" and qts == pytest.approx(0.3 + 0.5 * (0.5 / 0.95))
    qts, where = m.ts_location(c, qm, 0.95, 0.2)
    assert where == "F_side" and qts == pytest.approx(0.3 + 0.5 * (1 - 0.5 / 0.95))
    assert m.ts_location(c, qm, 0.1, 0.9) == (pytest.approx(0.3 + 0.5 * 0.9), "barrier")


def test_committor_frames_labels():
    q = [0.1, 0.4, 0.5, 0.2, 0.35, 0.7, 0.6, 0.1, 0.5, 0.9, 0.2]
    ti, fr, qv, oc = m.committor_frames([traj(q)], QU, QF)
    np.testing.assert_array_equal(fr, [1, 2, 4, 5, 6, 8])
    np.testing.assert_array_equal(oc, [0, 0, 0, 0, 0, 1])


def test_ks_heatmap_shapes_and_empty_pool():
    q = [0.1, 0.4, 0.5, 0.2, 0.35, 0.7, 0.6, 0.1, 0.5, 0.9, 0.2]
    trajs = [traj(q) for _ in range(12)]
    args = type("a", (), dict(t0=0.0, include_initial=True, nstitch=300, nsub=20))()
    ksp, cv = m.ks_heatmap(trajs, np.array([0.45, 0.75]), np.array([0.0, 1.0]), args,
                           np.random.default_rng(0))
    assert ksp.shape == (2, 2)
    assert np.all(np.isfinite(ksp[0])) and np.all((ksp[0] > 0) & (ksp[0] <= 1))
    assert np.all(np.isnan(ksp[1]))              # no failed attempt reaches 0.75: empty pool
