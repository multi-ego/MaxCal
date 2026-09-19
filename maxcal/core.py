"""Shared machinery: trajectory parsing, attempt counting at an interface, the MaxCal
tilt, survival/KS statistics and the status codes used by all maxcal tools."""
import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np
from scipy import stats
from scipy.special import logsumexp

# ----------------------------------------------------------------------------- I/O
def load_traj(path, dt_arg):
    data = np.loadtxt(path, comments=("#", "@"))
    if data.ndim == 1:
        if dt_arg is None:
            sys.exit(f"{path}: single column of Q, please pass --dt")
        return data.astype(float), float(dt_arg)
    t, q = data[:, 0], data[:, 1]
    if dt_arg is not None:
        return q.astype(float), float(dt_arg)
    d = np.diff(t)
    dt = float(np.median(d)) if d.size else 1.0
    if d.size and np.max(np.abs(d - dt)) > 1e-6 * abs(dt):
        warnings.warn(f"{path}: non-uniform time spacing, using median dt={dt}")
    return q.astype(float), dt


def parse_traj(q, qu, qf):
    """Folding frame (first Q >= Qf, or None) and failed excursions [start, end, maxQ].
    An excursion is a run with Q >= Qu that starts from U and returns below Qu
    before Qf.  'end' is the first frame back in U."""
    hit = np.flatnonzero(q >= qf)
    fold = int(hit[0]) if hit.size else None
    qq = q[:fold] if fold is not None else q
    if qq.size == 0:
        return fold, np.empty((0, 3))
    below = qq < qu
    change = np.flatnonzero(np.diff(below.astype(np.int8))) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, qq.size]
    maxq = np.maximum.reduceat(qq, starts)
    # above-Qu runs that were entered from U (start > 0) and left back to U (end < size)
    m = (~below[starts]) & (starts > 0) & (ends < qq.size)
    return fold, np.column_stack([starts[m], ends[m], maxq[m]]).astype(float)


# ------------------------------------------------------------------ per-Q‡ attempts
def attempts(trajs, qts, t0, include_initial, qtse=None):
    qtse = qts if qtse is None else qtse
    rows = []
    fail_cyc, fail_init, succ, succ_init, pairs, tse = [], [], [], [], [], []
    for i, tr in enumerate(trajs):
        dt, fold, F = tr["dt"], tr["fold"], tr["fails"]
        f0 = int(np.ceil(t0 / dt)) if t0 > 0 else 0
        if fold is not None and fold < f0:
            continue  # folded during the discarded initial window
        sel = (F[:, 2] >= qts) & (F[:, 1] > f0) if len(F) else np.zeros(0, bool)
        ends = F[sel, 1].astype(int)
        end_frame = fold if fold is not None else tr["n"]
        bounds = np.r_[f0, ends]
        segs = np.diff(bounds) * dt
        if segs.size:
            fail_init.append(segs[0])
            cyc = segs[1:]
            fail_cyc.extend(cyc)
            pairs.extend(zip(cyc[:-1], cyc[1:]))
        done = fold is not None
        if done:
            last = (fold - bounds[-1]) * dt
            (succ if ends.size else succ_init).append(last)
            seg = tr["q"][bounds[-1]:fold]
            b = np.flatnonzero(seg < qtse)
            tse.append((i, int(bounds[-1] + (b[-1] + 1 if b.size else 0))))
        rows.append((i, ends.size, (end_frame - f0) * dt, done))
    idx = np.array([r[0] for r in rows], int)
    k = np.array([r[1] for r in rows], int)
    T = np.array([r[2] for r in rows], float)
    done = np.array([r[3] for r in rows], bool)

    fail_pool = np.array(fail_cyc)
    succ_pool = np.array(succ)
    note = []
    if include_initial or fail_pool.size < 10:
        if not include_initial and fail_pool.size < 10:
            note.append("failure pool <10: initial segments included")
        fail_pool = np.r_[fail_pool, fail_init]
    if include_initial or succ_pool.size < 10:
        if not include_initial and succ_pool.size < 10:
            note.append("success pool <10: initial segments included")
        succ_pool = np.r_[succ_pool, succ_init]
    return dict(idx=idx, k=k, T=T, done=done, fail_pool=fail_pool,
                succ_pool=succ_pool, pairs=np.array(pairs), tse=tse, note="; ".join(note))


# --------------------------------------------------------------------- statistics
def success_prob(k, done):
    return done.sum() / (done.sum() + k.sum())


def log_weights(lams, k, done, p):
    """Normalized log weights, shape (len(lams), N)."""
    lams = np.atleast_1d(lams)
    lc = -np.log1p(-p * (-np.expm1(-lams)))                 # log c
    lw = np.outer(lc, k) + np.outer(lc - lams, done.astype(float))
    return lw - logsumexp(lw, axis=1, keepdims=True)


def tilted_p(p, lam):
    return p * np.exp(-lam) / (1.0 - p * (-np.expm1(-lam)))


def cv_curve(lams, k, T, done, p):
    """Weighted CV of completed folding times and N_eff (all trajectories)."""
    lw = log_weights(lams, k, done, p)
    w = np.exp(lw)
    neff = 1.0 / np.sum(w ** 2, axis=1)
    wc = w[:, done]
    wc = wc / wc.sum(axis=1, keepdims=True)
    Tc = T[done]
    m = wc @ Tc
    var = wc @ (Tc ** 2) - m ** 2
    return np.sqrt(np.clip(var, 0, None)) / m, neff


def first_crossing(x, y, level=1.0):
    """Linear interpolation of first x where y reaches level from below."""
    if y[0] >= level:
        return x[0]
    j = np.flatnonzero(y >= level)
    if j.size == 0:
        return np.nan
    j = j[0]
    return x[j - 1] + (level - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])


def weighted_km(t, done, w):
    o = np.argsort(t, kind="stable")
    t, d, w = t[o], done[o], w[o]
    at_risk = np.cumsum(w[::-1])[::-1]
    S = np.cumprod(np.where(d, 1.0 - w / at_risk, 1.0))
    return t, S, d


def ks_exp_weighted(t, done, w):
    """Sup distance between weighted KM survival and exponential with MLE tau."""
    tau = np.sum(w * t) / np.sum(w * done)
    ts, S, d = weighted_km(t, done, w)
    E = np.exp(-ts / tau)
    Sprev = np.r_[1.0, S[:-1]]
    D = np.max(np.maximum(np.abs(S - E), np.abs(Sprev - E))[d])
    return D, tau


_NULL = {}


def null_D(n, nsim=4000):
    """Null distribution of the KS statistic for an exponential with fitted scale
    (Lilliefors); scale-free, so simulated once per n."""
    n = int(max(n, 3))
    if n not in _NULL:
        r = np.random.default_rng(12345)
        _NULL[n] = np.sort(D_rows(r.exponential(size=(nsim, n))))
    return _NULL[n]


def D_rows(x):
    """KS distance to fitted exponential, one sample per row (uncensored)."""
    x = np.sort(x, axis=1)
    n = x.shape[1]
    F = 1.0 - np.exp(-x / x.mean(axis=1, keepdims=True))
    i = np.arange(1, n + 1)
    return np.max(np.maximum(i / n - F, F - (i - 1) / n), axis=1)


def lilliefors_p(D, n):
    null = null_D(n)
    return (np.sum(null >= D) + 1) / (null.size + 1)


def geometric_test(k):
    """Chi-square test that failed-attempt counts are geometric (independent attempts)."""
    n = k.size
    if n < 10:
        return np.nan, 0
    p = n / (n + k.sum())
    obs, prob, tail, j = [], [], 1.0, 0
    while True:
        pj = p * (1 - p) ** j
        if n * pj < 5 or n * (tail - pj) < 5:
            break
        obs.append(np.sum(k == j)); prob.append(pj); tail -= pj; j += 1
    obs.append(np.sum(k >= j)); prob.append(tail)
    if len(obs) < 3:
        return np.nan, len(obs)
    exp = n * np.array(prob)
    chi2 = np.sum((np.array(obs) - exp) ** 2 / exp)
    dof = len(obs) - 2
    return stats.chi2.sf(chi2, dof), len(obs)



# ------------------------------------------------------------------ status flags
# Every NaN in summary.csv comes with a reason in one of these columns.
def geom_status(n, pval):
    if n < 10:
        return "too_few_trajectories"
    return "too_few_bins" if not np.isfinite(pval) else "ok"


def lag1_status(pairs):
    """pairs: (n, 2) array of consecutive failure-cycle durations within trajectories."""
    if len(pairs) < 10:
        return "too_few_pairs"
    if np.ptp(pairs[:, 0]) == 0 or np.ptp(pairs[:, 1]) == 0:
        return "constant_cycles"         # correlation undefined
    return "ok"


def reweight_status(cv0, lam_star, neff, neff_min):
    if cv0 >= 1:
        return "cv_ge_1_at_lambda0"      # already over-dispersed: a missing barrier can't explain it
    if not np.isfinite(lam_star):
        return "no_root"                 # CV never reaches 1: observed attempt counts too few
    if not neff >= neff_min:
        return "low_neff"
    return "ok"


def stitch_status(lam_stitch, n_fail_pool, n_succ_pool, pool_note):
    if n_fail_pool == 0 or n_succ_pool == 0:
        return "empty_pool"              # no failed attempts past this Q‡ (at/after the barrier?)
    if not np.isfinite(lam_stitch):
        return "no_pass"
    return "ok_pool_fallback" if pool_note else "ok"   # fallback: initial segments were used



# --------------------------------------------------------------------- shared CLI
def add_common_args(ap, qts_help="reference Q‡ (default: middle of the scan grid)"):
    ap.add_argument("files", nargs="+", help="trajectory files or glob patterns")
    ap.add_argument("--qu", type=float, required=True, help="start basin: Q < Qu")
    ap.add_argument("--qf", type=float, required=True, help="target: Q >= Qf (trajectories stop here)")
    ap.add_argument("--dt", type=float, default=None, help="frame spacing (needed for 1-column files)")
    ap.add_argument("--qts", type=float, default=None, help=qts_help)
    ap.add_argument("--nqts", type=int, default=9, help="number of Q‡ values scanned in (Qu, Qf)")
    ap.add_argument("--t0", type=float, default=0.0, help="discard an initial relaxation window")
    ap.add_argument("--include-initial", action="store_true",
                    help="use the first segment of each trajectory in the stitching pools")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--cv-tol", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output directory")
    return ap


def load_trajectories(patterns, dt_arg, qu, qf):
    paths = sorted({p for pat in patterns for p in (glob.glob(pat) or [pat])})
    trajs = []
    for path in paths:
        q, dt = load_traj(path, dt_arg)
        fold, fails = parse_traj(q, qu, qf)
        trajs.append(dict(path=path, q=q, dt=dt, n=q.size, fold=fold, fails=fails))
    if not trajs:
        sys.exit("no trajectories found")
    return trajs


def qts_grid(args):
    """Scan grid inside (Qu, Qf), always containing the reference Q‡."""
    grid = np.linspace(args.qu, args.qf, args.nqts + 2)[1:-1]
    qref = args.qts if args.qts is not None else grid[len(grid) // 2]
    if not args.qu < qref < args.qf:
        sys.exit("--qts must lie between Qu and Qf")
    return np.unique(np.r_[grid, qref]), qref


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(cols)
        for r in rows:
            wr.writerow([r[c] if isinstance(r[c], str) else f"{r[c]:.6g}" for c in cols])
