#!/usr/bin/env python3
"""
maxcal_poisson.py - MaxCal barrier-tilt analysis of first-passage folding trajectories.

Model assumption: the simulations (e.g. multi-eGO) sample structures correctly but
underestimate the barrier between U and F.  The minimal (maximum-caliber) path
reweighting that raises the barrier at a dividing surface Q‡ multiplies the success
odds of every attempt by exp(-lambda).  lambda is an effective log-odds shift
(it approaches DDF‡/kT only in the high-barrier limit; see README).  For first-passage
paths (trajectories stopped at folding) this gives per-trajectory weights

    completed trajectory with k failed attempts : w ∝ c^(k+1) exp(-lambda)
    censored  trajectory with k failed attempts : w ∝ c^k
    c = 1 / (1 - p (1 - exp(-lambda)))

where p is the model success probability per attempt.  An "attempt" at Q‡ is an
excursion out of U (Q < Qu) that gets past Q‡ (place Q‡ at the FOOT of the
missing barrier, on the U side - see README); it fails if it returns below Qu
before reaching Qf.

For each Q‡ on a grid the script:
  1. counts failed attempts k_i per trajectory,
  2. tests attempt independence (k_i geometric? lag-1 correlation of cycle times),
  3. reweights: scans lambda, finds lambda* where the weighted CV of folding times = 1,
     reports N_eff and a Lilliefors-type KS p-value vs an exponential,
  4. stitches: builds synthetic first-passage times from resampled cycles with the
     tilted success probability, finds the smallest lambda giving Poisson statistics,
  5. bootstraps lambda* over trajectories,
and writes the transition-state frames (last upward crossing of --qtse, default Q‡)
with their weights.

Input: one file per trajectory, either one column Q (then give --dt) or two columns
(time, Q).  Lines starting with '#' or '@' (GROMACS .xvg) are ignored.
Trajectories are assumed to stop at folding (Q >= Qf); those that never reach Qf are
treated as right-censored.

Example:
  python maxcal_poisson.py "runs/q_*.xvg" --qu 0.3 --qf 0.8 --out results
"""
import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np
from scipy import stats
from scipy.special import logsumexp

import matplotlib.pyplot as plt   # backend set to Agg in main(), not at import


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


def stitch(pp, cycles, succ, M, rng, kexact=500):
    """Synthetic first-passage times: Geom(pp) failures of resampled cycles + one success."""
    k = rng.geometric(pp, size=M) - 1
    out = rng.choice(succ, size=M).astype(float)
    small = k <= kexact
    ks = k[small]
    if ks.sum() > 0:
        draws = rng.choice(cycles, size=int(ks.sum()))
        out[small] += np.bincount(np.repeat(np.arange(ks.size), ks),
                                  weights=draws, minlength=ks.size)
    kb = k[~small]
    if kb.size:  # CLT for very long sums
        out[~small] += rng.normal(kb * cycles.mean(), np.sqrt(kb) * cycles.std())
    return out


def stitch_scan(lams, p, A, n_sub_size, args, rng):
    """Smallest lambda at which stitched times are Poisson (median KS p >= alpha, |CV-1| <= tol)."""
    res = []
    lam_pass, streak = np.nan, 0
    if A["fail_pool"].size == 0 or A["succ_pool"].size == 0:
        return lam_pass, res
    for lam in lams:
        pp = tilted_p(p, lam)
        if (1 - pp) / pp > 1e7:
            break
        t = stitch(pp, A["fail_pool"], A["succ_pool"], args.nstitch, rng)
        cv = t.std() / t.mean()
        sub = rng.choice(t, size=(args.nsub, n_sub_size))
        null = null_D(n_sub_size)
        pv = (np.sum(null[None, :] >= D_rows(sub)[:, None], axis=1) + 1) / (null.size + 1)
        mp = float(np.median(pv))
        res.append((lam, cv, mp))
        ok = mp >= args.alpha and abs(cv - 1) <= args.cv_tol
        if ok:
            if streak == 0:
                first = lam
            streak += 1
            if streak >= 3:          # stable: first of 3 consecutive passes
                lam_pass = first
                break
        else:
            streak = 0
    return lam_pass, res


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


# --------------------------------------------------------------------------- main
def analyse(trajs, qts, args, rng, lams, full=False):
    A = attempts(trajs, qts, args.t0, args.include_initial, args.qtse)
    k, T, done = A["k"], A["T"], A["done"]
    p = success_prob(k, done)
    out = dict(qts=qts, N=k.size, n_done=int(done.sum()), sum_k=int(k.sum()),
               frac_k0=float(np.mean(k[done] == 0)) if done.any() else np.nan, p=p,
               note=A["note"])
    out["geom_p"], out["geom_bins"] = geometric_test(k[done])
    out["geom_status"] = geom_status(int(done.sum()), out["geom_p"])
    pr = A["pairs"]
    out["lag1_status"] = lag1_status(pr)
    out["lag1_rho"], out["lag1_p"] = (stats.spearmanr(pr[:, 0], pr[:, 1])
                                      if out["lag1_status"] == "ok" else (np.nan, np.nan))

    cv, neff = cv_curve(lams, k, T, done, p)
    lam_star = first_crossing(lams, cv)
    out.update(cv0=cv[0], lam_star=lam_star)
    w0 = np.full(k.size, 1.0 / k.size)
    D0, _ = ks_exp_weighted(T, done, w0)
    out["ks_p0"] = lilliefors_p(D0, k.size)
    if np.isfinite(lam_star):
        w = np.exp(log_weights(lam_star, k, done, p)[0])
        out["neff_star"] = 1.0 / np.sum(w ** 2)
        D, _ = ks_exp_weighted(T, done, w)
        out["ks_p_star"] = lilliefors_p(D, round(out["neff_star"]))
        out["reweight_ok"] = out["neff_star"] >= args.neff_min
    else:
        w = w0
        out.update(neff_star=np.nan, ks_p_star=np.nan, reweight_ok=False)

    # bootstrap lambda*
    boots = []
    for _ in range(args.boot):
        b = rng.integers(0, k.size, k.size)
        if not done[b].any():
            continue
        cvb, _ = cv_curve(lams, k[b], T[b], done[b], success_prob(k[b], done[b]))
        boots.append(first_crossing(lams, cvb))
    boots = np.array(boots)
    fin = boots[np.isfinite(boots)]
    out["lam_lo"], out["lam_hi"] = (np.percentile(fin, [2.5, 97.5]) if fin.size
                                    else (np.nan, np.nan))
    out["boot_noroot"] = float(np.mean(~np.isfinite(boots))) if boots.size else np.nan
    out["reweight_status"] = reweight_status(out["cv0"], lam_star, out["neff_star"], args.neff_min)

    lam_st, st_res = stitch_scan(lams, p, A, k.size, args, rng)
    out["lam_stitch"] = lam_st
    out["stitch_status"] = stitch_status(lam_st, A["fail_pool"].size, A["succ_pool"].size,
                                         A["note"])
    out["cv_stitch0"], out["ksp_stitch0"] = (st_res[0][1], st_res[0][2]) if st_res else (np.nan, np.nan)
    out["cv_lammax"] = cv[-1]
    out["kmax"] = int(k.max()) if k.size else 0
    out["p_stitch"] = tilted_p(p, lam_st) if np.isfinite(lam_st) else np.nan
    if full:
        out.update(A=A, lams=lams, cv=cv, neff=neff, w=w, st_res=st_res)
    return out


def main():
    plt.switch_backend("Agg")        # command-line use: files only, no display
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="trajectory files or glob patterns")
    ap.add_argument("--qu", type=float, required=True, help="U basin: Q < Qu")
    ap.add_argument("--qf", type=float, required=True, help="folded: Q >= Qf (trajectories stop here)")
    ap.add_argument("--dt", type=float, default=None, help="frame spacing (needed for 1-column files)")
    ap.add_argument("--qts", type=float, default=None, help="reference Q‡ for plots/TSE (default: grid middle)")
    ap.add_argument("--qtse", type=float, default=None,
                    help="surface for TSE frames (last upward crossing before folding); default: each Q‡. Put it at the barrier top when Q‡ is at the barrier foot")
    ap.add_argument("--nqts", type=int, default=9, help="number of Q‡ values scanned in (Qu, Qf)")
    ap.add_argument("--t0", type=float, default=0.0, help="discard initial relaxation up to this time")
    ap.add_argument("--lam-max", type=float, default=10.0, help="max lambda (effective barrier shift, kT) scanned")
    ap.add_argument("--nlam", type=int, default=201)
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for lambda*")
    ap.add_argument("--nstitch", type=int, default=5000, help="synthetic times per lambda")
    ap.add_argument("--nsub", type=int, default=200, help="KS subsamples (size = #trajectories)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--cv-tol", type=float, default=0.1)
    ap.add_argument("--neff-min", type=float, default=30.0)
    ap.add_argument("--include-initial", action="store_true",
                    help="use first segment of each trajectory in stitching pools")
    ap.add_argument("--out", default="maxcal_out")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if not args.qu < args.qf:
        sys.exit("need Qu < Qf")

    paths = sorted({p for pat in args.files for p in (glob.glob(pat) or [pat])})
    trajs = []
    for path in paths:
        q, dt = load_traj(path, args.dt)
        fold, fails = parse_traj(q, args.qu, args.qf)
        trajs.append(dict(path=path, q=q, dt=dt, n=q.size, fold=fold, fails=fails))
    n_cens = sum(t["fold"] is None for t in trajs)
    print(f"{len(trajs)} trajectories, {n_cens} censored (never reach Qf={args.qf})")
    if n_cens:
        print("  note: CV uses completed trajectories only; KM/KS handle censoring")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lams = np.linspace(0.0, args.lam_max, args.nlam)
    grid = np.linspace(args.qu, args.qf, args.nqts + 2)[1:-1]
    qref = args.qts if args.qts is not None else grid[len(grid) // 2]
    if not args.qu < qref < args.qf:
        sys.exit("--qts must lie between Qu and Qf")
    grid = np.unique(np.r_[grid, qref])

    results = [analyse(trajs, q, args, rng, lams, full=np.isclose(q, qref)) for q in grid]
    ref = next(r for r in results if np.isclose(r["qts"], qref))

    # ---- summary table
    cols = ["qts", "N", "n_done", "sum_k", "frac_k0", "p", "geom_p", "lag1_rho", "lag1_p",
            "cv0", "ks_p0", "lam_star", "lam_lo", "lam_hi", "boot_noroot", "neff_star",
            "ks_p_star", "reweight_ok", "cv_lammax", "kmax", "cv_stitch0", "ksp_stitch0",
            "lam_stitch", "p_stitch", "geom_status", "lag1_status", "reweight_status",
            "stitch_status", "note"]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(cols)
        for r in results:
            wr.writerow([r[c] if isinstance(r[c], str) else f"{r[c]:.6g}" for c in cols])

    print(f"\n{'Q‡':>6} {'p':>6} {'geom_p':>7} {'CV0':>5} {'KSp0':>6} {'λ*(kT)':>7} "
          f"{'95% CI':>13} {'Neff*':>6} {'KSp*':>6} {'λ_stitch':>8}")
    for r in results:
        ci = f"[{r['lam_lo']:.2f},{r['lam_hi']:.2f}]" if np.isfinite(r["lam_star"]) else "-"
        print(f"{r['qts']:6.3f} {r['p']:6.3f} {r['geom_p']:7.3f} {r['cv0']:5.2f} {r['ks_p0']:6.3f} "
              f"{r['lam_star']:7.2f} {ci:>13} {r['neff_star']:6.1f} {r['ks_p_star']:6.3f} "
              f"{r['lam_stitch']:8.2f}" + ("  *ref" if r is ref else ""))

    # ---- warnings for the reference surface
    print(f"\nReference Q‡ = {qref:.3f}  [geometric test: {ref['geom_status']}, "
          f"lag-1: {ref['lag1_status']}, reweighting: {ref['reweight_status']}, "
          f"stitching: {ref['stitch_status']}]")
    if ref["note"]:
        print("  " + ref["note"])
    if np.isfinite(ref["geom_p"]) and ref["geom_p"] < args.alpha:
        print("  WARNING: failed-attempt counts are not geometric -> attempts have memory;"
              " Poisson statistics after tilting would be imposed, not recovered.")
    elif not np.isfinite(ref["geom_p"]):
        print("  geometric test: too few distinct k values to test independence")
    if ref["cv0"] >= 1:
        print("  NOTE: CV >= 1 already at lambda=0; a missing barrier does not explain the"
              " deviation (look for intermediates / parallel channels).")
    if np.isfinite(ref["lam_star"]) and not ref["reweight_ok"]:
        print(f"  WARNING: N_eff={ref['neff_star']:.1f} < {args.neff_min} at lambda*;"
              " trust the stitched estimate instead.")
    if not np.isfinite(ref["lam_star"]):
        print(f"  reweighting: CV reaches only {ref['cv_lammax']:.2f} at lambda_max. Reweighting can only"
              f" reshuffle the observed attempt counts (max k = {ref['kmax']}); the weights saturate at"
              " (1-p)^-k, so the geometric tail of a higher barrier is not in the data."
              " Use the stitching result.")
    if np.isfinite(ref["cv_stitch0"]):
        print(f"  stitching at lambda=0 (initial segments removed): CV={ref['cv_stitch0']:.2f},"
              f" median KS p={ref['ksp_stitch0']:.3f}")
        if ref["ksp_stitch0"] >= args.alpha and abs(ref["cv_stitch0"] - 1) <= args.cv_tol:
            print("  -> already Poisson without a barrier correction: the original deviation"
                  " is mostly the initial relaxation, not the missing barrier.")

    # ---- TSE frames
    A, w = ref["A"], ref["w"]
    pos = {i: j for j, i in enumerate(A["idx"])}
    with open(os.path.join(args.out, f"tse_frames_qts{qref:.3f}_qtse{(args.qtse or qref):.3f}.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["trajectory", "frame", "time", "k_failed", "weight_at_lambda_star"])
        for i, fr in A["tse"]:
            j = pos[i]
            wr.writerow([trajs[i]["path"], fr, f"{fr * trajs[i]['dt']:.6g}",
                         A["k"][j], f"{w[j]:.6g}"])

    # ---- plots
    k, T, done = A["k"], A["T"], A["done"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = np.linspace(0, 5, 200)
    ax.semilogy(x, np.exp(-x), "k--", lw=1, label="exponential")
    for wt, lab in [(np.full(k.size, 1.0 / k.size), "original")] + \
                   ([(w, f"reweighted, λ*={ref['lam_star']:.2f} kT")]
                    if np.isfinite(ref["lam_star"]) else []):
        ts, S, _ = weighted_km(T, done, wt)
        tau = np.sum(wt * T) / np.sum(wt * done)
        ax.step(np.r_[0, ts / tau], np.r_[1, S], where="post", label=lab)
    if np.isfinite(ref["lam_stitch"]):
        t = stitch(ref["p_stitch"], A["fail_pool"], A["succ_pool"], 20000, rng)
        ts = np.sort(t) / t.mean()
        ax.step(ts, 1 - np.arange(1, ts.size + 1) / ts.size, where="post",
                label=f"stitched, λ={ref['lam_stitch']:.2f} kT")
    ax.set_xlim(0, 5); ax.set_ylim(1e-2, 1.05)
    ax.set_xlabel("t / ⟨t⟩"); ax.set_ylabel("survival S(t)")
    ax.set_title(f"Folding-time survival, Q‡ = {qref:.3f}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "survival.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(5.5, 7), sharex=True)
    axs[0].plot(lams, ref["cv"]); axs[0].axhline(1, color="k", ls="--", lw=1)
    axs[0].set_ylabel("weighted CV")
    axs[1].plot(lams, ref["neff"]); axs[1].axhline(args.neff_min, color="r", ls=":", lw=1)
    axs[1].set_ylabel("N_eff")
    if ref["st_res"]:
        s = np.array(ref["st_res"])
        axs[2].plot(s[:, 0], s[:, 2], "o-", ms=3, label="stitched (median KS p)")
    axs[2].axhline(args.alpha, color="r", ls=":", lw=1)
    axs[2].set_ylabel("KS p"); axs[2].set_xlabel("λ (kT)"); axs[2].legend(fontsize=8)
    for a in axs:
        if np.isfinite(ref["lam_star"]):
            a.axvline(ref["lam_star"], color="g", lw=1)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "lambda_scan.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    q = np.array([r["qts"] for r in results])
    ls_ = np.array([r["lam_star"] for r in results])
    lo = np.array([r["lam_lo"] for r in results]); hi = np.array([r["lam_hi"] for r in results])
    ax.errorbar(q, ls_, yerr=[ls_ - lo, hi - ls_], fmt="o-", capsize=3, label="reweighting λ*")
    ax.plot(q, [r["lam_stitch"] for r in results], "s--", label="stitching")
    ax.set_xlabel("Q‡"); ax.set_ylabel("effective barrier shift λ (kT)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "robustness_qts.png"), dpi=150); plt.close(fig)

    print(f"\nWrote summary.csv, tse_frames_*.csv, survival.png, "
          f"lambda_scan.png, robustness_qts.png to {args.out}/")


if __name__ == "__main__":
    main()
