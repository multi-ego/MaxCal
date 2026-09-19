#!/usr/bin/env python3
"""maxcal-stitch - recover two-state (Poisson) kinetics by thinning barrier crossings.

Each folding/binding event is rebuilt as a geometric(p') number of resampled failure
cycles plus one success segment (README 2.5).  Reports, for every attempt interface Q‡ on
a scan grid: the attempt statistics and their independence tests, the smallest lambda whose
stitched first-passage times pass the Poisson criterion (lambda_min, a LOWER bound on the
missing barrier), and the (Q‡, lambda) decision map.

Example:
  maxcal-stitch "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.4 --out stitch_out
"""
import argparse
import csv
import os

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt   # backend set in main()

from .core import (attempts, success_prob, tilted_p, null_D, D_rows, lilliefors_p,
                   ks_exp_weighted, weighted_km, geometric_test, geom_status, lag1_status,
                   stitch_status, add_common_args, load_trajectories, qts_grid, write_csv)


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



def ks_heatmap(trajs, qts_grid, lams, args, rng):
    """Median stitched KS p-value and CV on a (Q‡, lambda) grid.  Each cell stitches
    args.nstitch folding times with the tilted success probability at that Q‡ and
    tests subsamples of the original size, exactly as stitch_scan does."""
    ksp = np.full((qts_grid.size, lams.size), np.nan)
    cv = np.full_like(ksp, np.nan)
    for i, q in enumerate(qts_grid):
        A = attempts(trajs, q, args.t0, args.include_initial)
        if A["fail_pool"].size == 0 or A["succ_pool"].size == 0:
            continue                                          # empty_pool: NaN column
        p = success_prob(A["k"], A["done"])
        n = A["k"].size
        null = null_D(n)
        for j, lam in enumerate(lams):
            pp = tilted_p(p, lam)
            if (1 - pp) / pp > 1e7:
                break
            t = stitch(pp, A["fail_pool"], A["succ_pool"], args.nstitch, rng)
            sub = rng.choice(t, size=(args.nsub, n))
            pv = (np.sum(null[None, :] >= D_rows(sub)[:, None], axis=1) + 1) / (null.size + 1)
            ksp[i, j], cv[i, j] = np.median(pv), t.std() / t.mean()
    return ksp, cv


def write_heatmap(qts_grid, lams, ksp, cv, results, trajs, args):
    from .committor import committor_frames, committor_profile, q_at
    _, _, qv, oc = committor_frames(trajs, args.qu, args.qf)
    centers, _, _, iso = committor_profile(qv, oc, args.qu, args.qf, 20)
    com = dict(Q_barrier=q_at(centers, iso, args.barrier_q))
    passed = (ksp >= args.alpha) & (np.abs(cv - 1) <= args.cv_tol)
    with open(os.path.join(args.out, "heatmap.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["qts", "lambda", "median_ks_p", "cv", "poisson_pass", "qts_valid"])
        for i, q in enumerate(qts_grid):
            valid = bool(q < com["Q_barrier"]) if np.isfinite(com["Q_barrier"]) else True
            for j, lam in enumerate(lams):
                wr.writerow([f"{q:.4g}", f"{lam:.4g}", f"{ksp[i, j]:.4g}", f"{cv[i, j]:.4g}",
                             int(passed[i, j]), int(valid)])

    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    dq = np.diff(qts_grid).min() if qts_grid.size > 1 else 0.05
    dl = np.diff(lams).min() if lams.size > 1 else 0.5
    qe = np.r_[qts_grid - dq / 2, qts_grid[-1] + dq / 2]
    le = np.r_[lams - dl / 2, lams[-1] + dl / 2]
    im = ax.pcolormesh(qe, le, np.clip(ksp.T, 1e-4, 1), norm=LogNorm(1e-4, 1), cmap="viridis",
                       shading="flat")
    fig.colorbar(im, ax=ax, label="median KS p (stitched)")
    from matplotlib.patches import Rectangle
    for i, j in zip(*np.nonzero(passed)):                     # hatch exactly the passing cells
        ax.add_patch(Rectangle((qe[i], le[j]), qe[i + 1] - qe[i], le[j + 1] - le[j], fill=False,
                               hatch="//", edgecolor="white", lw=0, alpha=0.7))
    lm = [r["lam_stitch"] for r in results]
    ax.plot([r["qts"] for r in results], lm, "w^-", ms=6, mec="k", label="λ_min(Q‡)")
    if np.isfinite(com["Q_barrier"]):
        ax.axvspan(com["Q_barrier"], qe[-1], color="0.2", alpha=0.45, lw=0,
                   label=f"Q‡ beyond assumed barrier (q*={args.barrier_q:.2f}): invalid")
        ax.axvline(com["Q_barrier"], color="k", lw=1)
    ax.set_xlim(qe[0], qe[-1]); ax.set_ylim(le[0], le[-1])
    ax.set_xlabel("attempt interface Q‡"); ax.set_ylabel("λ (kT)")
    ax.set_title("Poisson decision map: hatched = passes (KS p ≥ α and |CV−1| ≤ tol)\n"
                 "the TS lies between Q‡ and Qf, so only Q‡ before the barrier is meaningful",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "heatmap.png"), dpi=150); plt.close(fig)
    return passed



def scan(trajs, qts, args, rng, lams):
    """Attempt statistics and lambda_min at one interface."""
    A = attempts(trajs, qts, args.t0, args.include_initial)
    k, T, done = A["k"], A["T"], A["done"]
    p = success_prob(k, done)
    Tc = T[done]
    D0, _ = ks_exp_weighted(T, done, np.full(k.size, 1.0 / k.size))
    out = dict(qts=qts, N=int(k.size), n_done=int(done.sum()), sum_k=int(k.sum()),
               kmax=int(k.max()) if k.size else 0, p=p,
               frac_k0=float(np.mean(k[done] == 0)) if done.any() else np.nan,
               cv0=float(Tc.std() / Tc.mean()) if Tc.size else np.nan,
               ks_p0=lilliefors_p(D0, k.size), note=A["note"])
    out["geom_p"], _ = geometric_test(k[done])
    out["geom_status"] = geom_status(int(done.sum()), out["geom_p"])
    pr = A["pairs"]
    out["lag1_status"] = lag1_status(pr)
    out["lag1_rho"], out["lag1_p"] = (stats.spearmanr(pr[:, 0], pr[:, 1])
                                      if out["lag1_status"] == "ok" else (np.nan, np.nan))
    lam, rows = stitch_scan(lams, p, A, k.size, args, rng)
    out["lam_stitch"] = lam
    out["p_stitch"] = tilted_p(p, lam) if np.isfinite(lam) else np.nan
    out["cv_stitch0"], out["ksp_stitch0"] = (rows[0][1], rows[0][2]) if rows else (np.nan, np.nan)
    out["stitch_status"] = stitch_status(lam, A["fail_pool"].size, A["succ_pool"].size, A["note"])
    return out, A


def survival_plot(A, lam, path, qref):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = np.linspace(0, 5, 200)
    ax.semilogy(x, np.exp(-x), "k--", lw=1, label="exponential")
    w = np.full(A["k"].size, 1.0 / A["k"].size)
    ts, S, _ = weighted_km(A["T"], A["done"], w)
    tau = np.sum(w * A["T"]) / np.sum(w * A["done"])
    ax.step(np.r_[0, ts / tau], np.r_[1, S], where="post", label="original")
    if np.isfinite(lam) and A["fail_pool"].size and A["succ_pool"].size:
        p = success_prob(A["k"], A["done"])
        t = stitch(tilted_p(p, lam), A["fail_pool"], A["succ_pool"], 20000,
                   np.random.default_rng(0))
        tt = np.sort(t) / t.mean()
        ax.step(tt, 1 - np.arange(1, tt.size + 1) / tt.size, where="post",
                label=f"stitched, lambda = {lam:.2f} kT")
    ax.set_xlim(0, 5); ax.set_ylim(1e-2, 1.05)
    ax.set_xlabel("t / <t>"); ax.set_ylabel("survival S(t)")
    ax.set_title(f"Folding-time survival, Q‡ = {qref:.3f}")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--lam-max", type=float, default=10.0, help="largest lambda (kT) scanned")
    ap.add_argument("--nlam", type=int, default=201)
    ap.add_argument("--nstitch", type=int, default=5000, help="synthetic times per lambda")
    ap.add_argument("--nsub", type=int, default=200, help="KS subsamples (size = #trajectories)")
    ap.add_argument("--heatmap-nlam", type=int, default=21,
                    help="lambda values in the (Q‡, lambda) decision map (0 disables)")
    ap.add_argument("--heatmap-lam-max", type=float, default=5.0)
    ap.add_argument("--barrier-q", type=float, default=0.5,
                    help="assumed barrier location (model committor value) used to mark the "
                         "valid Q‡ range in the decision map")
    args = ap.parse_args(argv)

    trajs = load_trajectories(args.files, args.dt, args.qu, args.qf)
    n_cens = sum(t["fold"] is None for t in trajs)
    print(f"{len(trajs)} trajectories, {n_cens} censored (never reach Qf={args.qf})")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lams = np.linspace(0.0, args.lam_max, args.nlam)
    grid, qref = qts_grid(args)

    results, ref_A = [], None
    for q in grid:
        row, A = scan(trajs, q, args, rng, lams)
        results.append(row)
        if np.isclose(q, qref):
            ref_A, ref = A, row
    cols = ["qts", "N", "n_done", "sum_k", "kmax", "frac_k0", "p", "geom_p", "lag1_rho",
            "lag1_p", "cv0", "ks_p0", "cv_stitch0", "ksp_stitch0", "lam_stitch", "p_stitch",
            "geom_status", "lag1_status", "stitch_status", "note"]
    write_csv(os.path.join(args.out, "summary.csv"), cols, results)

    print(f"\n{'Q‡':>6} {'p':>6} {'geom_p':>7} {'CV0':>5} {'KSp0':>6} {'lam_min':>8} {'status':>12}")
    for r in results:
        print(f"{r['qts']:6.3f} {r['p']:6.3f} {r['geom_p']:7.3f} {r['cv0']:5.2f} "
              f"{r['ks_p0']:6.3f} {r['lam_stitch']:8.2f} {r['stitch_status']:>12}"
              + ("  *ref" if np.isclose(r['qts'], qref) else ""))
    if np.isfinite(ref["geom_p"]) and ref["geom_p"] < args.alpha:
        print("\nWARNING: failed-attempt counts are not geometric -> attempts have memory;"
              " Poisson statistics after tilting would be imposed, not recovered.")
    if np.isfinite(ref["ksp_stitch0"]) and ref["ksp_stitch0"] >= args.alpha \
            and abs(ref["cv_stitch0"] - 1) <= args.cv_tol:
        print("\nNOTE: already Poisson at lambda = 0 (initial segments removed): the original"
              " deviation is mostly the initial relaxation, not a missing barrier.")

    survival_plot(ref_A, ref["lam_stitch"], os.path.join(args.out, "survival.png"), qref)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot([r["qts"] for r in results], [r["lam_stitch"] for r in results], "s--")
    ax.set_xlabel("Q‡"); ax.set_ylabel("lambda_min (kT)")
    ax.set_title("robustness of lambda_min to the attempt interface")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "robustness_qts.png"), dpi=150)
    plt.close(fig)

    if args.heatmap_nlam > 0:
        lams_h = np.linspace(0.0, args.heatmap_lam_max, args.heatmap_nlam)
        ksp, cvh = ks_heatmap(trajs, grid, lams_h, args, rng)
        passed = write_heatmap(grid, lams_h, ksp, cvh, results, trajs, args)
        print(f"\n(Q‡, lambda) decision map: {int(passed.sum())}/{passed.size} cells pass"
              " -> heatmap.png")
    print(f"\nWrote summary.csv, survival.png, robustness_qts.png"
          + (", heatmap.csv/.png" if args.heatmap_nlam > 0 else "") + f" to {args.out}/")


if __name__ == "__main__":
    main()
