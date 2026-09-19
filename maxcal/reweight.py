#!/usr/bin/env python3
"""maxcal-reweight - MaxCal reweighting of whole first-passage trajectories.

Each trajectory keeps its own history and gets the weight w ∝ c^k of the attempt tilt
(README 2.3), so no Markov-at-interface assumption enters: this is the maximum-entropy
reweighting of the trajectory ensemble under a constraint on the mean number of failed
attempts.  Useful for trajectory-level and route-level questions, for data with memory
(where stitching is not valid), and as a small-lambda cross-check on maxcal-stitch.

It cannot create folding times longer than those observed: the weights saturate at
(1-p)^-k, N_eff falls as lambda grows, and on lag-dominated data the CV moves away from 1
(README 2.4).  Read the KS p-value together with CV and N_eff, and use maxcal-stitch for
the Poisson question.

Outputs per Q‡: lambda* where the weighted CV reaches 1 (if it does) with a bootstrap
interval, N_eff and the KS p-value there, and the per-trajectory weights.

Example:
  maxcal-reweight "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.45 --out reweight_out
"""
import argparse
import csv
import os

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt   # backend set in main()

from .core import (attempts, success_prob, log_weights, cv_curve, first_crossing,
                   ks_exp_weighted, weighted_km, lilliefors_p, geometric_test, geom_status,
                   lag1_status, reweight_status, add_common_args, load_trajectories,
                   qts_grid, write_csv)


def scan(trajs, qts, args, rng, lams):
    """Reweighting at one attempt interface."""
    A = attempts(trajs, qts, args.t0, args.include_initial, args.qtse)
    k, T, done = A["k"], A["T"], A["done"]
    p = success_prob(k, done)
    cv, neff = cv_curve(lams, k, T, done, p)
    lam_star = first_crossing(lams, cv)
    w0 = np.full(k.size, 1.0 / k.size)
    D0, _ = ks_exp_weighted(T, done, w0)
    out = dict(qts=qts, N=int(k.size), n_done=int(done.sum()), sum_k=int(k.sum()),
               kmax=int(k.max()) if k.size else 0, p=p, cv0=cv[0], cv_lammax=cv[-1],
               ks_p0=lilliefors_p(D0, k.size), lam_star=lam_star, note=A["note"])
    out["geom_p"], _ = geometric_test(k[done])
    out["geom_status"] = geom_status(int(done.sum()), out["geom_p"])
    pr = A["pairs"]
    out["lag1_status"] = lag1_status(pr)
    out["lag1_rho"], out["lag1_p"] = (stats.spearmanr(pr[:, 0], pr[:, 1])
                                      if out["lag1_status"] == "ok" else (np.nan, np.nan))

    if np.isfinite(lam_star):
        w = np.exp(log_weights(lam_star, k, done, p)[0])
        out["neff_star"] = 1.0 / np.sum(w ** 2)
        D, _ = ks_exp_weighted(T, done, w)
        out["ks_p_star"] = lilliefors_p(D, round(out["neff_star"]))
    else:
        w = w0
        out.update(neff_star=np.nan, ks_p_star=np.nan)
    out["reweight_ok"] = bool(np.isfinite(lam_star) and out["neff_star"] >= args.neff_min)

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
    out["reweight_status"] = reweight_status(out["cv0"], lam_star, out["neff_star"],
                                             args.neff_min)
    return out, A, w, cv, neff


def main(argv=None):
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--qtse", type=float, default=None,
                    help="surface for the TSE frames (last upward crossing before folding); "
                         "default: each Q‡")
    ap.add_argument("--lam-max", type=float, default=10.0)
    ap.add_argument("--nlam", type=int, default=201)
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for lambda*")
    ap.add_argument("--neff-min", type=float, default=30.0,
                    help="N_eff below which the reweighting is flagged")
    args = ap.parse_args(argv)

    trajs = load_trajectories(args.files, args.dt, args.qu, args.qf)
    print(f"{len(trajs)} trajectories, "
          f"{sum(t['fold'] is None for t in trajs)} censored (never reach Qf={args.qf})")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lams = np.linspace(0.0, args.lam_max, args.nlam)
    grid, qref = qts_grid(args)

    results = []
    for q in grid:
        row, A, w, cv, neff = scan(trajs, q, args, rng, lams)
        results.append(row)
        if np.isclose(q, qref):
            ref, ref_A, ref_w, ref_cv, ref_neff = row, A, w, cv, neff
    cols = ["qts", "N", "n_done", "sum_k", "kmax", "p", "geom_p", "lag1_rho", "lag1_p",
            "cv0", "cv_lammax", "ks_p0", "lam_star", "lam_lo", "lam_hi", "boot_noroot",
            "neff_star", "ks_p_star", "reweight_ok", "geom_status", "lag1_status",
            "reweight_status", "note"]
    write_csv(os.path.join(args.out, "summary.csv"), cols, results)

    print(f"\n{'Q‡':>6} {'p':>6} {'CV0':>5} {'CV(max)':>8} {'lam*':>6} {'95% CI':>13} "
          f"{'N_eff':>6} {'KSp*':>6}  status")
    for r in results:
        ci = f"[{r['lam_lo']:.2f},{r['lam_hi']:.2f}]" if np.isfinite(r["lam_star"]) else "-"
        print(f"{r['qts']:6.3f} {r['p']:6.3f} {r['cv0']:5.2f} {r['cv_lammax']:8.2f} "
              f"{r['lam_star']:6.2f} {ci:>13} {r['neff_star']:6.1f} {r['ks_p_star']:6.3f}"
              f"  {r['reweight_status']}" + ("  *ref" if np.isclose(r["qts"], qref) else ""))
    if ref["reweight_status"] == "no_root":
        print(f"\nreweighting: the weighted CV only reaches {ref['cv_lammax']:.2f}. The tilt can"
              f" only reshuffle the observed attempt counts (max k = {ref['kmax']}) and the"
              " weights saturate at (1-p)^-k, so the geometric tail of a higher barrier is"
              " not in the data. Use maxcal-stitch.")
    elif ref["reweight_status"] == "cv_ge_1_at_lambda0":
        print("\nNOTE: CV >= 1 already at lambda = 0; a missing barrier does not explain the"
              " deviation (look for intermediates or parallel routes).")
    elif ref["reweight_status"] == "low_neff":
        print(f"\nWARNING: N_eff = {ref['neff_star']:.1f} < {args.neff_min} at lambda*;"
              " the reweighted ensemble rests on few trajectories.")

    with open(os.path.join(args.out, f"weights_qts{qref:.3f}.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["trajectory", "k_failed", "time", "completed", "weight_at_lambda_star"])
        for j, i in enumerate(ref_A["idx"]):
            wr.writerow([trajs[i]["path"], ref_A["k"][j], f"{ref_A['T'][j]:.6g}",
                         int(ref_A["done"][j]), f"{ref_w[j]:.6g}"])
    pos = {i: j for j, i in enumerate(ref_A["idx"])}
    with open(os.path.join(args.out, f"tse_frames_qts{qref:.3f}.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["trajectory", "frame", "time", "k_failed", "weight_at_lambda_star"])
        for i, fr in ref_A["tse"]:
            j = pos[i]
            wr.writerow([trajs[i]["path"], fr, f"{fr * trajs[i]['dt']:.6g}",
                         ref_A["k"][j], f"{ref_w[j]:.6g}"])

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = np.linspace(0, 5, 200)
    ax.semilogy(x, np.exp(-x), "k--", lw=1, label="exponential")
    for wt, lab in [(np.full(ref_A["k"].size, 1.0 / ref_A["k"].size), "original")] + \
                   ([(ref_w, f"reweighted, lambda* = {ref['lam_star']:.2f} kT")]
                    if np.isfinite(ref["lam_star"]) else []):
        ts, S, _ = weighted_km(ref_A["T"], ref_A["done"], wt)
        tau = np.sum(wt * ref_A["T"]) / np.sum(wt * ref_A["done"])
        ax.step(np.r_[0, ts / tau], np.r_[1, S], where="post", label=lab)
    ax.set_xlim(0, 5); ax.set_ylim(1e-2, 1.05)
    ax.set_xlabel("t / <t>"); ax.set_ylabel("survival S(t)")
    ax.set_title(f"Weighted survival, Q‡ = {qref:.3f}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "survival.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
    axs[0].plot(lams, ref_cv); axs[0].axhline(1, color="k", ls="--", lw=1)
    axs[0].set_ylabel("weighted CV")
    axs[1].plot(lams, ref_neff, color="C2")
    axs[1].axhline(args.neff_min, color="r", ls=":", lw=1)
    axs[1].set_ylabel("N_eff"); axs[1].set_xlabel("lambda (kT)")
    for a in axs:
        if np.isfinite(ref["lam_star"]):
            a.axvline(ref["lam_star"], color="C1", lw=1)
    axs[0].set_title("lambda* where the weighted CV reaches 1"
                     if np.isfinite(ref["lam_star"]) else "no root: CV never reaches 1")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "lambda_scan.png"), dpi=150)
    plt.close(fig)
    print(f"\nWrote summary.csv, weights_*.csv, tse_frames_*.csv, survival.png, "
          f"lambda_scan.png to {args.out}/")


if __name__ == "__main__":
    main()
