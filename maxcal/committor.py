#!/usr/bin/env python3
"""maxcal-committor - model committor along Q, corrected committor for a located missing
barrier, transition-state location, and TSE frames (README 2.10).

The tilt fixes how much barrier is missing (r = p'/p from lambda) but not where it sits.
The location is an input, --barrier-q, given as a value q* of the MODEL committor; the
default 0.5 puts the missing barrier on the model's own transition state, where the TS does
not move with lambda.

Example:
  maxcal-committor "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.4 --lams 0.3 1.3 2.3 --out com_out
"""
import argparse
import csv
import os

import numpy as np

import matplotlib.pyplot as plt   # backend set in main()

from .core import (attempts, success_prob, tilted_p, add_common_args, load_trajectories,
                   qts_grid)


def committor_frames(trajs, qu, qf):
    """Frames inside excursions out of U, labelled by the outcome of their excursion
    (1 = reaches Qf before returning below Qu).  For a frame x this outcome is a
    sample of the committor q(x).  Returns arrays (traj index, frame, Q, outcome)."""
    ti, fr, qq, oo = [], [], [], []
    for i, tr in enumerate(trajs):
        q, F = tr["q"], tr["fails"]
        for s, e, _ in F:
            s, e = int(s), int(e)
            idx = np.arange(s, e)
            ti.append(np.full(idx.size, i)); fr.append(idx); qq.append(q[idx]); oo.append(np.zeros(idx.size))
        f = tr["fold"]
        if f is not None:
            inU = np.flatnonzero(q[:f] < qu)
            s = inU[-1] + 1 if inU.size else 0
            idx = np.arange(s, f)
            ti.append(np.full(idx.size, i)); fr.append(idx); qq.append(q[idx]); oo.append(np.ones(idx.size))
    cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros(0, dt)
    return cat(ti, int), cat(fr, int), cat(qq, float), cat(oo, float)


def isotonic(y, w):
    """Weighted non-decreasing fit (pool-adjacent-violators)."""
    blocks = []                                  # [value, weight, length]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop(); v1, w1, n1 = blocks.pop()
            wt = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / wt if wt > 0 else 0.5 * (v1 + v2), wt, n1 + n2])
    return np.concatenate([np.full(n, v) for v, _, n in blocks])


def committor_profile(qvals, outcome, qu, qf, nbins):
    edges = np.linspace(qu, qf, nbins + 1)
    b = np.clip(np.digitize(qvals, edges) - 1, 0, nbins - 1)
    n = np.bincount(b, minlength=nbins).astype(float)
    s = np.bincount(b, weights=outcome, minlength=nbins)
    raw = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    ok = n > 0
    iso = np.full(nbins, np.nan)
    iso[ok] = isotonic(raw[ok], n[ok])
    return 0.5 * (edges[1:] + edges[:-1]), n, raw, iso


def corrected_committor(qm, r, qstar):
    qm = np.asarray(qm, float)
    return np.where(qm < qstar, r * qm, 1.0 - r * (1.0 - qm))


def q_at(centers, q_iso, level):
    """Q where the (monotone) model committor reaches `level` (linear interpolation)."""
    ok = np.isfinite(q_iso)
    c, q = centers[ok], q_iso[ok]
    if q.size == 0 or level < q[0] or level > q[-1]:
        return np.nan
    j = np.flatnonzero(q >= level)[0]
    if j == 0 or q[j] == q[j - 1]:
        return c[j]
    return c[j - 1] + (level - q[j - 1]) * (c[j] - c[j - 1]) / (q[j] - q[j - 1])


def ts_location(centers, q_iso, r, qstar):
    """Location of the corrected transition state q' = 1/2, and which part of the
    corrected committor puts it there ('barrier', 'U_side', 'F_side')."""
    if r * qstar <= 0.5 <= 1.0 - r * (1.0 - qstar):
        return q_at(centers, q_iso, qstar), "barrier"      # the jump straddles 1/2
    if r * qstar > 0.5:                                       # reached 1/2 before the barrier
        return q_at(centers, q_iso, 0.5 / r), "U_side"
    return q_at(centers, q_iso, 1.0 - 0.5 / r), "F_side"



def profile(trajs, args):
    """Model committor along Q from excursion outcomes, plus the frame table."""
    ti, fr, qv, oc = committor_frames(trajs, args.qu, args.qf)
    centers, n, raw, iso = committor_profile(qv, oc, args.qu, args.qf, args.committor_bins)
    return dict(ti=ti, fr=fr, qv=qv, oc=oc, centers=centers, n=n, raw=raw, iso=iso)


def lambdas_from(args):
    if args.from_stitch:
        with open(args.from_stitch) as fh:
            rows = list(csv.DictReader(fh))
        qts = args.qts if args.qts is not None else None
        pick = min(rows, key=lambda r: abs(float(r["qts"]) - qts)) if qts else rows[0]
        base = float(pick["lam_stitch"]) if pick["lam_stitch"] not in ("", "nan") else 0.0
        return [base, base + 1.0, base + 2.0]
    return list(args.lams) if args.lams else [0.0, 1.0, 2.0]


def main(argv=None):
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, qts_help="attempt interface used for p and for the validity check")
    ap.add_argument("--barrier-q", type=float, default=0.5,
                    help="assumed barrier location as a MODEL committor value q* (default 0.5)")
    ap.add_argument("--lams", type=float, nargs="*", default=None,
                    help="lambda values (default 0 1 2, or lambda_min + 0/1/2 with --from-stitch)")
    ap.add_argument("--from-stitch", default=None,
                    help="summary.csv from maxcal-stitch: take lambda_min at the reference Q‡")
    ap.add_argument("--committor-bins", type=int, default=20)
    ap.add_argument("--tse-window", type=float, default=None,
                    help="half-width in Q of the TSE window around Q_TS (default: one bin)")
    args = ap.parse_args(argv)

    trajs = load_trajectories(args.files, args.dt, args.qu, args.qf)
    _, qref = qts_grid(args)
    os.makedirs(args.out, exist_ok=True)
    P = profile(trajs, args)
    centers, iso = P["centers"], P["iso"]
    with open(os.path.join(args.out, "committor.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["Q", "n_frames", "q_model_raw", "q_model_isotonic"])
        for row in zip(centers, P["n"], P["raw"], iso):
            wr.writerow([f"{v:.6g}" for v in row])

    A = attempts(trajs, qref, args.t0, args.include_initial)
    p = success_prob(A["k"], A["done"])
    ok = np.isfinite(iso)
    q_iface = float(np.interp(qref, centers[ok], iso[ok])) if ok.any() else np.nan
    lams = lambdas_from(args)
    qstars = np.round(np.arange(0.1, 0.91, 0.05), 3)
    half = args.tse_window if args.tse_window else (args.qf - args.qu) / args.committor_bins

    with open(os.path.join(args.out, "ts_location.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["lambda", "r", "q_star", "Q_barrier", "Q_TS", "ts_set_by", "status"])
        for lam in lams:
            r = tilted_p(p, lam) / p
            for qs in sorted(set(qstars) | {args.barrier_q}):
                qb = q_at(centers, iso, qs)
                qts_, where = ts_location(centers, iso, r, qs)
                status = ("barrier_before_interface" if qs < q_iface else
                          "ts_outside_sampled_range" if not np.isfinite(qts_) else
                          "ts_before_interface" if qts_ <= qref else "ok")
                wr.writerow([f"{lam:.4g}", f"{r:.4g}", f"{qs:.3g}", f"{qb:.4g}",
                             f"{qts_:.4g}", where, status])

    print(f"model TS (q = 0.5) at Q = {q_at(centers, iso, 0.5):.3f}; assumed barrier "
          f"(q* = {args.barrier_q:.2f}) at Q = {q_at(centers, iso, args.barrier_q):.3f}; "
          f"p = {p:.3f} at Q‡ = {qref:.3f}")
    if args.barrier_q < q_iface:
        print(f"  WARNING: q* lies before the attempt interface (model q = {q_iface:.2f} "
              "at Q‡); the tilt assumes the barrier is beyond Q‡.")
    for lam in lams:
        r = tilted_p(p, lam) / p
        qts_, where = ts_location(centers, iso, r, args.barrier_q)
        sel = np.abs(P["qv"] - qts_) <= half if np.isfinite(qts_) else np.zeros(P["qv"].size, bool)
        name = f"tse_committor_lam{lam:.2f}_q{args.barrier_q:.2f}.csv"
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            wr = csv.writer(fh, lineterminator="\n")
            wr.writerow(["trajectory", "frame", "time", "Q", "excursion_outcome"])
            for i, f_, q_, o_ in zip(P["ti"][sel], P["fr"][sel], P["qv"][sel], P["oc"][sel]):
                wr.writerow([trajs[i]["path"], f_, f"{f_ * trajs[i]['dt']:.6g}",
                             f"{q_:.6g}", int(o_)])
        print(f"  lambda = {lam:.2f} (r = {r:.3f}): Q_TS = {qts_:.3f} set by {where:8s}"
              f" -> {int(sel.sum())} frames in {name}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(centers, P["raw"], "o", ms=3, color="0.5", label="model (raw)")
    axs[0].plot(centers, iso, "k-", label="model (isotonic)")
    grid = np.linspace(args.qu, args.qf, 400)
    qm_grid = np.interp(grid, centers[ok], iso[ok])
    for lam in lams:
        r = tilted_p(p, lam) / p
        axs[0].plot(grid, corrected_committor(qm_grid, r, args.barrier_q),
                    label=f"corrected, lambda={lam:.2f}")
    axs[0].axhline(0.5, color="k", ls=":", lw=1)
    axs[0].axvline(qref, color="r", ls=":", lw=1, label="Q‡")
    axs[0].set_xlabel("Q"); axs[0].set_ylabel("committor"); axs[0].legend(fontsize=8)
    axs[0].set_title(f"missing barrier at model q* = {args.barrier_q:.2f}")
    for lam in lams:
        r = tilted_p(p, lam) / p
        axs[1].plot(qstars, [ts_location(centers, iso, r, qs)[0] for qs in qstars], "o-", ms=3,
                    label=f"lambda = {lam:.2f}")
    axs[1].plot(qstars, [q_at(centers, iso, qs) for qs in qstars], "k--", lw=1,
                label="barrier location")
    axs[1].axvspan(0, q_iface, color="0.9", label="before Q‡ (not allowed)")
    axs[1].set_xlim(qstars[0], qstars[-1])
    axs[1].set_xlabel("assumed barrier location q* (model committor)")
    axs[1].set_ylabel("Q_TS"); axs[1].set_title("TS location vs assumed barrier location")
    axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ts_location.png"), dpi=150)
    plt.close(fig)
    print(f"\nWrote committor.csv, ts_location.csv/.png, tse_committor_*.csv to {args.out}/")


if __name__ == "__main__":
    main()
