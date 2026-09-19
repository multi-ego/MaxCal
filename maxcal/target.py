#!/usr/bin/env python3
"""
maxcal-target - reweight binding/unbinding (or folding/unfolding) trajectories so that
their first-passage times match experimental rates AND a Poisson (single-exponential)
target, then use the weights for structural analysis.

Motivation: a model such as multi-eGO can be systematically faster than experiment (a
global clock factor) while getting the *relative* rates only approximately right, and its
first-passage times need not be exactly exponential.  Here the clock factor is fitted once
across all systems, each system's target is then fully specified (an exponential with the
experimental mean in model time units), and the trajectories are reweighted onto it with
minimum perturbation.  The weights are the output: applying them to structural observables
shows which trajectories, and which mechanisms, the experimental kinetics favour.

Weights (default, mode "target"): with sorted times t_(1) < ... < t_(N) and midpoints m_i,
    w_i ∝ F_target(m_i) - F_target(m_{i-1}),   m_0 = 0, m_N = ∞
i.e. each trajectory carries the target probability mass of its own interval.  This matches
the target mean and shape without any density estimate.  Mode "mean" instead imposes only
the mean, w_i ∝ exp(-θ t_i) with θ from the target mean: the maximum-entropy solution under
a single constraint, useful as a comparison.

Input: one CSV describing the systems, e.g.

    system,kon_exp,koff_exp,bind_times,unbind_times
    EQVTAV_WT,2.6,22,times/EQVTAV_WT_bind.dat,times/EQVTAV_WT_unbind.dat
    EQVTAV_L18A,2.4,10.4,times/EQVTAV_L18A_bind.dat,times/EQVTAV_L18A_unbind.dat

kon_exp in uM^-1 s^-1, koff_exp in s^-1 (for folding/unfolding use --system-kind first-order
and give both rates in s^-1).  Each times file holds one first-passage time per line, in the
model's own time units (ps, ns, ...), in the order of the trajectories.

Example:
    maxcal-target systems.csv --conc 0.017 --clock decrease-only --out target_out
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import brentq

import matplotlib.pyplot as plt   # backend set in main()

AVOGADRO_uM = 1e-6                # 1 uM in M


# --------------------------------------------------------------------------- input
def read_times(path):
    t = np.loadtxt(path, comments=("#", "@"), ndmin=1)
    t = np.asarray(t, float).ravel()
    if t.size == 0 or np.any(t <= 0):
        sys.exit(f"{path}: needs positive first-passage times, one per line")
    return t


def read_systems(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    need = {"system", "kon_exp", "koff_exp", "bind_times", "unbind_times"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"{path}: needs columns {sorted(need)}")
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for r in rows:
        f_b = r["bind_times"] if os.path.isabs(r["bind_times"]) else os.path.join(base, r["bind_times"])
        f_u = r["unbind_times"] if os.path.isabs(r["unbind_times"]) else os.path.join(base, r["unbind_times"])
        out.append(dict(name=r["system"], kon=float(r["kon_exp"]), koff=float(r["koff_exp"]),
                        t_bind=read_times(f_b), t_unbind=read_times(f_u),
                        file_bind=f_b, file_unbind=f_u))
    return out


# ------------------------------------------------------------------------- clock
def model_rates(s, conc, kind):
    """Model rate constants in experimental units, from the mean first-passage times
    (the MLE for an exponential).  Binding is pseudo-first-order at concentration conc."""
    tb, tu = s["t_bind"].mean(), s["t_unbind"].mean()
    if kind == "binding":
        return 1.0 / (tb * conc / AVOGADRO_uM), 1.0 / tu      # per (uM * model time), per model time
    return 1.0 / tb, 1.0 / tu


def acceleration(systems, conc, kind):
    """A = k_model / k_exp for both directions, in consistent units up to the clock."""
    A = {}
    for s in systems:
        km_on, km_off = model_rates(s, conc, kind)
        A[s["name"]] = (km_on / s["kon"], km_off / s["koff"])
    return A


def neff_fraction(f):
    """N_eff/N for an exponential sample tilted to change its mean by a factor f
    (exact for an exponential): 2f - f^2; <= 0 means unreachable."""
    return 2 * f - f * f


def choose_clock(A, mode):
    vals = np.array([a for pair in A.values() for a in pair])
    if mode == "geometric":
        return float(np.exp(np.log(vals).mean()))
    if mode == "decrease-only":
        return float(vals.max())              # every system then needs a slowdown factor <= 1
    if mode == "balanced":                    # maximise the worst-case N_eff fraction
        grid = np.exp(np.linspace(np.log(vals.min()), np.log(vals.max() * 1.5), 600))
        worst = [min(neff_fraction(f) for f in vals / c) for c in grid]
        return float(grid[int(np.argmax(worst))])
    return float(mode)                        # an explicit number


# ------------------------------------------------------------------------ weights
def target_weights(t, tau, match_mean=True):
    """Density-ratio weights onto an exponential target with mean tau (mode 'target').

    w_i ∝ f_target(t_i) / f_model(t_i), with f_model a gamma fitted to the times by maximum
    likelihood: a smooth two-parameter stand-in for the model's own first-passage density.
    Smoothness matters: a nearest-neighbour (spacing) estimate would add noise and cost
    N_eff even when the model already matches the target.  When the target and the model
    agree the weights are flat, and N_eff ≈ N.  A final exponential tilt matches the target
    mean exactly (the experimental rate is the primary constraint); the sample cannot
    represent target mass beyond the longest observed time, reported as tail_gap."""
    a, _, scale = stats.gamma.fit(t, floc=0.0)
    logw = stats.expon.logpdf(t, scale=tau) - stats.gamma.logpdf(t, a, loc=0.0, scale=scale)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    if match_mean:
        m = lambda th: np.average(t, weights=w * np.exp(-th * (t - t.mean()))) - tau
        lo, hi = -0.999 / t.max(), 50.0 / t.mean()
        if m(lo) * m(hi) < 0:
            th = brentq(m, lo, hi)
            w = w * np.exp(-th * (t - t.mean()))
            w /= w.sum()
    return w


def mean_weights(t, tau):
    """Exponential tilt w ∝ exp(-θ t) with the mean constrained to tau (mode 'mean')."""
    lo, hi = -0.999 / t.max(), 50.0 / t.mean()
    f = lambda th: np.average(t, weights=np.exp(-th * (t - t.mean()))) - tau
    if f(lo) * f(hi) > 0:
        return None                            # target outside the reachable range
    th = brentq(f, lo, hi)
    w = np.exp(-th * (t - t.mean()))
    return w / w.sum()


def diagnostics(t, w, tau):
    neff = 1.0 / np.sum(w ** 2)
    o = np.argsort(t)
    ts, ws = t[o], w[o]
    F = np.cumsum(ws)
    T = 1.0 - np.exp(-ts / tau)
    D = float(np.max(np.maximum(np.abs(F - T), np.abs(np.r_[0, F[:-1]] - T))))
    return dict(neff=float(neff), mean_w=float(np.sum(w * t)), ks_D=D,
                ks_p=float(stats.kstwo.sf(D, max(int(round(neff)), 2))),
                w_max=float(w.max()), tail_gap=float(np.exp(-t.max() / tau)))


# --------------------------------------------------------------------------- main
def analyse(systems, args):
    A = acceleration(systems, args.conc, args.system_kind)
    c = choose_clock(A, args.clock)
    res = []
    for s in systems:
        row = dict(system=s["name"])
        for direction, k_exp, t in (("bind", s["kon"], s["t_bind"]),
                                    ("unbind", s["koff"], s["t_unbind"])):
            a = A[s["name"]][0 if direction == "bind" else 1]
            factor = a / c                                    # needed change of the mean time
            tau_target = t.mean() * factor
            w = (target_weights(t, tau_target) if args.mode == "target"
                 else mean_weights(t, tau_target))
            d = dict(A=a, factor=factor, tau_model=float(t.mean()), tau_target=float(tau_target),
                     N=int(t.size), neff_expected=float(neff_fraction(factor) * t.size))
            d.update(diagnostics(t, w, tau_target) if w is not None else
                     dict(neff=np.nan, mean_w=np.nan, ks_D=np.nan, ks_p=np.nan,
                          w_max=np.nan, tail_gap=np.nan))
            row[direction] = d
            row[direction + "_w"] = w
            row[direction + "_t"] = t
        res.append(row)
    return c, A, res


def write_outputs(c, res, systems, args):
    os.makedirs(args.out, exist_ok=True)
    cols = ["system", "direction", "N", "tau_model", "tau_target", "needed_factor",
            "N_eff", "N_eff_expected", "weighted_mean", "ks_D", "ks_p", "max_weight", "tail_gap"]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(cols)
        for r in res:
            for direction in ("bind", "unbind"):
                d = r[direction]
                wr.writerow([r["system"], direction, d["N"], f"{d['tau_model']:.6g}",
                             f"{d['tau_target']:.6g}", f"{d['factor']:.4g}", f"{d['neff']:.4g}",
                             f"{d['neff_expected']:.4g}", f"{d['mean_w']:.6g}", f"{d['ks_D']:.4g}",
                             f"{d['ks_p']:.4g}", f"{d['w_max']:.4g}", f"{d['tail_gap']:.3g}"])
    for r, s in zip(res, systems):
        for direction in ("bind", "unbind"):
            w, t = r[direction + "_w"], r[direction + "_t"]
            if w is None:
                continue
            with open(os.path.join(args.out, f"weights_{r['system']}_{direction}.csv"),
                      "w", newline="") as fh:
                wr = csv.writer(fh, lineterminator="\n")
                wr.writerow(["trajectory", "time", "weight"])
                for i, (ti, wi) in enumerate(zip(t, w)):
                    wr.writerow([i, f"{ti:.6g}", f"{wi:.6g}"])

    n = len(res)
    fig, axs = plt.subplots(2, 1, figsize=(1.1 * n + 3, 6.5), sharex=True)
    x = np.arange(n)
    for k, direction in enumerate(("bind", "unbind")):
        axs[0].bar(x + (k - 0.5) * 0.4, [r[direction]["factor"] for r in res], 0.4, label=direction)
        axs[1].bar(x + (k - 0.5) * 0.4, [r[direction]["neff"] for r in res], 0.4, label=direction)
    axs[0].axhline(1, color="k", lw=1); axs[0].set_ylabel("needed factor on ⟨t⟩")
    axs[0].set_title(f"clock factor c = {c:.3g}  (model faster than experiment)")
    axs[1].set_ylabel("N_eff"); axs[1].set_xticks(x)
    axs[1].set_xticklabels([r["system"] for r in res], rotation=45, ha="right")
    for a in axs:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "calibration.png"), dpi=150)
    plt.close(fig)

    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    for direction in ("bind", "unbind"):
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), squeeze=False)
        for ax, r in zip(axs.ravel(), res):
            t, w = r[direction + "_t"], r[direction + "_w"]
            o = np.argsort(t); ts = t[o]
            ax.step(ts, np.arange(1, ts.size + 1) / ts.size, where="post", color="k", label="model")
            if w is not None:
                ax.step(ts, np.cumsum(w[o]), where="post", color="C3", label="reweighted")
            xx = np.logspace(np.log10(ts.min() / 2), np.log10(ts.max() * 3), 200)
            ax.plot(xx, 1 - np.exp(-xx / r[direction]["tau_target"]), "--", color="C0",
                    label="target (exp.)")
            ax.set_xscale("log"); ax.set_title(f"{r['system']} {direction}", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"cdf_{direction}.png"), dpi=150)
        plt.close(fig)


def report(c, res, args):
    print(f"clock factor c = {c:.4g} ({args.clock}); model time x c = experimental time\n")
    print(f"{'system':22s} {'dir':7s} {'factor':>7s} {'N_eff':>7s} {'exp.':>6s} "
          f"{'KS p':>6s} {'w_max':>6s} {'tail':>6s}")
    for r in res:
        for direction in ("bind", "unbind"):
            d = r[direction]
            flag = ""
            if not np.isfinite(d["neff"]):
                flag = "  target unreachable with this clock"
            elif d["neff"] < 0.2 * d["N"]:
                flag = "  low N_eff"
            elif d["tail_gap"] > 0.05:
                flag = "  target tail beyond the longest observed time"
            print(f"{r['system']:22s} {direction:7s} {d['factor']:7.2f} {d['neff']:7.1f} "
                  f"{d['neff_expected']:6.1f} {d['ks_p']:6.2f} {d['w_max']:6.3f} "
                  f"{d['tail_gap']:6.3f}{flag}")
    if args.mode == "target":
        print("\n(in 'target' mode the KS check is satisfied by construction: read N_eff, "
              "max_weight and tail instead)")
    worst = min((r[x]["neff"] for r in res for x in ("bind", "unbind")
                 if np.isfinite(r[x]["neff"])), default=np.nan)
    print(f"\nworst N_eff = {worst:.1f}."
          " Try --clock decrease-only or balanced if some targets are unreachable.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems", help="CSV describing the systems (see the module docstring)")
    ap.add_argument("--conc", type=float, default=0.017,
                    help="peptide/ligand concentration in the simulations, in M (binding only)")
    ap.add_argument("--system-kind", choices=["binding", "first-order"], default="binding")
    ap.add_argument("--clock", default="balanced",
                    help="geometric | decrease-only | balanced | an explicit number")
    ap.add_argument("--mode", choices=["target", "mean"], default="target",
                    help="'target': match the experimental mean and the exponential shape; "
                         "'mean': match only the mean (exponential tilt)")
    ap.add_argument("--out", default="maxcal_target_out")
    return ap.parse_args(argv)


def main(argv=None):
    plt.switch_backend("Agg")
    args = parse_args(argv)
    systems = read_systems(args.systems)
    c, A, res = analyse(systems, args)
    write_outputs(c, res, systems, args)
    report(c, res, args)
    print(f"\nWrote summary.csv, weights_*.csv, calibration.png, cdf_bind.png, cdf_unbind.png "
          f"to {args.out}/")


if __name__ == "__main__":
    main()
