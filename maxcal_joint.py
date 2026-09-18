#!/usr/bin/env python3
"""
maxcal_joint.py - joint MaxCal analysis of forward (A->B) and backward (B->A)
first-passage trajectories: folding/unfolding or binding/unbinding.

Uses the single-direction machinery of maxcal_poisson.py (attempt counting, tilt,
stitching, Lilliefors test) and adds what only the pair of directions can give.

--temperature-relation same
    Both sets sample the same ensemble.  One barrier correction must explain both
    directions: the forward tilt lambda_f is scanned and the backward success
    probability is fixed by detailed balance,
        k'_AB / k'_BA = R_target = exp(-DeltaG_box / kT),
    where DeltaG_box is the model's own kinetic value (default) or --dG.
    Also reported: kinetic DeltaG (compare with --dG), time-reversal symmetry of
    the transition paths (durations, number of Q‡ recrossings), and the comparison
    of forward and backward transition-state ensembles.
--temperature-relation different
    The directions are analysed independently (lambda is not transferable across T,
    and DeltaG changes with T).  No kinetic DeltaG and no detailed-balance coupling;
    TSE differences are reported but are expected (Hammond shift).

Coordinate: one collective variable per frame.  A and B are the basins: if qa < qb,
A is Q < qa and B is Q >= qb (e.g. fraction of native/intermolecular contacts);
if qa > qb the orientation is reversed automatically (e.g. a distance, A = unbound
at d > qa, B = bound at d <= qb).  All thresholds are given in the original units.

Input files: columns time, Q[, f1, f2, ...].  Optional extra columns are per-frame
structural features (e.g. per-residue native-contact fractions) used to compare the
forward and backward TSE.  Lines starting with '#' or '@' are ignored.

Forward trajectories start in A and stop on reaching B; backward ones start in B and
stop on reaching A.

Example (folding, same temperature):
  python maxcal_joint.py --fwd "fold/*.xvg" --bwd "unfold/*.xvg" \\
      --qa 0.3 --qb 0.8 --qts-fwd 0.4 --qts-bwd 0.7 --qtse 0.55 \\
      --temperature-relation same --dG -2.5 --out joint
Binding (distance CV, same temperature, one ligand in a box of V nm^3):
  python maxcal_joint.py --fwd "on/*.xvg" --bwd "off/*.xvg" --system binding \\
      --qa 2.0 --qb 0.6 --qts-fwd 1.4 --qts-bwd 0.8 --qtse 1.0 \\
      --temperature-relation same --dG -8.0 --box-volume 343 --out joint
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import maxcal_poisson as mp

C_STANDARD = 0.602214076  # 1 M in molecules per nm^3


# ----------------------------------------------------------------------------- I/O
def load_file(path, dt_arg):
    data = np.loadtxt(path, comments=("#", "@"), ndmin=2)
    if data.shape[1] == 1:
        if dt_arg is None:
            sys.exit(f"{path}: single column of Q, please pass --dt")
        return data[:, 0].astype(float), float(dt_arg), None
    t, q = data[:, 0], data[:, 1]
    feats = data[:, 2:] if data.shape[1] > 2 else None
    if dt_arg is not None:
        return q, float(dt_arg), feats
    d = np.diff(t)
    dt = float(np.median(d)) if d.size else 1.0
    if d.size and np.max(np.abs(d - dt)) > 1e-6 * abs(dt):
        warnings.warn(f"{path}: non-uniform time spacing, using median dt={dt}")
    return q, dt, feats


def expand(patterns):
    return sorted({p for pat in patterns for p in (glob.glob(pat) or [pat])})


# --------------------------------------------------------------------- geometry
def orient(qa, qb, *levels):
    """Return sign s such that s*Q has A on the low side, and the oriented levels."""
    s = 1.0 if qa < qb else -1.0
    return s, [s * x if x is not None else None for x in (qa, qb) + levels]


def build(raw, lo, hi, sign):
    """Parse raw (path, q, dt, feats) with basin 'lo' (start) below `lo` and
    target at >= `hi`, after multiplying Q by `sign` (+1 forward, -1 backward)."""
    out = []
    for path, q, dt, feats in raw:
        qq = sign * q
        fold, fails = mp.parse_traj(qq, lo, hi)
        out.append(dict(path=path, q=qq, dt=dt, n=qq.size, fold=fold, fails=fails,
                        feats=feats))
    return out


def transition_paths(trajs, lo, hi, qtse):
    """Transition path of each completed trajectory: from the last frame in the
    starting basin (Q < lo) to the first frame in the target (Q >= hi).
    Returns durations, number of crossings, and TSE frames = ALL crossings of qtse
    inside the path (the bracketing frame closer to qtse), each weighted
    1/(number of crossings) so that every path counts once.  Using all crossings makes
    the forward and backward definitions time-reversal symmetric.  Under the MaxCal
    tilt the transition paths are unchanged (conditional invariance), so these frames
    need no reweighting."""
    dur, ncross, frames = [], [], []
    for i, tr in enumerate(trajs):
        f = tr["fold"]
        if f is None:
            continue
        q = tr["q"]
        inA = np.flatnonzero(q[:f] < lo)
        if inA.size == 0:
            continue                         # never in the start basin: no defined path
        s = inA[-1]
        dur.append((f - s) * tr["dt"])
        seg = q[s:f + 1] >= qtse
        j = np.flatnonzero(seg[1:] != seg[:-1]) + 1 + s      # crossing between j-1 and j
        # of the two frames bracketing a crossing, keep the one closer to qtse:
        # symmetric under time reversal (the first-frame-after rule is not)
        j = np.where(np.abs(q[j - 1] - qtse) < np.abs(q[j] - qtse), j - 1, j)
        ncross.append(j.size)
        for fr in j:
            frames.append((i, int(fr), 1.0 / j.size))
    return np.array(dur), np.array(ncross), frames


# ------------------------------------------------------------- kinetics helpers
def renewal_mean(pp, muC, muS):
    """Mean first-passage time of the stitched (renewal) process."""
    return muC * (1.0 - pp) / pp + muS


def solve_p(target_mean, muC, muS):
    """Success probability giving a renewal mean equal to target_mean (inverse of
    renewal_mean); nan if the target is shorter than one success segment."""
    x = (target_mean - muS) / muC
    return 1.0 / (1.0 + x) if x > 0 else np.nan


def logit(p):
    return np.log(p / (1.0 - p))


def dG_box_to_standard(dG_box, volume_nm3):
    """Binding with one ligand and one receptor in volume V:
    K_box = P_bound/P_unbound = k_on' / k_off,  K_a = K_box * V,
    DeltaG° = -kT ln(K_a C°) = DeltaG_box - ln(V C°)   (kT units)."""
    return dG_box - np.log(volume_nm3 * C_STANDARD)


def dG_standard_to_box(dG_std, volume_nm3):
    return dG_std + np.log(volume_nm3 * C_STANDARD)


def poisson_check(pp, A, n_sub, args, rng):
    t = mp.stitch(pp, A["fail_pool"], A["succ_pool"], args.nstitch, rng)
    cv = t.std() / t.mean()
    sub = rng.choice(t, size=(args.nsub, n_sub))
    null = mp.null_D(n_sub)
    pv = (np.sum(null[None, :] >= mp.D_rows(sub)[:, None], axis=1) + 1) / (null.size + 1)
    mpv = float(np.median(pv))
    return cv, mpv, (mpv >= args.alpha and abs(cv - 1) <= args.cv_tol), t


# ------------------------------------------------------------------ one direction
def direction(trajs, lo, hi, qts, qtse, args):
    A = mp.attempts(trajs, qts, args.t0, args.include_initial)
    k, T, done = A["k"], A["T"], A["done"]
    p = mp.success_prob(k, done)
    w0 = np.full(k.size, 1.0 / k.size)
    D0, tau = mp.ks_exp_weighted(T, done, w0)
    muC = A["fail_pool"].mean() if A["fail_pool"].size else np.nan
    muS = A["succ_pool"].mean() if A["succ_pool"].size else np.nan
    Tc = T[done]
    geom_p, _ = mp.geometric_test(k[done])
    pr = A["pairs"]
    rho, rho_p = stats.spearmanr(pr[:, 0], pr[:, 1]) if len(pr) >= 10 else (np.nan, np.nan)
    dur, ncross, frames = transition_paths(trajs, lo, hi, qtse)
    return dict(A=A, p=p, N=int(k.size), n_done=int(done.sum()), sum_k=int(k.sum()),
                geom_p=geom_p, lag1_rho=rho, lag1_p=rho_p,
                cv0=float(Tc.std() / Tc.mean()) if Tc.size else np.nan,
                ks_p0=mp.lilliefors_p(D0, k.size), tau_mle=tau, muC=muC, muS=muS,
                M0=renewal_mean(p, muC, muS), tp_dur=dur, tp_ncross=ncross,
                tse=frames, trajs=trajs)


# --------------------------------------------------------------------- scans
def joint_scan(F, B, R_target, lams, args, rng):
    """Same temperature: scan lambda_f; the backward success probability follows from
    detailed balance, M'_BA = R_target * M'_AB with M the renewal mean FPTs
    (k_AB/k_BA = M_BA/M_AB).  Returns the first lambda_f of 3 consecutive joint passes."""
    rows, streak, first, lam_joint = [], 0, None, np.nan
    for lam in lams:
        pf = mp.tilted_p(F["p"], lam)
        Mf = renewal_mean(pf, F["muC"], F["muS"])
        pb = solve_p(R_target * Mf, B["muC"], B["muS"])
        if not np.isfinite(pb) or not (0 < pb <= 1):
            rows.append(dict(lam_f=lam, p_f=pf, p_b=np.nan, lam_b=np.nan))
            streak = 0
            continue
        if (1 - pf) / pf > 1e7 or (1 - pb) / pb > 1e7:
            break
        cvf, kpf, okf, _ = poisson_check(pf, F["A"], F["N"], args, rng)
        cvb, kpb, okb, _ = poisson_check(pb, B["A"], B["N"], args, rng)
        rows.append(dict(lam_f=lam, p_f=pf, p_b=pb, lam_b=logit(B["p"]) - logit(pb),
                         cv_f=cvf, ksp_f=kpf, cv_b=cvb, ksp_b=kpb, ok=okf and okb))
        if okf and okb:
            first = lam if streak == 0 else first
            streak += 1
            if streak >= 3:
                lam_joint = first
                break
        else:
            streak = 0
    return lam_joint, rows


def row_at(rows, lam):
    for r in rows:
        if np.isclose(r["lam_f"], lam):
            return r
    return None


# ------------------------------------------------------------------ TSE features
def tse_features(D, n_boot, rng):
    """Weighted mean feature vector over TSE crossing frames, with a bootstrap SE
    obtained by resampling transition paths."""
    fr = D["tse"]
    if not fr or D["trajs"][fr[0][0]]["feats"] is None:
        return None
    by_path = {}
    for i, f, w in fr:
        by_path.setdefault(i, []).append(D["trajs"][i]["feats"][f] * w)
    per_path = np.array([np.sum(v, axis=0) for v in by_path.values()])  # each path sums to 1 weight
    mean = per_path.mean(axis=0)
    bs = np.array([per_path[rng.integers(0, len(per_path), len(per_path))].mean(axis=0)
                   for _ in range(n_boot)])
    return mean, bs.std(axis=0), len(per_path)


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fwd", nargs="+", required=True, help="forward (A->B) trajectories")
    ap.add_argument("--bwd", nargs="+", required=True, help="backward (B->A) trajectories")
    ap.add_argument("--temperature-relation", choices=["same", "different"], required=True,
                    help="were forward and backward sets simulated at the same temperature?")
    ap.add_argument("--system", choices=["folding", "binding"], default="folding")
    ap.add_argument("--qa", type=float, required=True, help="A basin boundary")
    ap.add_argument("--qb", type=float, required=True, help="B basin boundary")
    ap.add_argument("--qts-fwd", type=float, required=True,
                    help="forward attempt interface, at the foot of the barrier on the A side")
    ap.add_argument("--qts-bwd", type=float, required=True,
                    help="backward attempt interface, at the foot of the barrier on the B side")
    ap.add_argument("--qtse", type=float, default=None,
                    help="TSE surface (barrier top); default: midpoint of the two interfaces")
    ap.add_argument("--dG", type=float, default=None,
                    help="G_B - G_A in kT (binding: standard DeltaG°, needs --box-volume); "
                         "same-T only: imposed as the detailed-balance target")
    ap.add_argument("--box-volume", type=float, default=None, help="binding: box volume in nm^3")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--lam-max", type=float, default=10.0)
    ap.add_argument("--nlam", type=int, default=201)
    ap.add_argument("--nstitch", type=int, default=5000)
    ap.add_argument("--nsub", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--cv-tol", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples for TSE features")
    ap.add_argument("--include-initial", action="store_true")
    ap.add_argument("--out", default="maxcal_joint_out")
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args(argv)


def analyse(args):
    same = args.temperature_relation == "same"
    rng = np.random.default_rng(args.seed)
    qtse = args.qtse if args.qtse is not None else 0.5 * (args.qts_fwd + args.qts_bwd)
    s, (qa, qb, qtf, qtb, qte) = orient(args.qa, args.qb, args.qts_fwd, args.qts_bwd, qtse)
    if not (qa < qtf < qb and qa < qtb < qb and qa < qte < qb):
        sys.exit("interfaces and --qtse must lie between the two basin boundaries")
    notes = []
    if not qtf <= qte <= qtb:
        notes.append("expected order: qts-fwd (A-side foot) <= qtse <= qts-bwd (B-side foot)")
    if args.system == "binding" and args.dG is not None and args.box_volume is None:
        sys.exit("--system binding with --dG needs --box-volume (nm^3)")
    if not same and args.dG is not None:
        notes.append("--dG ignored: DeltaG differs between the two temperatures")

    raw_f = [(p,) + load_file(p, args.dt) for p in expand(args.fwd)]
    raw_b = [(p,) + load_file(p, args.dt) for p in expand(args.bwd)]
    # forward: start below qa, target >= qb.  backward: mirror (-sQ): start below -qb, target >= -qa
    F = direction(build(raw_f, qa, qb, s), qa, qb, qtf, qte, args)
    B = direction(build(raw_b, -qb, -qa, -s), -qb, -qa, -qtb, -qte, args)
    lams = np.linspace(0.0, args.lam_max, args.nlam)

    res = dict(temperature_relation=args.temperature_relation, system=args.system,
               orientation="A low" if s > 0 else "A high (CV negated internally)",
               qtse=qtse, notes=notes)
    for name, D in (("fwd", F), ("bwd", B)):
        res[name] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in D.items()
                     if k in ("p", "N", "n_done", "sum_k", "geom_p", "lag1_rho", "lag1_p",
                              "cv0", "ks_p0", "muC", "muS", "M0", "tau_mle")}
        res[name]["mean_tp_duration"] = float(D["tp_dur"].mean()) if D["tp_dur"].size else None
        res[name]["mean_tse_crossings"] = float(D["tp_ncross"].mean()) if D["tp_ncross"].size else None
        if np.isfinite(D["geom_p"]) and D["geom_p"] < args.alpha:
            notes.append(f"{name}: failed-attempt counts not geometric (memory)")

    # ---- transition-path time-reversal symmetry
    if F["tp_dur"].size >= 5 and B["tp_dur"].size >= 5:
        res["tp_duration_ks_p"] = float(stats.ks_2samp(F["tp_dur"], B["tp_dur"]).pvalue)
        res["tp_crossings_ks_p"] = float(stats.ks_2samp(F["tp_ncross"], B["tp_ncross"]).pvalue)
        if same and res["tp_duration_ks_p"] < args.alpha:
            notes.append("forward and backward transition-path durations differ although the "
                         "temperature is the same: check ensembles, basin definitions, or "
                         "non-equilibrium starting conditions")

    # ---- kinetic DeltaG and the tilt
    if same:
        R_model = B["M0"] / F["M0"]              # k_AB/k_BA = M_BA/M_AB
        dG_box_model = -np.log(R_model)
        dG_box_raw = -np.log(B["tau_mle"] / F["tau_mle"])
        res["dG_kin_box_renewal"] = float(dG_box_model)
        res["dG_kin_box_raw"] = float(dG_box_raw)
        if args.system == "binding" and args.box_volume:
            res["dG_kin_standard_renewal"] = float(dG_box_to_standard(dG_box_model, args.box_volume))
            res["dG_kin_standard_raw"] = float(dG_box_to_standard(dG_box_raw, args.box_volume))
        if args.dG is not None:
            dG_box_t = (dG_standard_to_box(args.dG, args.box_volume)
                        if args.system == "binding" else args.dG)
            res["dG_target_box"] = float(dG_box_t)
            res["dG_kin_minus_target"] = float(dG_box_model - dG_box_t)
            if abs(dG_box_model - dG_box_t) > 0.5:
                notes.append(f"model kinetic DeltaG differs from --dG by "
                             f"{dG_box_model - dG_box_t:+.2f} kT: the joint correction "
                             "will tilt the two directions asymmetrically")
            R_target = np.exp(-dG_box_t)
        else:
            R_target = R_model
        lam_j, rows = joint_scan(F, B, R_target, lams, args, rng)
        res["lambda_joint_f"] = float(lam_j)
        r = row_at(rows, lam_j) if np.isfinite(lam_j) else None
        res["lambda_joint_b"] = float(r["lam_b"]) if r else None
        res["p_f_tilted"] = float(r["p_f"]) if r else None
        res["p_b_tilted"] = float(r["p_b"]) if r else None
        if r:
            res["barrier_shift_f"] = float(-np.log(r["p_f"] / F["p"]))
            res["barrier_shift_b"] = float(-np.log(r["p_b"] / B["p"]))
            if r["lam_b"] < 0:
                notes.append("backward tilt is negative at the joint solution: "
                             "the imposed DeltaG requires faster backward crossing")
        else:
            notes.append("no joint lambda gives Poisson statistics in both directions")
        res["scan"] = [{k: (float(v) if isinstance(v, (float, np.floating, bool, np.bool_)) else v)
                        for k, v in row.items()} for row in rows]
        pf_plot = r["p_f"] if r else None
        pb_plot = r["p_b"] if r else None
    else:
        out = {}
        for name, D in (("fwd", F), ("bwd", B)):
            lam, _ = mp.stitch_scan(lams, D["p"], D["A"], D["N"], args, rng)
            out[name] = lam
            res[f"lambda_{name}_independent"] = float(lam)
        res["dG_kin_box_renewal"] = None
        pf_plot = mp.tilted_p(F["p"], out["fwd"]) if np.isfinite(out["fwd"]) else None
        pb_plot = mp.tilted_p(B["p"], out["bwd"]) if np.isfinite(out["bwd"]) else None

    # ---- TSE comparison
    tf, tb = tse_features(F, args.boot, rng), tse_features(B, args.boot, rng)
    if tf is not None and tb is not None:
        (mf, sf, nf), (mb, sb, nb) = tf, tb
        z = (mf - mb) / np.sqrt(sf ** 2 + sb ** 2 + 1e-300)
        r_prof = (float(np.corrcoef(mf, mb)[0, 1])
                  if mf.size > 1 and mf.std() > 0 and mb.std() > 0 else None)
        res["tse_features"] = dict(n_paths_fwd=nf, n_paths_bwd=nb, profile_corr=r_prof,
                                   mean_shift_fwd_minus_bwd=float(np.mean(mf - mb)),
                                   frac_abs_z_gt2=float(np.mean(np.abs(z) > 2)),
                                   max_abs_z=float(np.max(np.abs(z))))
        if same and np.mean(np.abs(z) > 2) > 0.1:
            notes.append("forward and backward TSE differ at the same temperature: "
                         "violates microscopic reversibility (check sampling / definitions)")
        if not same:
            notes.append("different temperatures: TSE differences are expected "
                         "(Hammond shift toward the destabilised state)")
        feat_table = (mf, sf, mb, sb, z)
    else:
        feat_table = None
    return res, F, B, feat_table, (pf_plot, pb_plot), rng


def write_outputs(args, res, F, B, feat_table, pplot, rng):
    os.makedirs(args.out, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    with open(os.path.join(args.out, "joint_summary.json"), "w") as fh:
        json.dump(clean(res), fh, indent=2)

    for name, D in (("forward", F), ("backward", B)):
        with open(os.path.join(args.out, f"tse_{name}.csv"), "w") as fh:
            fh.write("trajectory,frame,time,weight\n")
            for i, fr, w in D["tse"]:
                tr = D["trajs"][i]
                fh.write(f"{tr['path']},{fr},{fr * tr['dt']:.6g},{w:.6g}\n")
    if feat_table is not None:
        mf, sf, mb, sb, z = feat_table
        with open(os.path.join(args.out, "tse_features.csv"), "w") as fh:
            fh.write("feature,mean_fwd,se_fwd,mean_bwd,se_bwd,z\n")
            for j in range(mf.size):
                fh.write(f"{j + 1},{mf[j]:.6g},{sf[j]:.6g},{mb[j]:.6g},{sb[j]:.6g},{z[j]:.4g}\n")

    # survival plots
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    x = np.linspace(0, 5, 200)
    for ax, D, pp, title in ((axs[0], F, pplot[0], "forward A→B"),
                             (axs[1], B, pplot[1], "backward B→A")):
        A = D["A"]
        w = np.full(A["k"].size, 1.0 / A["k"].size)
        ts, S, _ = mp.weighted_km(A["T"], A["done"], w)
        tau = np.sum(A["T"]) / np.sum(A["done"])
        ax.semilogy(x, np.exp(-x), "k--", lw=1, label="exponential")
        ax.step(np.r_[0, ts / tau], np.r_[1, S], where="post", label="original")
        if pp is not None and A["fail_pool"].size and A["succ_pool"].size:
            t = mp.stitch(pp, A["fail_pool"], A["succ_pool"], 20000, rng)
            tt = np.sort(t) / t.mean()
            ax.step(tt, 1 - np.arange(1, tt.size + 1) / tt.size, where="post",
                    label=f"stitched, p′={pp:.3g}")
        ax.set_xlim(0, 5); ax.set_ylim(1e-2, 1.05); ax.set_title(title)
        ax.set_xlabel("t / ⟨t⟩"); ax.legend(fontsize=8)
    axs[0].set_ylabel("survival S(t)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "survival_joint.png"), dpi=150)
    plt.close(fig)

    if F["tp_dur"].size and B["tp_dur"].size:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for D, lab in ((F, "forward"), (B, "backward")):
            d = np.sort(D["tp_dur"])
            ax.step(d, np.arange(1, d.size + 1) / d.size, where="post", label=lab)
        ax.set_xlabel("transition-path duration"); ax.set_ylabel("CDF")
        ax.set_title(f"TP time-reversal check, KS p = {res.get('tp_duration_ks_p', np.nan):.3f}")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "tp_durations.png"), dpi=150); plt.close(fig)

    if feat_table is not None:
        mf, sf, mb, sb, z = feat_table
        j = np.arange(1, mf.size + 1)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.errorbar(j, mf, yerr=sf, fmt="o-", ms=3, capsize=2, label="forward TSE")
        ax.errorbar(j, mb, yerr=sb, fmt="s--", ms=3, capsize=2, label="backward TSE")
        ax.set_xlabel("feature"); ax.set_ylabel("mean at TSE"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "tse_features.png"), dpi=150)
        plt.close(fig)


def _f(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def report(res):
    f, b = res["fwd"], res["bwd"]
    print(f"temperature relation: {res['temperature_relation']}   system: {res['system']}"
          f"   ({res['orientation']})")
    for name, d in (("forward ", f), ("backward", b)):
        print(f"{name}: N={d['N']} done={d['n_done']} p={d['p']:.3f} geom_p={d['geom_p']:.3f} "
              f"CV0={d['cv0']:.2f} KSp0={d['ks_p0']:.3f} "
              f"<TP>={_f(d['mean_tp_duration'])} <crossings>={_f(d['mean_tse_crossings'])}")
    if "tp_duration_ks_p" in res:
        print(f"TP durations fwd vs bwd: KS p = {res['tp_duration_ks_p']:.3f}")
    if res["temperature_relation"] == "same":
        print(f"kinetic DeltaG (box, kT): renewal {res['dG_kin_box_renewal']:.3f}, "
              f"raw {res['dG_kin_box_raw']:.3f}")
        if "dG_kin_standard_renewal" in res:
            print(f"kinetic DeltaG° (1 M, kT): renewal {res['dG_kin_standard_renewal']:.3f}")
        if "dG_target_box" in res:
            print(f"target DeltaG (box): {res['dG_target_box']:.3f}  "
                  f"-> kinetic minus target {res['dG_kin_minus_target']:+.3f} kT")
        print(f"joint tilt: lambda_f = {_f(res['lambda_joint_f'])}, "
              f"lambda_b = {_f(res['lambda_joint_b'])}")
    else:
        print(f"independent tilts: lambda_f = {_f(res['lambda_fwd_independent'])}, "
              f"lambda_b = {_f(res['lambda_bwd_independent'])}")
    if "tse_features" in res:
        t = res["tse_features"]
        print(f"TSE features: profile r = {_f(t['profile_corr'])}, "
              f"|z|>2 fraction = {t['frac_abs_z_gt2']:.2f}, "
              f"mean shift fwd-bwd = {t['mean_shift_fwd_minus_bwd']:+.4f}")
    for n in res["notes"]:
        print("NOTE: " + n)


def main(argv=None):
    args = parse_args(argv)
    res, F, B, feat_table, pplot, rng = analyse(args)
    write_outputs(args, res, F, B, feat_table, pplot, rng)
    report(res)
    print(f"\nWrote joint_summary.json, tse_forward.csv, tse_backward.csv and plots to {args.out}/")


if __name__ == "__main__":
    main()
