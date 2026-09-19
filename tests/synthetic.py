"""Hand-built first-passage trajectories with a controlled attempt structure.

Used to exercise the reweighting path end to end: the population is chosen so that the
weighted CV of folding times starts below 1 and crosses 1 at a finite lambda with a
healthy effective sample size (lambda* ≈ 1.1 kT, N_eff ≈ 115 for the defaults), which
does not happen for the Langevin regression data.

  n0 trajectories with no failed attempt,  folding time ≈ T0
  n3 trajectories with 3 failed attempts,  folding time ≈ TS (short) or TL (long, fraction f)

Up-weighting trajectories with more failures (the MaxCal tilt) shifts weight to the
over-dispersed k = 3 group, so the CV rises through 1.  The construction is a test of
the code path, not a physical model.
"""
import os
import numpy as np

Q_U, Q_EXC, Q_F = 0.1, 0.5, 0.9      # U basin, excursion level (fails), folded


def trajectory(k, T, b=2):
    """Q(t) folding exactly at frame T with k failed excursions to Q_EXC.
    Each failure is 2 frames at Q_EXC followed by b frames back in U."""
    a = T - 1 - k * (2 + b)
    if a < 1:
        raise ValueError("folding time too short for k failures")
    q = [Q_U] * a + ([Q_EXC, Q_EXC] + [Q_U] * b) * k + [Q_EXC, Q_F]
    return np.array(q)


def write_reweight_set(outdir, n0=200, n3=40, f=0.25, T0=50, TS=20, TL=200, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    nl = int(round(f * n3))
    specs = ([(0, T0 + int(rng.integers(-5, 6))) for _ in range(n0)]
             + [(3, TS + int(rng.integers(-3, 4))) for _ in range(n3 - nl)]
             + [(3, TL + int(rng.integers(-20, 21))) for _ in range(nl)])
    for i, (k, T) in enumerate(specs):
        q = trajectory(k, T)
        np.savetxt(os.path.join(outdir, f"s_{i:03d}.xvg"),
                   np.column_stack([np.arange(q.size, dtype=float), q]), fmt="%.6f")
    return specs


# --------------------------------------------------------------- rate-target dataset
# Twelve PDZ2-like systems: experimental kon/koff from Gianni et al. as tabulated in the
# multi-eGO PDZ2 paper, and model mean binding/unbinding times derived from that paper's
# Table 1 (kon in uM^-1 s^-1, koff in s^-1, model taus in ps).  The trajectories
# themselves are exponential samples with those means: enough to exercise the clock fit,
# the weights and their diagnostics reproducibly.
PDZ2_LIKE = {
    "EQVTAV_WT":   (2.6, 22.0, 391.0, 909_000.0),
    "EQVTAV_L18A": (2.4, 10.4, 520.0, 1_430_000.0),
    "EQVTAV_L25A": (1.12, 53.8, 533.0, 526_000.0),
    "EQVTAV_T35G": (2.7, 49.0, 592.0, 1_000_000.0),
    "EQVTAV_V44A": (2.7, 23.0, 532.0, 1_110_000.0),
    "EQVTAV_A53G": (2.4, 21.0, 490.0, 1_250_000.0),
    "EQVSAV_WT":   (2.9, 26.0, 525.0, 435_000.0),
    "EQVSAV_L18A": (2.2, 13.8, 560.0, 909_000.0),
    "EQVSAV_L25A": (1.69, 37.7, 612.0, 312_000.0),
    "EQVSAV_T35G": (3.1, 81.0, 446.0, 435_000.0),
    "EQVSAV_V44A": (3.3, 30.0, 595.0, 714_000.0),
    "EQVSAV_A53G": (2.78, 43.0, 637.0, 769_000.0),
}


def write_rate_dataset(outdir, n=50, seed=7):
    """Times files plus the systems.csv that maxcal-target reads."""
    import csv
    os.makedirs(os.path.join(outdir, "times"), exist_ok=True)
    rows = []
    for i, (name, (kon, koff, tb, tu)) in enumerate(sorted(PDZ2_LIKE.items())):
        for direction, tau in (("bind", tb), ("unbind", tu)):
            rng = np.random.default_rng([seed, i, 0 if direction == "bind" else 1])
            x = rng.exponential(tau, n)
            x *= tau / x.mean()                      # exact mean, as the paper's fits report
            np.savetxt(os.path.join(outdir, "times", f"{name}_{direction}.dat"), x, fmt="%.6f")
        rows.append(dict(system=name, kon_exp=kon, koff_exp=koff,
                         bind_times=f"times/{name}_bind.dat",
                         unbind_times=f"times/{name}_unbind.dat"))
    path = os.path.join(outdir, "systems.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader(); wr.writerows(rows)
    return path
