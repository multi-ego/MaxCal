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
