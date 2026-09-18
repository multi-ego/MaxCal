"""Overdamped Langevin folding trajectories on a 1D double well in Q (units of kT).

F(Q) = B * [(Q-a)(Q-b) / ((m-a)(m-b))]^2, minima at a (U) and b (F), barrier B at m.
Trajectories start at q0 (below the U minimum, giving an initial relaxation lag) and
stop as soon as Q >= q_stop, mimicking first-passage runs stopped at folding.
Each trajectory uses its own seeded generator, so the data set is reproducible.
"""
import os
import numpy as np


def grad_F(q, B, a, b, m):
    K = (m - a) * (m - b)
    g = (q - a) * (q - b)
    return 2.0 * B * g * (2.0 * q - a - b) / K ** 2


def run_one(rng, B=0.8, a=0.2, b=0.9, m=0.55, D=0.05, dt=0.01, q0=0.02,
            q_stop=0.85, stride=5, max_steps=500_000):
    q, Q = q0, [q0]
    noise = np.sqrt(2.0 * D * dt)
    for step in range(1, max_steps + 1):
        q = abs(q - D * grad_F(q, B, a, b, m) * dt + noise * rng.normal())  # reflect at 0
        if q >= q_stop:
            Q.append(q)                       # always keep the folding frame
            break
        if step % stride == 0:
            Q.append(q)
    return np.array(Q), dt * stride


def write_dataset(outdir, n=100, seed=2024, **kw):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i in range(n):
        Q, dts = run_one(np.random.default_rng([seed, i]), **kw)
        t = np.arange(Q.size) * dts
        p = os.path.join(outdir, f"q_{i:03d}.xvg")
        with open(p, "w") as fh:
            fh.write("# Langevin test trajectory\n@ title \"Q(t)\"\n")
            np.savetxt(fh, np.column_stack([t, Q]), fmt="%.10f")
        paths.append(p)
    return paths


def grad_bump(q, dB, m, sigma):
    """Gradient of a Gaussian bump dB*exp(-(q-m)^2/2sigma^2) that raises only the barrier."""
    return -dB * (q - m) / sigma ** 2 * np.exp(-(q - m) ** 2 / (2 * sigma ** 2))


def run_many(n, seed, B=0.8, dB=0.0, sigma=0.04, a=0.2, b=0.9, m=0.55, D=0.05, dt=0.01,
             q0=0.2, q_stop=0.85, stride=5, max_steps=2_000_000, tilt=0.0, Tr=1.0,
             stop_below=None, n_feat=0):
    """Vectorized version of run_one for n trajectories at once, with an optional
    Gaussian bump of height dB on the barrier top (basins essentially unchanged).
    Returns a list of Q(t) arrays (strided, last frame = folding frame) and the frame dt.

    tilt : linear term tilt*(q-m) added to F (in kT at the reference temperature),
           making the two basins unequal (DeltaG != 0).
    Tr   : temperature relative to the reference; overdamped dynamics with D ∝ T, so the
           drift -(D/kT) dU/dq is T-independent and only the noise grows as sqrt(Tr).
    stop_below : if given, trajectories stop when Q <= stop_below (unfolding/unbinding
           runs started in the high-Q basin) instead of Q >= q_stop.
    n_feat : if > 0, also return per-frame synthetic "structural features"
           f_j = Q**(j+1) (deterministic functions of Q, identical in both directions)."""
    rng = np.random.default_rng(seed)
    q = np.full(n, q0, float)
    active = np.ones(n, bool)
    rows = [q.copy()]
    fold_step = np.full(n, -1)
    fold_q = np.zeros(n)
    noise = np.sqrt(2.0 * D * dt)
    for step in range(1, max_steps + 1):
        x = q[active]
        x = np.abs(x - D * (grad_F(x, B, a, b, m) + grad_bump(x, dB, m, sigma) + tilt) * dt
                   + noise * np.sqrt(Tr) * rng.normal(size=x.size))
        q[active] = x
        f = active & ((q <= stop_below) if stop_below is not None else (q >= q_stop))
        fold_step[f] = step
        fold_q[f] = q[f]
        active &= ~f
        if step % stride == 0:
            rows.append(np.where(active, q, np.nan))
        if not active.any():
            break
    frames = np.array(rows)
    out = []
    for i in range(n):
        s = fold_step[i]
        if s < 0:                       # never folded: censored
            col = frames[:, i]
            out.append(col[~np.isnan(col)])
        else:
            out.append(np.r_[frames[: (s - 1) // stride + 1, i], fold_q[i]])
    if n_feat:
        feats = [np.column_stack([x ** (j + 1) for j in range(n_feat)]) for x in out]
        return out, dt * stride, feats
    return out, dt * stride


def F_total(q, B=0.8, dB=0.0, sigma=0.04, a=0.2, b=0.9, m=0.55, tilt=0.0):
    K = (m - a) * (m - b)
    return (B * ((q - a) * (q - b) / K) ** 2 + dB * np.exp(-(q - m) ** 2 / (2 * sigma ** 2))
            + tilt * (q - m))


def delta_G_eq(qa, qb, Tr=1.0, **pot):
    """G_B - G_A (kT at temperature Tr) for the core sets A: q < qa and B: q >= qb.
    Populations are committor-split (transition path theory), pi_A = <1 - q+>,
    pi_B = <q+>, which is what the ratio of core-to-core rates measures,
    k_AB / k_BA = pi_B / pi_A.  In 1D the committor is exact:
    q+(x) = int_qa^x e^{F/kT} / int_qa^qb e^{F/kT} between the cores.
    The reflecting wall at 0 is handled by the mirror image."""
    q = np.linspace(0.0, 2.0, 400_001)
    F = F_total(q, **pot) / Tr
    rho = np.exp(-F) + np.exp(-F_total(-q, **pot) / Tr)
    rho[0] *= 0.5
    inside = (q >= qa) & (q < qb)
    e = np.where(inside, np.exp(F - F[inside].min()), 0.0)
    qplus = np.clip(np.cumsum(e) / e.sum(), 0, 1)
    qplus[q < qa] = 0.0
    qplus[q >= qb] = 1.0
    return -np.log(np.sum(rho * qplus) / np.sum(rho * (1 - qplus)))


def write_set(outdir, Qs, dt, feats=None, prefix="q"):
    os.makedirs(outdir, exist_ok=True)
    for i, Q in enumerate(Qs):
        cols = [np.arange(Q.size) * dt, Q] + ([feats[i]] if feats is not None else [])
        np.savetxt(os.path.join(outdir, f"{prefix}_{i:03d}.xvg"), np.column_stack(cols),
                   fmt="%.10f", header="Langevin test trajectory", comments="# ")
