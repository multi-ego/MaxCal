# maxcal

Maximum-caliber (MaxCal) correction of an underestimated folding barrier in
first-passage trajectories, with the aim of recovering two-state (Poisson) kinetics
and characterising the transition-state ensemble (TSE).

The intended use case is a coarse-grained model such as multi-eGO [1] that samples
the relevant structures correctly but places too low a barrier between the unfolded
(U) and folded (F) states. Absolute rates in such models are not meaningful because
the clock is rescaled, so the analysis focuses on the *shape* of the folding-time
distribution and on dimensionless quantities.

---

## 1. Setting

**Input.** N trajectories Q(t), all started in U and stopped as soon as Q ≥ Q_f
(first-passage ensemble). Trajectories that never reach Q_f are right-censored.

**Basins and attempts.** U is defined by Q < Q_u. An *excursion* starts when a
trajectory leaves U and ends either back in U (failure) or at Q_f (success). Given an
interface Q‡ with Q_u < Q‡ < Q_f, an *attempt* is an excursion whose maximum Q
exceeds Q‡.

This is the decomposition used in transition interface sampling and forward-flux
sampling [2, 3]:

$$
k_{UF} = \Phi_{U,Q^\ddagger} \cdot P(Q_f \mid Q^\ddagger),
$$

the flux of attempts through Q‡ times the probability that an attempt reaches F
before returning to U. Per trajectory we record the number of failed attempts k_i
and the folding time T_i. The model success probability per attempt is estimated as

$$
\hat p = \frac{N_\text{done}}{N_\text{done} + \sum_i k_i}.
$$

---

## 2. The MaxCal tilt

### 2.1 Assumptions

**(A1) Markovian at the interface.** Attempt outcomes are independent Bernoulli(p)
variables. The reference path measure factorises as

$$
P(\omega) = P(\omega \mid Y) P(Y),\qquad P(Y)=\prod_j p^{s_j}(1-p)^{1-s_j},
$$

where Y = (s_1, s_2, …) is the sequence of attempt outcomes (s_j = 1 for success).

**(A2) Correct sampling, wrong barrier.** The true dynamics differ from the model
only in the probability that an attempt succeeds, not in the paths themselves once
the outcomes are fixed.

### 2.2 Derivation

Maximum caliber [4–6] selects, among all path measures that satisfy a set of
constraints, the one closest to a reference measure P. "Closest" means minimum
relative path entropy (Kullback–Leibler divergence):

$$
\min_{P'} \sum_\omega P'(\omega)\ln\frac{P'(\omega)}{P(\omega)}
\quad\text{s.t.}\quad \langle g(Y)\rangle_{P'} = G, \sum_\omega P'(\omega)=1 .
$$

Take as constraint the mean number of successful crossings in a fixed number of
attempts, g(Y) = Σ_j s_j. This fixes the crossing flux, i.e. the barrier. The
Lagrangian solution is the exponential tilt

$$
P'(\omega) = \frac{1}{Z} P(\omega) e^{-\lambda g(Y)} .
$$

Two consequences follow directly.

1. **Conditionals are untouched.** Because the tilt depends on ω only through Y,
   P'(ω | Y) = P(ω | Y). Given the outcome of each attempt, the dwell times, excursion
   shapes and structures are exactly those of the model. This is assumption (A2)
   derived as the least-biased choice rather than postulated.

2. **The outcome process stays Bernoulli, with shifted log-odds.** For an i.i.d.
   Bernoulli reference, the tilt e^{-λ Σ s_j} factorises over attempts. Normalising
   each factor gives an i.i.d. Bernoulli(p′) process with

$$
\mathrm{logit} p' = \mathrm{logit} p - \lambda,
\qquad
p' = \frac{p e^{-\lambda}}{1-p (1-e^{-\lambda})}.
$$

   The Lagrange multiplier λ is therefore the shift in log-odds per attempt.

**Detailed balance and ΔG.** Counting crossings in both directions tilts U→F and F→U
attempts by the same factor. In the rare-event limit both rates scale by the same
amount, so their ratio, and hence ΔG, is unchanged.

### 2.3 Weights for first-passage trajectories

A completed trajectory with k failures followed by one success has reference
probability (1−p)^k p. Under the tilt it has (1−p′)^k p′. The likelihood ratio is
therefore

$$
w_\text{done}(k) \propto \Big(\tfrac{1-p'}{1-p}\Big)^{k} \frac{p'}{p}
 = c^{k+1} e^{-\lambda},
$$

and for a censored trajectory with k failures and no success

$$
w_\text{cens}(k) \propto c^{k},
\qquad
c = \frac{1}{1-p (1-e^{-\lambda})}.
$$

These are the weights computed by `log_weights`. The unit test
`test_maxcal_identity_geometric_to_geometric` checks that tilting an exactly
geometric(p) population with these weights gives exactly geometric(p′).

### 2.4 Limitation of pure reweighting

As λ → ∞, c → 1/(1−p): the weights saturate at (1−p)^{−k}. Reweighting can therefore
only redistribute probability over the values of k that were actually observed.

A higher barrier implies a geometric distribution of k with mean (1−p′)/p′ ≫ 1. If
the data contain only k ≤ k_max, that tail is simply absent. Upweighting the
trajectories with the most failures then makes the distribution *more regular*,
because it favours sums of more cycles. The coefficient of variation (CV) of the
reweighted folding times decreases instead of approaching 1. This is observed on the
regression-test data: the CV falls from 0.86 at λ = 0 to about 0.5 at large λ.

Pure reweighting is therefore reliable only when the trajectories already contain
many failed attempts. The script reports the Kish effective sample size [7]

$$
N_\text{eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}
$$

and flags the result when N_eff falls below a threshold (default 30).

### 2.5 Stitching: sampling the tilted measure generatively

Under (A1) and the conditional invariance of §2.2, a tilted first-passage time can be
sampled directly. Draw

$$
K \sim \text{Geom}(p') (\text{failures}),\qquad
T' = \sum_{j=1}^{K} C_j + S,
$$

where each C_j is resampled from the pool of observed cycles ("return to U → end of
the next failed attempt") and S from the pool of successful final segments. By
default, the first segment of each trajectory is excluded from both pools, because
it contains the relaxation from the starting structures.

The moments of this compound geometric sum are

$$
\langle T'\rangle = \frac{1-p'}{p'} \mu_C + \mu_S,
\qquad
\mathrm{Var} T' = \frac{1-p'}{p'} \sigma_C^2 + \frac{1-p'}{p'^2} \mu_C^2 + \sigma_S^2 .
$$

As p′ → 0 the second term dominates, CV → 1, and p′T′/μ_C converges in distribution
to Exp(1). This is Rényi's theorem on geometric sums [8, 9]: thinning a renewal
process of attempts produces a Poisson process of successes.

This is the mechanism by which the barrier correction restores two-state kinetics.
It also shows that recovering Poisson statistics does not by itself validate the
model, because any process with independent attempts becomes Poisson under strong
enough thinning.

### 2.6 What the Poisson criterion does and does not determine

Because CV → 1 monotonically as p′ → 0, the smallest λ at which the stitched times
pass a test for exponentiality is only a **lower bound** on the true tilt. The
validation test confirms this. The blind threshold found from model data alone is
0–0.35, against true values of 0.9–1.7.

Evidence for two-state behaviour must come from quantities that thinning cannot
manufacture:

- **Attempt independence.** Test whether the observed k_i follow a geometric
  distribution (χ² test) and whether successive cycle durations are uncorrelated
  (lag-1 Spearman correlation).
- **Timescale separation.** Compare the relaxation time within U with the cycle time.
- **Committor.** Check that committor values on the transition-state surface are
  peaked around 1/2 [10, 11].

### 2.7 Interpretation of λ

λ is an **effective log-odds shift per attempt**, not the missing barrier height. In
the high-barrier (Kramers) regime [12, 13], P(Q_f | Q‡) ∝ e^{−βΔF‡} up to prefactors,
so λ → βΔΔF‡. At finite barriers λ is smaller. In the validation, Gaussian bumps of
2 kT and 3 kT gave λ ≈ 0.9–1.1 and λ ≈ 1.7–1.9, respectively.

### 2.8 Where to put Q‡

The attempt interface must sit at the **foot of the missing barrier, on the U
side**. Only then is the attempt flux Φ_{U,Q‡} unaffected by the correction, so that
the whole effect falls on P(Q_f | Q‡), as the tilt assumes. This is the same
requirement as for the first interface in TIS and FFS [2, 3].

If Q‡ is placed on the barrier top, the missing barrier also changes how often Q‡ is
reached. In the validation, stitching then underestimates the true mean folding time
by a factor of 3–5.

**The transition state lies between Q‡ and Q_f.** The tilt acts only on what happens
after an attempt passes Q‡, so the missing barrier, and with it the corrected TS, must
lie beyond Q‡. Q‡ is therefore a lower bound on the TS location. Conversely, a Q‡
placed at or past the barrier is invalid.

**Decision map.** `heatmap.png` shows the median stitched KS p-value on a grid of Q‡
(the scan values) and λ:
- **Hatched cells** pass the Poisson criterion (KS p ≥ α and |CV − 1| ≤ tol).
- **Triangles** mark λ_min(Q‡).
- **The shaded region** covers Q‡ at or beyond the assumed barrier location (from
  `--barrier-q` and the model committor, §2.10), where the analysis does not apply.

A flat λ_min over the valid Q‡ values means the foot interface is well placed and
λ_min is robust. The shaded boundary is only an upper limit: validity ends where the
missing barrier *starts*, not at its centre. In the validation (a bump centred at
0.55), the true λ already falls once Q‡ exceeds about 0.45, and λ_min rises there.
With real data the observable signal is a rise of λ_min above its low-Q‡ level: use
the flat region before it. The map says nothing about *where* the TS sits beyond Q‡: folding
times are insensitive to the barrier location. For the same 3 kT missing barrier
placed at Q = 0.46, 0.55 or 0.64, the TS moves by 0.17 in Q, but the folding-time
shapes cannot be told apart (two-sample KS p ≥ 0.25).

The TSE should nevertheless be taken at the barrier top. Use `--qtse` to set a
separate surface for the transition-state frames (last upward crossing before
folding), or, better, locate it from the committor as described in §2.10.

### 2.9 Relation to time reweighting

Read as one long trajectory, the tilted ensemble is equivalent to stretching time in
the basins relative to the barrier region by roughly 1/p′. This is the analogue of
the acceleration factor in hyperdynamics [14] and infrequent metadynamics [15], with
the missing barrier playing the role of the (negative) bias. The KS-based validation
of exponential statistics follows Salvalaglio *et al.* [16].


### 2.10 Committor and transition-state location

The tilt determines *how much* barrier is missing, through r = p′/p. It does **not**
determine *where* the missing barrier sits between Q‡ and Q_f, and the location of the
transition state depends on it. The location is therefore an extra assumption, which
the script takes as an input: `--barrier-q`, expressed as a value q* of the **model**
committor.

**Model committor.** Every frame inside an excursion out of U is labelled with the
outcome of its excursion: 1 if the excursion reaches Q_f before returning below Q_u,
0 otherwise. For a frame x, this label is a sample of the committor q(x). Binning
along Q and imposing monotonicity (pool-adjacent-violators) gives the model committor
q_m(Q), written to `committor.csv`.

**Corrected committor for a located barrier.** Suppose the missing barrier is narrow
and lies on a model isocommittor surface q_m = q*. For diffusive dynamics the committor
is harmonic away from the barrier, so a narrow barrier only adds a jump across that
surface. With r = p′/p fixed by the attempt statistics at Q‡, the corrected committor
is

$$
q'(x) = r q_m(x) \quad (q_m < q^*), \qquad
1 - q'(x) = r \left(1 - q_m(x)\right) \quad (q_m > q^*).
$$

In words:
- **On the U side of the barrier**, the probability of reaching F drops by r.
- **On the F side**, the probability of *returning* to U drops by r. The barrier
  blocks recrossing in both directions.

For the Langevin test potential with a Gaussian bump as the missing barrier, this is
exact to about 10⁻⁴ away from the bump (`tests/test_committor.py`).

**TS location.** The corrected transition state is the surface where q′ = 1/2:
- **At the barrier** (q_m = q*) whenever the jump straddles 1/2, i.e. when
  r·q* ≤ 1/2 ≤ 1 − r(1 − q*). For a barrier on the model's own TS (q* = 1/2) this
  holds for every λ.
- **Otherwise** (weak tilt, barrier far from the model TS), on the model surface
  q_m = 1/(2r) on the U side, or q_m = 1 − 1/(2r) on the F side.

**Consequences.**
- **Default assumption.** With `--barrier-q 0.5` (the default), the missing barrier
  sits on the model's own transition state. Then the TS location, and the TSE
  composition at it, are independent of λ. This is the precise form of the statement
  "correct structures, underestimated barrier."
- **Other locations.** For q* ≠ 1/2 the TS moves toward the barrier as λ grows, and
  reaches it once the tilt is strong enough. `ts_location.csv` and `ts_location.png`
  scan q* from 0.1 to 0.9 for λ_min, λ_min+1 and λ_min+2 (or `--tse-lams`), so you
  can see how much your TSE depends on this assumption. Values of q* before the
  attempt interface are flagged, because the tilt assumes the barrier lies beyond Q‡.
- **Bounds.** The corrected TS always lies between Q‡ and Q_f (§2.8). `ts_location.csv`
  flags barrier locations before the interface (`barrier_before_interface`) and TS
  positions that would fall at or before it (`ts_before_interface`).
- **Fixing q* from experiment.** Φ-values measure the TS structure, so q* can be
  chosen as the model isocommittor surface whose structures best match them.
- **A naive shortcut fails.** Shifting the model committor by λ in log-odds,
  logit q′ = logit q_m − λ, implicitly puts the whole missing barrier at Q_f. On the
  test potential it misplaces the TS by more than 0.1 in Q; a test documents this.

**Outputs:**
- `tse_committor_lam{λ}_q{q*}.csv`: frames within one bin of the corrected TS for each
  λ, with the outcome of their excursion. Use them to compute structural observables
  from your structures.
- `committor.csv`: the model committor along Q (raw and monotone).
- `ts_location.csv`: TS location for each (λ, q*), and which case sets it.

---

## 3. Statistics used

- **Survival function.** Weighted Kaplan–Meier estimator [17], handling censored
  trajectories. Weights act as frequency weights: a weight of 2 is equivalent to
  duplicating a trajectory, which is unit-tested.
- **Exponential fit.** Maximum-likelihood time constant with censoring,
  τ̂ = Σ w_i T_i / Σ w_i δ_i, where δ_i = 1 for completed trajectories.
- **Exponentiality test.** Kolmogorov–Smirnov statistic against an exponential with
  a fitted scale. Because the scale is estimated, standard KS p-values are
  anti-conservative. The null distribution is simulated once per sample size
  (Lilliefors [18]); it is scale-free.
- **Uncertainty.** Nonparametric bootstrap over trajectories [19] gives confidence
  intervals on λ*.
- **Stitching test power.** The stitched sample is subsampled to the original N
  before testing, so the test has the same power as on the original data.

---

## 4. Installation and usage

```bash
git clone git@github.com:multi-ego/MaxCal.git
cd MaxCal
pip install -e ".[test]"
pytest -q                     # fast tests; add --runslow for the validation tests
```

The code is the package `maxcal`; one executable per step, all sharing `maxcal/core.py`
(trajectory parsing, attempt counting, the tilt, survival/KS statistics, status codes):

| command | module | what it does |
|---|---|---|
| `maxcal-stitch` | `maxcal/stitching.py` | thinning barrier crossings: λ_min, attempt statistics, the (Q‡, λ) decision map (§2.5–2.8) |
| `maxcal-reweight` | `maxcal/reweight.py` | MaxCal weights on whole trajectories: λ*, N_eff, per-trajectory weights (§2.3–2.4) |
| `maxcal-committor` | `maxcal/committor.py` | model and corrected committor, TS location, committor-based TSE frames (§2.10) |
| `maxcal-joint` | `maxcal/joint.py` | forward + backward runs: kinetic ΔG, transition-path symmetry, coupled tilt (§6) |
| `maxcal-target` | `maxcal/target.py` | reweighting onto experimental rates with a Poisson target (§7) |

Each is also runnable as `python -m maxcal.stitching ...`, and `import maxcal as m`
re-exports the shared functions (`m.attempts`, `m.stitch`, `m.cv_curve`, `m.ts_location`).

**Demo notebooks.** `notebooks/demo.ipynb` walks through the method interactively,
using the test datasets:
- attempt counting as Q‡ moves;
- the reweighting weights, their saturation, and the resulting CV and N_eff;
- stitching and the emergence of Poisson statistics;
- validation against a simulated higher barrier, including what goes wrong with Q‡
  on the barrier top;
- the joint-mode checks (kinetic ΔG, transition-path symmetry across temperatures).

```bash
pip install -e ".[demo]"
jupyter lab notebooks/demo.ipynb
```

`notebooks/target_demo.ipynb` does the same for `maxcal-target` (§7) on the twelve-system
PDZ2-like dataset of its regression test: the clock calibration and what is left per
system, the weights and their CDFs, the N_eff cost of the correction, how weights change a
structural observable, and the affinity scatter before and after.

`examples/walkthrough_ks_lambda.py` is a short script version of the key steps: raw
folding times fail the KS test, reweighting and stitching are compared as λ grows, and
λ_min is compared with the true λ of a simulated higher barrier.

The committed copy is rendered with static figures, so it can be read on GitHub.
The sliders appear when you run it live with `ipywidgets` installed. To refresh the
committed outputs:
`MAXCAL_DEMO_STATIC=1 jupyter nbconvert --to notebook --execute --inplace notebooks/demo.ipynb`.

```bash
maxcal-stitch    "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 --out stitch_out
maxcal-reweight  "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 --qtse 0.55 --out reweight_out
maxcal-committor "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 \
                 --from-stitch stitch_out/summary.csv --out committor_out
```

Input files contain either one column (Q; give `--dt`) or two columns (time, Q).
Lines starting with `#` or `@` are ignored, so GROMACS `.xvg` files work directly.

| option | meaning |
|---|---|
| `--qu`, `--qf` | U basin (Q < Qu); folded state (Q ≥ Qf), where trajectories stop |
| `--qts` | reference attempt interface, at the barrier foot (default: middle of the scan grid) |
| `--qtse` | *(reweight)* surface for TSE frames (default: same as each Q‡) |
| `--barrier-q` | *(committor, stitch)* assumed location of the missing barrier, as a model committor value q* (default 0.5, the model TS; §2.10) |
| `--lams`, `--from-stitch` | *(committor)* λ values for the TSE, or λ_min (+1, +2) taken from a `maxcal-stitch` summary |
| `--committor-bins`, `--tse-window` | *(committor)* binning along Q; half-width of the TSE window |
| `--nstitch`, `--nsub`, `--lam-max`, `--nlam` | *(stitch)* synthetic times per λ, KS subsamples, λ grid |
| `--boot`, `--neff-min` | *(reweight)* bootstrap resamples for λ*, N_eff threshold |
| `--heatmap-nlam`, `--heatmap-lam-max` | *(stitch)* λ grid of the decision map (default 21 values in [0, 5] kT; 0 disables) |
| `--nqts` | number of Q‡ values scanned for robustness |
| `--t0` | discard an initial relaxation window |
| `--include-initial` | keep the first segment of each trajectory in the stitching pools |
| `--alpha`, `--cv-tol` | Poisson criterion for stitching: median KS p ≥ α and \|CV − 1\| ≤ tol |
| `--neff-min` | N_eff threshold below which reweighting is flagged |

**Outputs** (each tool writes into its own `--out` directory)

- **stitch**: `summary.csv` (one row per Q‡: attempt statistics, independence tests, CV
  and KS at λ = 0, `cv_stitch0`/`ksp_stitch0`, λ_min, status codes), `heatmap.csv`/`.png`
  (the decision map, §2.8), `survival.png`, `robustness_qts.png`.
- **reweight**: `summary.csv` (λ* with bootstrap interval, N_eff, KS at λ*, `cv_lammax`,
  status codes), `weights_qts*.csv` (one weight per trajectory), `tse_frames_qts*.csv`,
  `survival.png`, `lambda_scan.png`.
- **committor**: `committor.csv`, `ts_location.csv`/`.png`, `tse_committor_*.csv` (§2.10).

**Status columns.** Every NaN in `summary.csv` comes with a reason code, so it can be
told apart from a failure of the code.

| column | values |
|---|---|
| `geom_status` | `ok`; `too_few_trajectories` (fewer than 10 completed); `too_few_bins` (too few distinct k for the χ² test) |
| `lag1_status` | `ok`; `too_few_pairs` (fewer than 10 consecutive failure cycles); `constant_cycles` (correlation undefined) |
| `reweight_status` | `ok`; `no_root` (weighted CV never reaches 1, §2.4); `low_neff` (N_eff below `--neff-min`); `cv_ge_1_at_lambda0` (already over-dispersed: a missing barrier does not explain the deviation) |
| `stitch_status` | `ok`; `ok_pool_fallback` (initial segments had to be used); `no_pass` (Poisson criterion never met in the scan); `empty_pool` (no failed attempts past this Q‡, typically at or beyond the barrier) |

**Reading the results**

1. If `geom_p` < α, attempts have memory. Poisson statistics after tilting are then
   imposed, not recovered.
2. If stitching at λ = 0 already passes, the original deviation from Poisson was the
   initial relaxation, not the barrier.
3. If CV ≥ 1 at λ = 0, a missing barrier does not explain the deviation; look for
   intermediates or parallel pathways.
4. Only Q‡ values on the U side of the barrier region are meaningful (§2.8).

---

## 5. Tests

```bash
pytest -q tests/                  # 67 fast tests (~15 s)
pytest -q tests/ --runslow        # + 15 validation tests and the demo notebook (~32 s)
MAXCAL_UPDATE_REF=1 pytest tests/test_regtest.py   # regenerate the reference
```

- **`test_unit.py`**: parsing, attempt counting, tilt identities (§2.2–2.4),
  Kaplan–Meier, KS/Lilliefors size and power, geometric test, stitching moments
  (§2.5), and I/O.
- **`test_regtest.py`**: runs `maxcal-stitch`, `maxcal-reweight` and `maxcal-committor`
  on 100 seeded overdamped Langevin trajectories and compares the two summaries with
  `tests/regtest/reference_stitching.csv` and `reference_reweight.csv`. Deterministic
  columns use rtol 1e-5; RNG-dependent stitching columns use loose absolute tolerances.
- **`test_reweight_path.py`**: end-to-end test of the path where reweighting
  succeeds, which the Langevin data never reach. It uses hand-built trajectories
  (`tests/synthetic.py`) in which the weighted CV crosses 1 at λ* ≈ 1.1 with
  N_eff ≈ 115. It checks that:
  - λ* is finite and bracketed by its bootstrap confidence interval;
  - N_eff and the KS p-value at λ* are finite;
  - the weighted CV equals 1 at λ*;
  - the TSE weight ratio between trajectories with 3 and 0 failures is exactly c³;
  - every NaN has the right reason code.
- The regression test also checks that every NaN in the summary has a non-`ok`
  reason code, and it compares the reason codes with the reference.
- **`test_committor.py`**: on exact 1D committors of the test potential, the
  located-barrier formula matches the truth to 10⁻³ away from the bump and puts the
  TS at the bump; the naive log-odds shift misplaces it. The slow variant estimates
  committors from simulated trajectories and checks that the predicted TS matches the
  truth within 0.03 in Q, while a wrongly assumed barrier location does not.
- **`test_validation.py`**: low-barrier and high-barrier Langevin runs; the high
  barrier is the low one plus a Gaussian bump of 2 or 3 kT on the barrier top. It
  checks four things.
  - The high-barrier ground truth is Poissonian.
  - Stitching the low-barrier cycles with the true p′ reproduces the high-barrier
    folding-time distribution: mean within 15% (1–7% observed across seeds), and a
    two-sample KS test passes.
  - The blind Poisson threshold is a lower bound on the true λ.
  - The construction fails, as it should, with Q‡ on the barrier top (§2.8).

---

## 6. Joint mode: forward and backward trajectories (`maxcal-joint`)

This mode analyses first-passage runs in both directions together: A→B (folding or
binding) and B→A (unfolding or unbinding). You must state explicitly whether the
two sets were simulated at the same temperature, because that decides which
relations between the two directions hold.

### 6.1 Geometry

The script uses one collective variable (CV) with basins A and B.
- **If `--qa < --qb`:** A is Q < qa and B is Q ≥ qb, as for native or intermolecular
  contacts.
- **If `--qa > --qb`:** the orientation is reversed automatically, as for a
  distance: A (unbound) is d > qa, B (bound) is d ≤ qb.

All thresholds are given in the original units. Backward runs are analysed in the
mirrored coordinate −Q, so all of §2 applies unchanged. Each direction has its own
attempt interface, at the foot of the barrier on its own side (§2.8):
`--qts-fwd` on the A side and `--qts-bwd` on the B side. The TSE surface `--qtse`
sits in between, at the barrier top.

### 6.2 Same temperature: one barrier correction for both directions

**Detailed balance.** At equilibrium, the rates between core sets satisfy [23, 24]

$$
\frac{k_{AB}}{k_{BA}} = \frac{\pi_B}{\pi_A} = e^{-\beta\Delta G_{AB}},
$$

where π_A and π_B are the committor-split populations. The script estimates each
rate as the inverse of the renewal mean first-passage time of §2.5, k = 1/M, with

$$
M(p) = \mu_C \frac{1-p}{p} + \mu_S .
$$

This excludes the initial relaxation. The rates are ratios on the same clock, so the
scaled time of the model cancels.

**Kinetic ΔG.** The quantity

$$
\Delta G_\text{kin} = -kT\ln\left(\frac{M_{BA}}{M_{AB}}\right)
$$

must equal the equilibrium ΔG for two-state behaviour [25]. It is reported
(`dG_kin_box_renewal`) and compared with `--dG` when given. Because thinning cannot
manufacture this agreement, it is the most informative two-state test here.

**Coupled tilt.** MaxCal on the pair of attempt processes has two constraints: the
crossing flux (the barrier) and the stationary ratio (ΔG). The second constraint
removes one of the two multipliers. The script scans the forward tilt λ_f
(p′_f = tilted p_f) and fixes the backward success probability through

$$
M'_{BA} = R M'_{AB},\qquad
p'_b = \frac{\mu_{C,B}}{R M'_{AB} - \mu_{S,B} + \mu_{C,B}},
$$

where R = exp(−βΔG_target). By default ΔG_target is the model's own kinetic value,
which keeps its ΔG. If you give `--dG`, the experimental value is imposed instead.

In the rare-event limit M ≈ μ_C/p, and the constraint reduces to
p′_f/p_f = p′_b/p_b, i.e. the same barrier increase seen from both sides. The
reported joint λ is the first λ_f at which **both** stitched directions pass the
Poisson criterion. Like λ in §2.6, it is a lower bound. If the model's kinetic ΔG
differs from `--dG`, the two directions are tilted asymmetrically; a negative
backward tilt is flagged.

**Transition-path time-reversal symmetry.** At equilibrium, the B→A transition-path
ensemble is the time reverse of the A→B one [11, 26]. Transition paths run from the
last frame in the starting basin to the first frame in the target. Two consequences
can be tested using the CV alone:
- the distributions of transition-path durations must be identical (two-sample KS
  test);
- the numbers of Q_tse recrossings per path must be identical too.

Under the MaxCal tilt, transition paths are unchanged (§2.2, conditional
invariance). These tests therefore do not depend on λ and need no reweighting.

**TSE comparison.** TSE frames are **all** crossings of `--qtse` inside each
transition path. For each crossing the script takes whichever of the two bracketing
frames is closer to Q_tse, so the rule is symmetric under time reversal, and each
crossing is weighted by 1/(number of crossings) so every path counts once. If the
input files contain extra columns, e.g. per-residue native-contact fractions, the
mean TSE feature profiles of the two directions are compared, with bootstrap
standard errors over paths. At the same temperature they must agree, by
microscopic reversibility.

### 6.3 Different temperatures

λ is not transferable between temperatures, and ΔG changes with T. In a model with
an effective potential such as multi-eGO, the temperature dependence is not
calibrated anyway. The two directions are therefore analysed **independently**:
- no kinetic ΔG is reported;
- `--dG` is ignored;
- there is no detailed-balance coupling.

TSE differences are expected. Destabilising one state shifts the TSE toward it
(Hammond/Leffler [27, 28]), so a shift is consistent with one robust pathway, while a
completely different set of contacts is not. Rigorous reweighting of paths between
temperatures would need Girsanov path likelihood ratios [29], which require the
random forces of every integration step and are therefore not applicable after the
fact.

**Declaring "same" for data at different temperatures is detected.** The
transition-path durations then differ, and the script says so. In the validation
data, a 1.5× temperature gives KS p < 10⁻³.

### 6.4 Binding

For one ligand and one receptor in a box of volume V (`--box-volume`, in nm³), the
box equilibrium constant is K_box = P_bound/P_unbound = k′_on/k_off, where k′_on is
the pseudo-first-order rate at [L] = 1/V. The standard binding free energy is [30]

$$
\Delta G^\circ = \Delta G_\text{box} - kT\ln\left(V C^\circ\right),
\qquad C^\circ = 1\ \text{M} = 0.6022\ \text{nm}^{-3}.
$$

`--dG` is interpreted as ΔG° for `--system binding`, and the kinetic ΔG° is
reported. The unbound basin must be defined consistently with this volume: the
ligand should be free anywhere beyond the A threshold. On the attempt interface for
binding, the foot of the barrier is typically the encounter complex, and the
missing barrier may be desolvation or induced fit.

### 6.5 Usage

```bash
# folding / unfolding at the same temperature, experimental DeltaG imposed
maxcal-joint --fwd "fold/*.xvg" --bwd "unfold/*.xvg" \
    --qa 0.3 --qb 0.8 --qts-fwd 0.4 --qts-bwd 0.7 --qtse 0.55 \
    --temperature-relation same --dG -2.5 --out joint

# unfolding simulated at a higher temperature
maxcal-joint --fwd "fold/*.xvg" --bwd "unfold_hot/*.xvg" \
    --qa 0.3 --qb 0.8 --qts-fwd 0.4 --qts-bwd 0.7 --qtse 0.55 \
    --temperature-relation different --out joint_hot

# binding / unbinding on a distance CV (A = unbound at d > 2.0 nm)
maxcal-joint --fwd "on/*.xvg" --bwd "off/*.xvg" --system binding \
    --qa 2.0 --qb 0.6 --qts-fwd 1.4 --qts-bwd 0.8 --qtse 1.0 \
    --temperature-relation same --dG -8.0 --box-volume 343 --out joint_bind
```

**Outputs**

- `joint_summary.json`: per-direction statistics (p, geometric test, CV, KS,
  renewal mean, mean TP duration and crossings); TP symmetry p-values; kinetic ΔG
  (box and, for binding, standard); joint or independent λ with the tilted p′ in
  both directions; the scan; TSE feature comparison; notes.
- `tse_forward.csv`, `tse_backward.csv` (frames with weights), and `tse_features.csv`
  if features are present.
- `survival_joint.png`, `tp_durations.png`, `tse_features.png`.

### 6.6 Joint-mode tests

- **`test_joint.py`** (fast, 13 tests):
  - orientation and mirroring;
  - transition-path extraction and exact time-reversal symmetry of the TSE frame
    rule;
  - renewal mean and its inverse;
  - standard-state conversion;
  - the detailed-balance constraint holding along the whole scan;
  - TSE path weighting;
  - required `--temperature-relation`;
  - end-to-end runs at the same and at different temperatures;
  - invariance under reversing the CV (Q vs d = 1.5 − Q);
  - binding requiring a volume.
- **`test_joint_validation.py`** (slow, 5 tests), on an asymmetric double well
  (ΔG ≈ −0.6 kT) with a low-barrier model, a 2 kT-bump truth, and a hot backward set:
  - kinetic ΔG equals the equilibrium committor-split ΔG within 0.3 kT (observed
    ≤ 0.1 kT);
  - the detailed-balance coupling predicts the true backward success probability
    within 25% (observed ≤ 11%) and the backward mean folding time within 20%
    (observed ≤ 8%);
  - transition paths are time-reversal symmetric at the same temperature;
  - a different temperature is detected, and misuse of "same" is flagged.

---

## 7. Matching experimental rates with a Poisson target (`maxcal-target`)

A complementary tool for the case where **experimental rate constants are available for
several systems** (e.g. mutant series) and the model is systematically fast. Instead of
raising a barrier, it reweights trajectories so that their first-passage times match the
experimental rates *and* a single-exponential target. The weights are the product: applied
to structural observables, they show which trajectories, and which mechanisms, the
experimental kinetics favour.

### 7.1 Clock calibration

Model rates are computed from the mean first-passage times (binding is pseudo-first-order
at the simulation concentration, `--conc`) and compared with experiment,
A = k_model/k_exp. A single global clock factor c is fitted across all systems and both
directions, leaving per-system residual factors A/c: the change each mean time still needs.
Only these residuals are physical, since c cancels in every ratio.

`--clock` chooses c:
- `geometric` — the geometric mean of A (the natural estimate);
- `decrease-only` — the largest A, so every system needs a *slowdown*;
- `balanced` (default) — maximises the worst-case N_eff;
- an explicit number.

The choice matters because slowing a sample down is cheap while speeding it up is not: for
an exponential sample, changing the mean by a factor f keeps N_eff/N ≈ 2f − f², and f = 2
is a hard limit.

### 7.2 Weights

- **`--mode target` (default).** w ∝ f_target/f_model, with f_model a gamma fitted to the
  times by maximum likelihood and f_target the exponential with the experimental mean. Both
  the mean and the exponential shape are imposed. The weights are flat when the model
  already agrees (N_eff ≈ N), and a final exponential tilt fixes the mean exactly.
- **`--mode mean`.** w ∝ exp(−θt), the maximum-entropy solution with the mean alone
  constrained; useful for comparison.

Diagnostics per system and direction: `N_eff`, its analytic expectation, the largest single
weight, and `tail_gap` = exp(−t_max/τ_target), the target mass beyond the longest observed
time, which is the honest limit on how much slower the target can be. In `target` mode the
KS check is satisfied by construction and is only reported for completeness.

### 7.3 Usage

```bash
maxcal-target systems.csv --conc 0.017 --clock balanced --out target_out
```

with `systems.csv`:

```
system,kon_exp,koff_exp,bind_times,unbind_times
EQVTAV_WT,2.6,22,times/EQVTAV_WT_bind.dat,times/EQVTAV_WT_unbind.dat
EQVTAV_L18A,2.4,10.4,times/EQVTAV_L18A_bind.dat,times/EQVTAV_L18A_unbind.dat
```

`kon_exp` in µM⁻¹s⁻¹ and `koff_exp` in s⁻¹ (`--system-kind first-order` takes both in s⁻¹,
e.g. folding/unfolding). Each times file holds one first-passage time per line, in model
time units, in trajectory order.

**Outputs:** `summary.csv` (clock residuals and diagnostics), `weights_<system>_<dir>.csv`
(one weight per trajectory, for structural analysis), `calibration.png` (needed factors and
N_eff), `cdf_bind.png` / `cdf_unbind.png` (model, reweighted and target CDFs).

**Using the weights.** For any per-trajectory observable O_i (contacts formed at the
transition, encounter-complex lifetime, order of contact formation), compare Σ w_i O_i with
the unweighted mean; bootstrap over trajectories for errors. A large change with healthy
N_eff means the experimental kinetics select a distinguishable subset of the model's
mechanisms.

### 7.4 Tests

`test_target.py` (unit): the weights reproduce the target mean and CDF; they are flat when the
target equals the model; the `mean` mode is an exact exponential tilt; clock modes; the
N_eff formula; an observable test in which reweighting a mis-shaped model sample recovers
the target's average of a descriptor correlated with the first-passage time; and an
end-to-end run whose needed factors scale with 1/c.

`test_target_regtest.py` (regression): a seeded 12-system PDZ2-like dataset built from
the experimental table and the model rates (`tests/synthetic.py: write_rate_dataset`) is
run through `maxcal-target` and compared with `tests/regtest/reference_target.csv`, with
sanity checks that every weighted mean equals its target, the needed factors stay in a
reachable range, and N_eff stays above 20 of 50.

---

## References

1. Scalone E., Broggini L., Visentin C., Erba D., Bačić Toplek F., Peqini K.,
   Pellegrino S., Ricagno S., Paissoni C., Camilloni C. "Multi-eGO: An in silico lens
   to look into protein aggregation kinetics at atomic resolution." *PNAS* **119**,
   e2203181119 (2022).
2. van Erp T. S., Moroni D., Bolhuis P. G. "A novel path sampling method for the
   calculation of rate constants." *J. Chem. Phys.* **118**, 7762 (2003).
3. Allen R. J., Warren P. B., ten Wolde P. R. "Sampling rare switching events in
   biochemical networks." *Phys. Rev. Lett.* **94**, 018104 (2005).
4. Jaynes E. T. "The minimum entropy production principle." *Annu. Rev. Phys. Chem.*
   **31**, 579–601 (1980).
5. Pressé S., Ghosh K., Lee J., Dill K. A. "Principles of maximum entropy and maximum
   caliber in statistical physics." *Rev. Mod. Phys.* **85**, 1115–1141 (2013).
6. Ghosh K., Dixit P. D., Agozzino L., Dill K. A. "The maximum caliber variational
   principle for nonequilibria." *Annu. Rev. Phys. Chem.* **71**, 213–238 (2020).
7. Kish L. *Survey Sampling*. Wiley, New York (1965).
8. Rényi A. "A characterization of Poisson processes." *Magyar Tud. Akad. Mat. Kutató
   Int. Közl.* **1**, 519–527 (1956).
9. Kalashnikov V. *Geometric Sums: Bounds for Rare Events with Applications*. Kluwer,
   Dordrecht (1997).
10. Du R., Pande V. S., Grosberg A. Yu., Tanaka T., Shakhnovich E. I. "On the
    transition coordinate for protein folding." *J. Chem. Phys.* **108**, 334 (1998).
11. Bolhuis P. G., Chandler D., Dellago C., Geissler P. L. "Transition path sampling:
    throwing ropes over rough mountain passes, in the dark." *Annu. Rev. Phys. Chem.*
    **53**, 291–318 (2002).
12. Kramers H. A. "Brownian motion in a field of force and the diffusion model of
    chemical reactions." *Physica* **7**, 284–304 (1940).
13. Hänggi P., Talkner P., Borkovec M. "Reaction-rate theory: fifty years after
    Kramers." *Rev. Mod. Phys.* **62**, 251–341 (1990).
14. Voter A. F. "Hyperdynamics: accelerated molecular dynamics of infrequent events."
    *Phys. Rev. Lett.* **78**, 3908 (1997).
15. Tiwary P., Parrinello M. "From metadynamics to dynamics." *Phys. Rev. Lett.*
    **111**, 230602 (2013).
16. Salvalaglio M., Tiwary P., Parrinello M. "Assessing the reliability of the
    dynamics reconstructed from metadynamics." *J. Chem. Theory Comput.* **10**,
    1420–1425 (2014).
17. Kaplan E. L., Meier P. "Nonparametric estimation from incomplete observations."
    *J. Am. Stat. Assoc.* **53**, 457–481 (1958).
18. Lilliefors H. W. "On the Kolmogorov–Smirnov test for the exponential
    distribution with mean unknown." *J. Am. Stat. Assoc.* **64**, 387–389 (1969).
19. Efron B. "Bootstrap methods: another look at the jackknife." *Ann. Stat.* **7**,
    1–26 (1979).

**Related MaxCal approaches to kinetics** (alternatives to the attempt-level tilt used
here):

20. Dixit P. D., Dill K. A. "Caliber corrected Markov modeling (C2M2): correcting
    equilibrium Markov models." *J. Chem. Theory Comput.* **14**, 1111–1119 (2018).
21. Brotzakis Z. F., Vendruscolo M., Bolhuis P. G. "A method of incorporating rate
    constants as kinetic constraints in molecular dynamics simulations." *PNAS*
    **118**, e2012423118 (2021).
22. Fersht A. R., Matouschek A., Serrano L. "The folding of an enzyme. I. Theory of
    protein engineering analysis of stability and pathway of protein folding."
    *J. Mol. Biol.* **224**, 771–782 (1992). (Φ-value analysis, for comparing the TSE
    with experiment.)

**Joint mode**

23. Hummer G. "From transition paths to transition states and rate coefficients."
    *J. Chem. Phys.* **120**, 516–523 (2004).
24. E W., Vanden-Eijnden E. "Towards a theory of transition paths."
    *J. Stat. Phys.* **123**, 503–523 (2006).
25. Jackson S. E., Fersht A. R. "Folding of chymotrypsin inhibitor 2. 1. Evidence for
    a two-state transition." *Biochemistry* **30**, 10428–10435 (1991).
26. Best R. B., Hummer G. "Reaction coordinates and rates from transition paths."
    *PNAS* **102**, 6732–6737 (2005).
27. Hammond G. S. "A correlation of reaction rates." *J. Am. Chem. Soc.* **77**,
    334–338 (1955).
28. Leffler J. E. "Parameters for the description of transition states."
    *Science* **117**, 340–341 (1953).
29. Donati L., Hartmann C., Keller B. G. "Girsanov reweighting for path ensembles and
    Markov state models." *J. Chem. Phys.* **146**, 244112 (2017).
30. Gilson M. K., Given J. A., Bush B. L., McCammon J. A. "The statistical-thermodynamic
    basis for computation of binding affinities: a critical review."
    *Biophys. J.* **72**, 1047–1069 (1997).
