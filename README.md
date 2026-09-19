# maxcal

Maximum-caliber (MaxCal) analysis of first-passage trajectories: correcting an
underestimated barrier, recovering two-state (Poisson) kinetics, locating the transition
state, and reweighting onto measured rate constants.

The intended use case is a coarse-grained or structure-based model such as multi-eGO [1] that
samples the right structures but has a smoothed landscape: barriers are too low, kinetics
are accelerated, and folding or binding times need not be exponential. The tools quantify
how much barrier is missing, test whether the model's kinetics are two-state at all, and
turn the answer into weights you can apply to structural observables.

| | |
|---|---|
| **Theory** | [Part I](#part-i--theory): the tilt, its two implementations, what the data do and do not determine |
| **Applications** | [Part II](#part-ii--applications): task-shaped recipes with commands and how to read the output |
| **Syntax** | [Part III](#part-iii--reference): per-tool options, outputs, file formats, status codes |
| **Rate targets** | [`docs/TARGET.md`](docs/TARGET.md): the `maxcal-target` workflow in detail |
| **Interactive** | `notebooks/demo.ipynb`, `notebooks/target_demo.ipynb` |

---

## Installation

```bash
git clone git@github.com:multi-ego/MaxCal.git
cd MaxCal
pip install -e ".[test]"          # add ,demo for JupyterLab + ipywidgets
pytest -q                          # 67 fast tests; --runslow adds 16 more
```

Python >= 3.9, NumPy, SciPy, Matplotlib.

## Quick start

Input: one file per trajectory holding `time Q` (or a single column of Q with `--dt`);
`#` and `@` lines are ignored, so GROMACS `.xvg` works directly. Trajectories start in the
unfolded/unbound basin and stop when they reach the folded/bound state.

```bash
# 1. how much barrier is missing, and are the kinetics two-state?
maxcal-stitch    "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 --out stitch_out

# 2. weights on whole trajectories (memory-safe, no stitching)
maxcal-reweight  "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 --out reweight_out

# 3. committor, corrected transition state, TSE frames
maxcal-committor "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 \
                 --from-stitch stitch_out/summary.csv --out committor_out
```

## What is in the repository

| command | module | purpose |
|---|---|---|
| `maxcal-stitch` | `maxcal/stitching.py` | thinning barrier crossings: λ_min, attempt statistics, (Q‡, λ) decision map |
| `maxcal-reweight` | `maxcal/reweight.py` | MaxCal weights on whole trajectories: λ*, N_eff, per-trajectory weights |
| `maxcal-committor` | `maxcal/committor.py` | model and corrected committor, TS location, committor-based TSE frames |
| `maxcal-joint` | `maxcal/joint.py` | forward + backward runs: kinetic ΔG, transition-path symmetry, coupled tilt |
| `maxcal-target` | `maxcal/target.py` | reweighting onto experimental rate constants with a Poisson target |

All share `maxcal/core.py` (parsing, attempt counting, the tilt, survival/KS statistics,
status codes, common CLI arguments). Each runs as `python -m maxcal.<module>` too, and
`import maxcal as m` re-exports the shared functions (`m.attempts`, `m.stitch`,
`m.cv_curve`, `m.ts_location`).

---

# Part I — Theory

## 1. Setting: basins, attempts, first-passage times

**Input ensemble.** N trajectories Q(t), all started in U (Q < Q_u) and stopped as soon as
Q >= Q_f. Trajectories that never arrive are right-censored.

**Excursions and attempts.** An excursion starts when a trajectory leaves U and ends either
back in U (failure) or at Q_f (success). Given an interface Q‡ with Q_u < Q‡ < Q_f, an
*attempt* is an excursion whose maximum Q exceeds Q‡. This is the decomposition of
transition interface sampling and forward flux sampling [2, 3]:

$$
k_{UF} = \Phi_{U,Q^\ddagger} \cdot P(Q_f \mid Q^\ddagger)
$$

the flux of attempts through Q‡ times the probability that an attempt reaches F before
returning to U. Per trajectory the tools record the number of failed attempts k_i and the
first-passage time T_i, and estimate the success probability per attempt as

$$
\hat p = \frac{N_\text{done}}{N_\text{done} + \sum_i k_i}
$$

**Why this decomposition.** Everything below acts on attempt *outcomes* only. That is what
makes the correction minimal: the trajectories themselves are never modified.

## 2. The MaxCal tilt

### 2.1 Assumptions

**(A1) Markovian at the interface.** Attempt outcomes are independent Bernoulli(p)
variables, so the reference path measure factorises as P(ω) = P(ω | Y) P(Y), with
Y = (s_1, s_2, ...) the sequence of outcomes.

**(A2) Correct sampling, wrong barrier.** Model and reality differ only in how often an
attempt succeeds, not in what the trajectories look like once the outcomes are fixed.

(A1) is testable (§10), and (A2) is derived rather than assumed, as the next paragraph
shows.

### 2.2 Derivation

Maximum caliber [4–6] picks the path measure closest to a reference P, in relative path
entropy, subject to constraints:

$$
\min_{P'} \sum_\omega P'(\omega) \ln \frac{P'(\omega)}{P(\omega)} \quad \text{s.t.} \quad \langle g(Y)\rangle_{P'} = G
$$

Taking g(Y) = Σ_j s_j, the mean number of successful crossings in a fixed number of
attempts (i.e. fixing the crossing flux, i.e. the barrier), the solution is the exponential
tilt P'(ω) = P(ω) e^(−λ g(Y)) / Z. Two consequences:

1. **Conditionals are untouched:** P'(ω | Y) = P(ω | Y). Given the outcome of each attempt,
   dwell times, excursion shapes and structures are exactly the model's. This is (A2),
   obtained as the least-biased choice.
2. **The outcome process stays Bernoulli with shifted log-odds:**

$$
\mathrm{logit} p' = \mathrm{logit} p - \lambda, \qquad p' = \frac{p e^{-\lambda}}{1 - p(1 - e^{-\lambda})}
$$

λ is therefore a shift in log-odds per attempt. Counting crossings in both directions tilts
U→F and F→U equally, so in the rare-event limit ΔG is unchanged.

### 2.3 Weights for first-passage trajectories

A completed trajectory with k failures has reference probability (1−p)^k p and tilted
probability (1−p')^k p', so

$$
w_\text{done}(k) \propto c^{k+1} e^{-\lambda}, \qquad w_\text{cens}(k) \propto c^{k}, \qquad c = \frac{1}{1 - p(1 - e^{-\lambda})}
$$

Tilting an exactly geometric(p) population with these weights gives exactly geometric(p');
this identity is unit-tested.

### 2.4 Two implementations, and when each applies

| | `maxcal-reweight` | `maxcal-stitch` |
|---|---|---|
| acts on | whole trajectories, weights w ∝ c^k | resampled segments, glued into new events |
| needs (A1)? | **no** — each trajectory keeps its history | **yes** — segments must be exchangeable |
| reaches large λ? | no: weights saturate at (1−p)^−k, N_eff collapses | yes: builds events longer than any observed |
| recovers Poisson from a lag? | no (CV moves away from 1) | yes |
| natural use | memory, route heterogeneity, small corrections, matching a target mean | the Poisson question, λ_min, the decision map |

**Saturation.** As λ grows, c tends to 1/(1−p), so reweighting can only redistribute
probability over the k values actually observed. If a higher barrier implies a geometric
distribution of k with mean (1−p')/p' much larger than k_max, that tail is absent from the
data; upweighting the trajectories with most failures then makes the times *more* regular
(sums of more cycles), and the weighted CV falls instead of approaching 1. On the
regression data it falls from 0.86 to about 0.5. The tools report the Kish effective sample
size [7]

$$
N_\text{eff} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
$$

and flag `low_neff` below `--neff-min`.

### 2.5 Stitching: sampling the tilted measure generatively

Under (A1) and the conditional invariance of §2.2, a tilted first-passage time can be drawn
directly:

$$
K \sim \mathrm{Geom}(p'), \qquad T' = \sum_{j=1}^{K} C_j + S
$$

with C_j resampled from the observed cycles ("return to U, then the next failed attempt")
and S from the successful final segments. By default the first segment of each trajectory
is excluded from both pools, because it carries the relaxation from the starting
structures. The moments are

$$
\langle T'\rangle = \frac{1-p'}{p'} \mu_C + \mu_S, \qquad \mathrm{Var} T' = \frac{1-p'}{p'}\sigma_C^2 + \frac{1-p'}{p'^2}\mu_C^2 + \sigma_S^2
$$

As p' tends to 0 the second variance term dominates, CV tends to 1, and p'T'/μ_C converges
to Exp(1): Rényi's theorem on geometric sums [8, 9]. Thinning a renewal process of attempts
produces a Poisson process of successes, which is how a higher barrier restores two-state
kinetics.

**The lag does not vanish; it becomes negligible.** No event is shorter than one crossing,
so a fixed dead time survives in absolute terms (about 0.6–0.7 time units in the validation
data, both for the stitched result and for the simulated ground truth). Raising the barrier
multiplies the waiting time (7.4 to 31) without changing that dead time, so its relative
weight falls from about 10% to about 2% of the mean. That is two-state kinetics: a fixed
dead time plus a long, memoryless wait.

## 3. What the data determine, and what they do not

### 3.1 λ_min is a lower bound

Because CV tends to 1 monotonically as p' tends to 0, the smallest λ passing the Poisson
criterion is only a lower bound on the true tilt. In the validation, λ_min came out 0–0.35
against true values of 0.9–1.7. Recovering Poisson statistics is therefore **not** evidence
for the model: any process with independent attempts becomes Poisson under strong enough
thinning.

Evidence for two-state behaviour must come from quantities thinning cannot manufacture:

- **attempt independence** — the k_i geometric (χ²), successive cycle durations uncorrelated
  (lag-1 Spearman);
- **timescale separation** — relaxation inside U fast compared with a cycle;
- **committor** — values on the TS surface peaked around 1/2 [10, 11];
- **with both directions** — kinetic ΔG equal to the equilibrium ΔG, and transition paths
  time-reversal symmetric (§5).

### 3.2 λ is an effective log-odds shift, not a barrier height

In the Kramers regime [12, 13] P(Q_f | Q‡) scales as exp(−βΔF‡), so λ tends to βΔΔF‡; at
finite barriers λ is smaller. Gaussian bumps of 2 and 3 kT in the validation gave λ ≈
0.9–1.1 and 1.7–1.9.

### 3.3 Folding times cannot locate the barrier

The same 3 kT missing barrier placed at Q = 0.46, 0.55 or 0.64 moves the true TS by 0.17 in
Q, but the folding-time shapes are statistically indistinguishable (two-sample KS p >= 0.25).
Where the barrier sits is an *input* (§4.2), not an output.

## 4. Geometry: where to put the interfaces

### 4.1 Q‡ at the foot of the barrier

The tilt assumes the extra barrier acts only after Q‡, so the attempt flux must be
unaffected. Put Q‡ on the U side, at the foot. With Q‡ on the barrier top, the missing
barrier also changes how often Q‡ is reached, and stitching underestimates the true mean
first-passage time by a factor of 3–5.

The corrected **TS always lies between Q‡ and Q_f**, so Q‡ is a lower bound on its
location, and a Q‡ at or beyond the barrier is invalid.

**Decision map.** `heatmap.png` shows the median stitched KS p-value on a grid of Q‡ and λ:
hatched cells pass the Poisson criterion, triangles mark λ_min(Q‡), and the shaded region
covers Q‡ at or beyond the assumed barrier. A flat λ_min over the valid range means the
interface is well placed. Validity ends where the missing barrier *starts*, not at its
centre: in the validation (bump centred at 0.55) the true λ already falls beyond Q‡ ≈ 0.45.
With real data the usable signal is a **rise of λ_min** above its low-Q‡ plateau; use the
flat region before it. Moving Q‡ very close to Q_u is not better: on a projected coordinate
attempts there become correlated (the independence tests fail first), and the bound on the
TS location weakens.

### 4.2 Committor and transition-state location

The tilt fixes *how much* barrier is missing, r = p'/p, but not *where* it sits. The
location is an input, `--barrier-q`, given as a value q* of the **model** committor.

**Model committor.** Every frame inside an excursion is labelled by the outcome of its
excursion (1 if it reaches Q_f before returning below Q_u). For a frame x this label is a
sample of q(x); binning along Q and imposing monotonicity (pool-adjacent-violators) gives
q_m(Q).

**Corrected committor.** For a narrow missing barrier on the model isocommittor surface
q_m = q*, the committor is harmonic away from it, so the barrier only adds a jump:

$$
q'(x) = r q_m(x) \quad \text{for } q_m < q^*
$$

$$
1 - q'(x) = r (1 - q_m(x)) \quad \text{for } q_m > q^*
$$

On the U side the chance of reaching F drops by r; on the F side the chance of returning to
U drops by r, so the barrier blocks recrossing both ways. On the test potential this is
exact to about 1e-4 away from the bump.

**TS location.** q' = 1/2 sits at the barrier (q_m = q*) whenever r q* <= 1/2 <= 1 − r(1 − q*),
and otherwise on the model surface q_m = 1/(2r) (U side) or 1 − 1/(2r) (F side). Two
readings follow:

- **default q\* = 0.5** — the missing barrier sits on the model's own TS, and the TS (with
  the TSE composition there) does not move with λ. This is the precise form of "correct
  structures, underestimated barrier".
- **otherwise** — the TS moves toward the barrier as λ grows and reaches it once the tilt is
  strong. Whatever λ, the TS lies on a model surface with 1 − 1/(2r) <= q_m <= 1/(2r), a band
  that is narrow at λ_min (q_m in [0.36, 0.64] at λ = 0.3, p = 0.22) and covers everything
  by λ ≈ 1.7. Reporting the TSE at both edges of that band is the honest robustness check.

Φ-values [22] would fix q* from experiment; `ts_location.csv` scans q* from 0.1 to 0.9
meanwhile.

**A naive shortcut fails.** Shifting the model committor by λ in log-odds implicitly puts
the whole missing barrier at Q_f and misplaces the TS by more than 0.1 in Q on the test
potential; a test documents this.

## 5. Two directions: folding and unfolding, binding and unbinding

With runs in both directions the relations below hold **only at the same temperature**, so
`maxcal-joint` requires `--temperature-relation same|different` explicitly.

**Detailed balance.** Between core sets [23, 24],

$$
\frac{k_{AB}}{k_{BA}} = \frac{\pi_B}{\pi_A} = e^{-\beta \Delta G_{AB}}
$$

with committor-split populations. Each rate is estimated as 1/M, M being the renewal mean
first-passage time of §2.5 (which excludes the initial relaxation), so the model's scaled
clock cancels in the ratio.

**Kinetic ΔG.** ΔG_kin = −kT ln(M_BA/M_AB) must equal the equilibrium ΔG for a two-state
system [25]. This is the strongest available test, because thinning cannot manufacture it.
In validation it matched the committor-split equilibrium ΔG within 0.1 kT.

**Coupled tilt.** MaxCal on the pair of attempt processes has two constraints, the crossing
flux and the stationary ratio, so fixing ΔG removes one multiplier. The forward tilt is
scanned and the backward success probability follows from M'_BA = R M'_AB with
R = exp(−βΔG_target), the model's own kinetic value by default or `--dG`. In the rare-event
limit this reduces to p'_f/p_f = p'_b/p_b, the same barrier seen from both sides.
Validation: predicting the backward success probability this way was within 11% of truth,
and the backward mean within 8%.

**Transition-path time-reversal symmetry.** At equilibrium the B→A transition-path ensemble
is the time reverse of A→B [11, 26], so the distributions of path durations and of Q_tse
recrossings must agree. This needs only the CV and does not depend on λ. It also detects
data wrongly declared as same-temperature: a 1.5x temperature gave KS p < 1e-3.

**Different temperatures.** λ is not transferable and ΔG changes, so the directions are
analysed independently, `--dG` is ignored, and TSE differences are expected (Hammond shift
[27, 28]). Rigorous cross-temperature path reweighting would need Girsanov likelihood
ratios [29], i.e. the random forces of every step.

**Binding.** For one ligand and one receptor in a box of volume V, K_box = k'_on/k_off with
k'_on pseudo-first-order at [L] = 1/V, and

$$
\Delta G^\circ = \Delta G_\text{box} - kT \ln (V C^\circ), \qquad C^\circ = 1\ \text{M} = 0.6022\ \text{nm}^{-3}
$$

[30]. With `--system binding`, `--dG` is ΔG° and `--box-volume` is required.

## 6. Rate targets (`maxcal-target`)

When experimental rate constants are available for several systems the question changes:
not "how much barrier is missing" but "what must the model look like to reproduce the
measured rates". One global clock factor is fitted across all systems and directions,
leaving per-system residual factors, and each system is reweighted onto an exponential with
its experimental mean. The weights are the minimum-relative-entropy solution and equal the
importance ratio f_target/f_model. The cost is N_eff/N = 2f − f² for a change of the mean by
a factor f, with f = 2 a hard limit.

Full derivation, diagnostics, worked example and references: [`docs/TARGET.md`](docs/TARGET.md).

## 7. Relation to time reweighting

Read as one long trajectory, the tilted ensemble stretches time in the basins relative to
the barrier region by roughly 1/p', the analogue of the acceleration factor in
hyperdynamics [14] and infrequent metadynamics [15], with the missing barrier as a negative
bias. The KS-based validation of exponential statistics follows Salvalaglio *et al.* [16].

## 8. Frame-level weights (what reweighting does *not* give)

Equilibrium averages need Boltzmann weights of the missing barrier,
w(x) = exp(−βΔF_missing(Q(x))), which depend on the barrier's shape as well as on λ. They
are *not* per-path-class weights: in equilibrium the density at a configuration cannot
depend on which trajectory visits it. Given a location and width, the height follows from r
through the model committor, and on the test potential this recovers the transition-region
population almost exactly (0.042 against 0.041 true, versus 0.126 for the model). This is
recorded here for completeness; it is not implemented in the tools.

## 9. Assumption checklist

| assumption | test | where |
|---|---|---|
| attempts independent (A1) | k_i geometric; lag-1 correlation of cycles | `geom_p`, `lag1_rho` |
| Q‡ at the barrier foot | λ_min flat over a Q‡ plateau | `heatmap.png`, `robustness_qts.png` |
| deviation is a missing barrier | CV < 1 (lag), not CV > 1 (mixture) | `cv0` |
| deviation is not just initial relaxation | stitching at λ = 0 | `cv_stitch0`, `ksp_stitch0` |
| barrier location | an assumption; check the TSE band at λ_min | `ts_location.csv` |
| same ensemble in both directions | TP duration symmetry; kinetic vs equilibrium ΔG | `joint_summary.json` |

## 10. Statistics used

- **Survival function** — weighted Kaplan–Meier [17], handling censored trajectories;
  weights act as frequency weights (weight 2 is equivalent to duplicating a trajectory,
  unit-tested).
- **Exponential fit** — maximum likelihood with censoring, tau = Σ w_i T_i / Σ w_i δ_i.
- **Exponentiality test** — KS against an exponential with fitted scale; because the scale
  is estimated, the null is simulated once per sample size (Lilliefors [18]) and is
  scale-free. For stitched samples the test is run on subsamples of the original size, so
  its power matches the data.
- **Independence of attempts** — χ² against a geometric distribution of k; lag-1 Spearman
  correlation of successive cycle durations.
- **Uncertainty** — nonparametric bootstrap over trajectories [19].

---

# Part II — Applications

Each recipe lists the commands, what to read, and what can go wrong. All use the input
convention of the [quick start](#quick-start).

## A. Are my model's kinetics two-state?

```bash
maxcal-stitch "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 --out stitch_out
```

**Read, in this order:**

1. `geom_p` and `lag1_rho` (`summary.csv`) — if the geometric test fails, attempts have
   memory and any Poisson result afterwards is imposed, not recovered. Stop and look for
   intermediates or route heterogeneity.
2. `cv0` — below 1 means a lag (missing barrier, possibly plus initial relaxation); above 1
   means over-dispersion, i.e. a mixture, which a barrier correction cannot explain.
3. `cv_stitch0`, `ksp_stitch0` — stitching at λ = 0, with the initial segments removed. If
   this already passes, the original deviation was mostly the starting-structure relaxation.
4. `lam_stitch` (λ_min) and `heatmap.png` — see recipe B.

**Pitfalls.** A funnelled model produces few failed attempts: watch for `empty_pool`,
`ok_pool_fallback` and `too_few_bins`. Aim for at least ~50 failure cycles at the chosen
Q‡; lower Q‡, run more trajectories, or generate failures by shooting from the transition
region.

## B. How much barrier is missing?

```bash
maxcal-stitch "runs/q_*.xvg" --qu 0.3 --qf 0.8 --nqts 9 --out stitch_out
```

- **λ_min** is a *lower bound* on the missing barrier (§3.1) and an effective log-odds
  shift, not a barrier height (§3.2). Report it as "the minimal correction consistent with
  two-state kinetics".
- **Choose Q‡** on the plateau of λ_min in `heatmap.png` and `robustness_qts.png`, at its
  upper end but before the rise (§4.1), and report λ_min as the spread over that plateau.
- **Pinning λ** requires an absolute rate on a calibrated clock; the Poisson criterion
  cannot supply it. For relative statements between mutants, a shared λ cancels.

## C. Transition-state ensemble and its robustness

```bash
maxcal-committor "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 \
                 --from-stitch stitch_out/summary.csv --out committor_out
```

- `tse_committor_lam*_q0.50.csv` lists frames within one bin of the corrected TS, with the
  outcome of their excursion. Compute your structural observables on those frames
  (per-residue contacts, contact maps, clustering).
- **State the assumption:** with the default `--barrier-q 0.5` the TS does not move with λ.
  Re-run at the band edges (§4.2) to show whether the structural conclusion survives.
- The transition-*path* ensemble is λ-independent in any case: the tilt changes how often
  paths occur, not what they contain. Mechanism analyses (order of contact formation, route
  fractions) need no weights.

## D. Both directions: folding + unfolding, binding + unbinding

```bash
# same temperature, experimental DeltaG imposed
maxcal-joint --fwd "fold/*.xvg" --bwd "unfold/*.xvg" \
    --qa 0.3 --qb 0.8 --qts-fwd 0.4 --qts-bwd 0.7 --qtse 0.55 \
    --temperature-relation same --dG -2.5 --out joint_out

# binding on a distance CV (A = unbound at d > 2.0 nm), one ligand in a box of 343 nm^3
maxcal-joint --fwd "on/*.xvg" --bwd "off/*.xvg" --system binding \
    --qa 2.0 --qb 0.6 --qts-fwd 1.4 --qts-bwd 0.8 --qtse 1.0 \
    --temperature-relation same --dG -8.0 --box-volume 343 --out joint_bind
```

**Read** (`joint_summary.json`): `dG_kin_box_renewal` against `--dG`; `tp_duration_ks_p`
(time-reversal symmetry); `lambda_joint_f` and `lambda_joint_b`; the TSE feature comparison
if the input files carry extra columns.

**Pitfalls.** The interfaces must sit at the foot of the barrier *on their own side*.
Declaring `same` for data at different temperatures is detected by the TP symmetry test.
For binding, define the unbound basin consistently with the box volume.

## E. A mutant series with measured rate constants

```bash
maxcal-target systems.csv --conc 0.017 --clock balanced --out target_out
```

Fits one clock across all systems, then reweights each onto its experimental rates with a
Poisson shape; `weights_<system>_<direction>.csv` is what you join to a structural analysis.
Read `N_eff` and `tail_gap` to see whether the experimental kinetics are reachable within
the mechanisms the model samples. Details and a worked twelve-system example:
[`docs/TARGET.md`](docs/TARGET.md).

## F. When reweighting is the right tool instead of stitching

`maxcal-reweight` keeps each trajectory intact, so it applies where stitching does not
(related approaches that constrain kinetics differently: caliber-corrected Markov models
[20] and rate constants as MD restraints [21]):

- **attempts with memory** (`geom_p` small) — stitching would glue non-exchangeable
  segments;
- **route heterogeneity** — weights shift route proportions consistently, since each
  trajectory keeps its route;
- **small corrections** — at λ ≈ 0.25–0.5, N_eff stays near N and any per-trajectory
  observable can be weighted directly;
- **cross-check** — at small λ both methods should agree if (A1) holds; disagreement is a
  memory diagnostic.

Read `ks_p_star` together with `cv0`, `cv_lammax` and `neff_star`: with pure reweighting a
KS "pass" can come from lost power rather than from a Poissonian shape.

---

# Part III — Reference

## Input format

- One file per trajectory: `time Q [f1 f2 ...]`, or a single column of `Q` with `--dt`.
  Lines starting with `#` or `@` are ignored.
- Extra columns are per-frame features, used by `maxcal-joint` to compare TSE profiles.
- Trajectories start in the start basin and stop on reaching the target state; those that
  never arrive are treated as right-censored.
- `maxcal-target` instead reads a CSV of systems plus one file of first-passage times per
  system and direction (see [`docs/TARGET.md`](docs/TARGET.md)).

## Common options (`maxcal-stitch`, `maxcal-reweight`, `maxcal-committor`)

| option | meaning |
|---|---|
| `--qu`, `--qf` | start basin (Q < Qu) and target state (Q >= Qf) |
| `--dt` | frame spacing, required for single-column files |
| `--qts`, `--nqts` | reference attempt interface; number of Q‡ values scanned in (Qu, Qf) |
| `--t0` | discard an initial relaxation window |
| `--include-initial` | keep the first segment of each trajectory in the stitching pools |
| `--alpha`, `--cv-tol` | Poisson criterion: median KS p >= alpha and abs(CV − 1) <= tol |
| `--seed`, `--out` | RNG seed; output directory (required) |

## `maxcal-stitch`

```bash
maxcal-stitch "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 --nqts 9 \
              --nstitch 5000 --nsub 200 --heatmap-nlam 21 --out stitch_out
```

| option | default | meaning |
|---|---|---|
| `--lam-max`, `--nlam` | 10, 201 | λ grid for the λ_min scan |
| `--nstitch` | 5000 | synthetic first-passage times per λ |
| `--nsub` | 200 | KS subsamples, each of the original sample size |
| `--heatmap-nlam`, `--heatmap-lam-max` | 21, 5 | decision-map λ grid (0 disables) |
| `--barrier-q` | 0.5 | assumed barrier location (model committor) marking the valid Q‡ range |

**Outputs:** `summary.csv` (per Q‡: N, n_done, sum_k, kmax, frac_k0, p, geom_p, lag1_rho,
lag1_p, cv0, ks_p0, cv_stitch0, ksp_stitch0, lam_stitch, p_stitch, status codes, note),
`heatmap.csv`, `heatmap.png`, `survival.png`, `robustness_qts.png`.

## `maxcal-reweight`

```bash
maxcal-reweight "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.45 --qtse 0.55 \
                --boot 500 --neff-min 30 --out reweight_out
```

| option | default | meaning |
|---|---|---|
| `--qtse` | each Q‡ | surface for TSE frames (last upward crossing before the target) |
| `--lam-max`, `--nlam` | 10, 201 | λ grid for the CV curve |
| `--boot` | 500 | bootstrap resamples for λ* |
| `--neff-min` | 30 | N_eff below which the result is flagged |

**Outputs:** `summary.csv` (per Q‡: p, cv0, cv_lammax, ks_p0, lam_star with `lam_lo`,
`lam_hi` and `boot_noroot`, neff_star, ks_p_star, reweight_ok, status codes),
`weights_qts<Q‡>.csv` (trajectory, k_failed, time, completed, weight_at_lambda_star),
`tse_frames_qts<Q‡>.csv`, `survival.png`, `lambda_scan.png`.

## `maxcal-committor`

```bash
maxcal-committor "runs/q_*.xvg" --qu 0.3 --qf 0.8 --qts 0.40 \
                 --barrier-q 0.5 --lams 0.3 1.3 2.3 --committor-bins 20 --out committor_out
```

| option | default | meaning |
|---|---|---|
| `--barrier-q` | 0.5 | assumed barrier location as a model committor value q* |
| `--lams` | 0 1 2 | λ values for the corrected committor and the TSE |
| `--from-stitch` | – | take λ_min from a `maxcal-stitch` summary and use λ_min, +1, +2 |
| `--committor-bins` | 20 | bins along Q |
| `--tse-window` | one bin | half-width in Q of the TSE window around Q_TS |

**Outputs:** `committor.csv` (Q, n_frames, raw and isotonic model committor),
`ts_location.csv` (λ, r, q*, Q_barrier, Q_TS, which case sets it, status),
`ts_location.png`, `tse_committor_lam<λ>_q<q*>.csv` (trajectory, frame, time, Q, excursion
outcome).

## `maxcal-joint`

```bash
maxcal-joint --fwd "fold/*.xvg" --bwd "unfold/*.xvg" \
             --qa 0.3 --qb 0.8 --qts-fwd 0.4 --qts-bwd 0.7 --qtse 0.55 \
             --temperature-relation same --dG -2.5 --boot 200 --out joint_out
```

| option | meaning |
|---|---|
| `--fwd`, `--bwd` | forward (A→B) and backward (B→A) trajectory patterns |
| `--temperature-relation` | `same` or `different` — **required** |
| `--qa`, `--qb` | basin boundaries; `--qa > --qb` flips the orientation (e.g. a distance) |
| `--qts-fwd`, `--qts-bwd` | attempt interfaces, each at the foot on its own side |
| `--qtse` | TSE surface (barrier top); default: midpoint of the two interfaces |
| `--system`, `--box-volume` | `folding` (default) or `binding`; box volume in nm³ for ΔG° |
| `--dG` | G_B − G_A in kT (binding: ΔG°); same-temperature only |
| `--boot` | bootstrap resamples for the TSE feature comparison |

**Outputs:** `joint_summary.json` (per-direction statistics, TP symmetry p-values, kinetic
ΔG, joint or independent λ, TSE feature comparison, notes), `tse_forward.csv`,
`tse_backward.csv`, `survival_joint.png`, `tp_durations.png`, and — only when the input
files carry feature columns — `tse_features.csv` and `tse_features.png`.

## `maxcal-target`

```bash
maxcal-target systems.csv --conc 0.017 --clock balanced --mode target --out target_out
```

| option | default | meaning |
|---|---|---|
| `--conc` | 0.017 | ligand concentration in the box, in M (binding) |
| `--system-kind` | `binding` | or `first-order` (both rates in s⁻¹) |
| `--clock` | `balanced` | `geometric`, `decrease-only`, `balanced`, or a number |
| `--mode` | `target` | rate and Poisson shape; `mean` constrains only the rate |

**Outputs:** `summary.csv`, `weights_<system>_<direction>.csv`, `calibration.png`,
`cdf_bind.png`, `cdf_unbind.png`. See [`docs/TARGET.md`](docs/TARGET.md).

## Status codes

Every NaN in a summary comes with a reason.

| column | values |
|---|---|
| `geom_status` | `ok`; `too_few_trajectories` (fewer than 10 completed); `too_few_bins` (too few distinct k for the χ² test) |
| `lag1_status` | `ok`; `too_few_pairs` (fewer than 10 consecutive failure cycles); `constant_cycles` (correlation undefined) |
| `reweight_status` | `ok`; `no_root` (weighted CV never reaches 1, §2.4); `low_neff` (below `--neff-min`); `cv_ge_1_at_lambda0` (already over-dispersed: not a missing barrier) |
| `stitch_status` | `ok`; `ok_pool_fallback` (initial segments had to be used); `no_pass` (Poisson criterion never met); `empty_pool` (no failed attempts past this Q‡) |

---

# Validation and tests

```bash
pytest -q                     # 67 fast tests (~20 s)
pytest -q --runslow           # + 16 validation tests and the two notebooks (~50 s)
MAXCAL_UPDATE_REF=1 pytest tests/test_regtest.py tests/test_target_regtest.py   # refresh references
```

| file | what it covers |
|---|---|
| `test_unit.py` | parsing, attempt counting, the tilt identities (§2.2–2.4), Kaplan–Meier, KS/Lilliefors size and power, the geometric test, stitching moments, status codes, I/O |
| `test_committor.py` | the located-barrier formula against exact 1D committors (1e-3 away from the bump); the naive log-odds shift misplaces the TS; sampled prediction within 0.03 in Q |
| `test_joint.py` | orientation and mirroring, transition paths and the time-reversal-symmetric TSE rule, renewal mean and its inverse, standard-state conversion, detailed balance along the scan, end-to-end runs, CV-orientation invariance |
| `test_target.py` | target and mean-mode weights, clock modes, the N_eff formula, recovery of a descriptor's target average, end-to-end |
| `test_reweight_path.py` | the finite-λ* branch end to end on hand-built data where the weighted CV crosses 1 |
| `test_regtest.py` | the three trajectory tools on 100 seeded Langevin trajectories against `tests/regtest/reference_stitching.csv` and `reference_reweight.csv` |
| `test_target_regtest.py` | `maxcal-target` on twelve PDZ2-like systems against `reference_target.csv` |
| `test_validation.py` (slow) | ground truth: the same potential plus a Gaussian bump. Stitching the low-barrier segments with the true p' reproduces the high-barrier distribution (mean within 15%, observed 1–7%); the Poisson threshold is a lower bound; Q‡ on the barrier top fails |
| `test_joint_validation.py` (slow) | kinetic ΔG equals the equilibrium value within 0.3 kT (observed 0.1 or better); the detailed-balance coupling predicts the backward rate within 25% (observed 11% or better); TP symmetry holds at one temperature and breaks at another |
| `test_notebook.py` (slow) | both notebooks execute top to bottom |

Test data are generated on the fly by `tests/langevin.py` (overdamped Langevin on a double
well, optional Gaussian bump, tilt, temperature, reverse runs) and `tests/synthetic.py`
(hand-built attempt structures; the twelve-system rate dataset).

---

# References

1. Scalone E. *et al.* "Multi-eGO: an in silico lens to look into protein aggregation
   kinetics at atomic resolution." *PNAS* **119**, e2203181119 (2022).
2. van Erp T. S., Moroni D., Bolhuis P. G. "A novel path sampling method for the calculation
   of rate constants." *J. Chem. Phys.* **118**, 7762 (2003).
3. Allen R. J., Warren P. B., ten Wolde P. R. "Sampling rare switching events in biochemical
   networks." *Phys. Rev. Lett.* **94**, 018104 (2005).
4. Jaynes E. T. "The minimum entropy production principle." *Annu. Rev. Phys. Chem.* **31**,
   579–601 (1980).
5. Pressé S., Ghosh K., Lee J., Dill K. A. "Principles of maximum entropy and maximum
   caliber in statistical physics." *Rev. Mod. Phys.* **85**, 1115–1141 (2013).
6. Ghosh K., Dixit P. D., Agozzino L., Dill K. A. "The maximum caliber variational principle
   for nonequilibria." *Annu. Rev. Phys. Chem.* **71**, 213–238 (2020).
7. Kish L. *Survey Sampling*. Wiley (1965).
8. Rényi A. "A characterization of Poisson processes." *Magyar Tud. Akad. Mat. Kutató Int.
   Közl.* **1**, 519–527 (1956).
9. Kalashnikov V. *Geometric Sums: Bounds for Rare Events with Applications*. Kluwer (1997).
10. Du R., Pande V. S., Grosberg A. Yu., Tanaka T., Shakhnovich E. I. "On the transition
    coordinate for protein folding." *J. Chem. Phys.* **108**, 334 (1998).
11. Bolhuis P. G., Chandler D., Dellago C., Geissler P. L. "Transition path sampling:
    throwing ropes over rough mountain passes, in the dark." *Annu. Rev. Phys. Chem.* **53**,
    291–318 (2002).
12. Kramers H. A. "Brownian motion in a field of force and the diffusion model of chemical
    reactions." *Physica* **7**, 284–304 (1940).
13. Hänggi P., Talkner P., Borkovec M. "Reaction-rate theory: fifty years after Kramers."
    *Rev. Mod. Phys.* **62**, 251–341 (1990).
14. Voter A. F. "Hyperdynamics: accelerated molecular dynamics of infrequent events."
    *Phys. Rev. Lett.* **78**, 3908 (1997).
15. Tiwary P., Parrinello M. "From metadynamics to dynamics." *Phys. Rev. Lett.* **111**,
    230602 (2013).
16. Salvalaglio M., Tiwary P., Parrinello M. "Assessing the reliability of the dynamics
    reconstructed from metadynamics." *J. Chem. Theory Comput.* **10**, 1420–1425 (2014).
17. Kaplan E. L., Meier P. "Nonparametric estimation from incomplete observations." *J. Am.
    Stat. Assoc.* **53**, 457–481 (1958).
18. Lilliefors H. W. "On the Kolmogorov–Smirnov test for the exponential distribution with
    mean unknown." *J. Am. Stat. Assoc.* **64**, 387–389 (1969).
19. Efron B. "Bootstrap methods: another look at the jackknife." *Ann. Stat.* **7**, 1–26
    (1979).
20. Dixit P. D., Dill K. A. "Caliber corrected Markov modeling (C2M2): correcting equilibrium
    Markov models." *J. Chem. Theory Comput.* **14**, 1111–1119 (2018).
21. Brotzakis Z. F., Vendruscolo M., Bolhuis P. G. "A method of incorporating rate constants
    as kinetic constraints in molecular dynamics simulations." *PNAS* **118**, e2012423118
    (2021).
22. Fersht A. R., Matouschek A., Serrano L. "The folding of an enzyme. I. Theory of protein
    engineering analysis of stability and pathway of protein folding." *J. Mol. Biol.*
    **224**, 771–782 (1992).
23. Hummer G. "From transition paths to transition states and rate coefficients." *J. Chem.
    Phys.* **120**, 516–523 (2004).
24. E W., Vanden-Eijnden E. "Towards a theory of transition paths." *J. Stat. Phys.* **123**,
    503–523 (2006).
25. Jackson S. E., Fersht A. R. "Folding of chymotrypsin inhibitor 2. 1. Evidence for a
    two-state transition." *Biochemistry* **30**, 10428–10435 (1991).
26. Best R. B., Hummer G. "Reaction coordinates and rates from transition paths." *PNAS*
    **102**, 6732–6737 (2005).
27. Hammond G. S. "A correlation of reaction rates." *J. Am. Chem. Soc.* **77**, 334–338
    (1955).
28. Leffler J. E. "Parameters for the description of transition states." *Science* **117**,
    340–341 (1953).
29. Donati L., Hartmann C., Keller B. G. "Girsanov reweighting for path ensembles and Markov
    state models." *J. Chem. Phys.* **146**, 244112 (2017).
30. Gilson M. K., Given J. A., Bush B. L., McCammon J. A. "The statistical-thermodynamic
    basis for computation of binding affinities: a critical review." *Biophys. J.* **72**,
    1047–1069 (1997).
