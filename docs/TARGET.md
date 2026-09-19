# maxcal-target: matching experimental rates with a Poisson target

Reweighting first-passage trajectories so that they reproduce **measured rate constants**
and a **single-exponential (Poisson) shape**, with minimum perturbation. The weights are
the result: applied to structural observables, they show which trajectories — and which
mechanisms — the experimental kinetics favour.

This is the companion of the barrier correction in the main [README](../README.md): there
the missing barrier is unknown and inferred from the shape of the folding-time
distribution; here the rates are known from experiment and the question is what the model
must look like to reproduce them.

---

## 1. When to use it

Suitable when all three hold:

- **Experimental rate constants** are available, ideally for several related systems
  (a mutant series), so the model's systematic time-scale error can be separated from its
  per-system error.
- **The model is trustworthy structurally** but quantitatively off in kinetics, e.g. a
  coarse-grained or structure-based model with a smoothed landscape.
- **The needed correction is mild**, within a factor of ~2 per system after the global
  time-scale factor. Larger corrections are not reachable by reweighting, which is itself
  informative (§6).

Not suitable for absolute-rate prediction: the time-scale factor is fitted, not derived.

---

## 2. Setup and notation

For each system and direction (binding/unbinding, or folding/unfolding) the model provides
N independent first-passage times t₁ … t_N, drawn from its own density f_model. Experiment
provides a rate constant k_exp. The corresponding target is a Poisson process:

$$
f_\text{target}(t) = \frac{1}{\tau_\text{target}} e^{-t/\tau_\text{target}}
$$

with τ_target the experimental mean first-passage time expressed in model time units (§3).
For binding at simulation concentration [C] (one ligand in the box, pseudo-first-order),
k_on = 1/(τ_bind·[C]); for unbinding, k_off = 1/τ_unbind.

---

## 3. Time-scale calibration

Model and experimental times are not directly comparable: a simplified model with a
smoother landscape crosses barriers faster, often by orders of magnitude. Define, per
system s and direction d,

$$
A_{s,d} = k^\text{model}_{s,d} / k^\text{exp}_{s,d}
$$

If the model's acceleration were a pure clock rescaling, all A would be equal. They are
not, and the deviations are the model's per-system error. Fit one global factor c and keep
the residuals:

$$
f_{s,d} = A_{s,d} / c
$$

f is the factor by which the mean first-passage time must change. Only the f are physical:
c cancels in every ratio, so relative rates, Φ-values and affinities do not depend on it.

**Choosing c matters** because of the asymmetric cost in §5: slowing a sample down is
cheap, speeding it up is expensive and impossible beyond a factor of 2.

| `--clock` | c | effect |
|---|---|---|
| `geometric` | geometric mean of A | the natural estimate; some systems may need lengthening |
| `decrease-only` | max A | every system needs a slowdown (f ≤ 1) |
| `balanced` (default) | maximises min(2f − f²) | best worst-case effective sample size |
| a number | as given | e.g. a clock calibrated independently |

The spread of ln f is a useful summary: it is the model's own per-system kinetic error,
free of the clock.

---

## 4. The weights

### 4.1 Minimum-perturbation reweighting onto a specified target

The target is fully specified, so this is a maximum-entropy (minimum relative entropy)
problem with the whole marginal distribution of t constrained. Minimising the
Kullback–Leibler divergence D(P′‖P_model) over path measures P′ whose marginal in t is
f_target gives

$$
P'(\omega) = P_\text{model}(\omega) \frac{f_\text{target}(t(\omega))}{f_\text{model}(t(\omega))}
$$

so the weights are the importance ratio,

$$
w_i \propto f_\text{target}(t_i) / f_\text{model}(t_i)
$$

Two consequences:

- **Conditional structure is preserved.** P′(ω | t) = P_model(ω | t): given the
  first-passage time, everything the trajectory does is the model's. The correction acts
  only on how much each trajectory counts, never on what it contains.
- **Any observable can be reweighted.** For a per-trajectory observable O,
  ⟨O⟩_target ≈ Σ wᵢOᵢ. Note the correction only reaches the part of the model's error that
  is correlated with the first-passage time (§6).

### 4.2 Estimating f_model

f_model is estimated by a gamma fitted to the times by maximum likelihood: a smooth,
two-parameter stand-in flexible enough for first-passage densities (exponential and
lag-shifted shapes alike). Smoothness matters: a nearest-neighbour (spacing) estimate makes
the weights fluctuate with the random gaps between sorted times, wasting N_eff even when
the model already matches the target. With the gamma fit, the weights are flat when nothing
needs changing, and N_eff ≈ N.

A final exponential tilt matches the target mean exactly, since the experimental rate is
the primary constraint and the sample cannot represent target mass beyond its longest
observed time (§5, `tail_gap`).

### 4.3 Mean-only alternative (`--mode mean`)

Constraining only the mean, ⟨t⟩ = τ_target, gives the classic one-parameter maximum-entropy
solution

$$
w_i \propto e^{-\theta t_i}
$$

with θ from the constraint. This leaves the shape to the model instead of imposing a
Poisson one; comparing the two modes shows how much of the correction is rate and how much
is shape.

---

## 5. What the correction costs

With Kish's effective sample size, N_eff = (Σw)²/Σw², the cost of an exponential tilt on an
exponential sample can be computed in closed form. Writing the achieved mean as a factor f
of the original, θτ = 1/f − 1, and

$$
N_\text{eff}/N = \frac{(E[w])^2}{E[w^2]} = 2f - f^2
$$

independent of N. Hence:

- **f = 1:** no cost.
- **f = 0.5** (twice as fast): 75% retained.
- **f = 0.2:** 36% retained.
- **f → 2⁻:** N_eff → 0. **Lengthening beyond a factor of 2 is impossible**, because the
  second moment of the tilted exponential diverges.

Three diagnostics accompany every system and direction:

| diagnostic | meaning |
|---|---|
| `N_eff`, `N_eff_expected` | achieved vs the analytic 2f − f² prediction |
| `max_weight` | largest single weight; a value near 1 means one trajectory dominates |
| `tail_gap` = exp(−t_max/τ_target) | target probability beyond the longest observed time: what the sample cannot represent |

In `target` mode the KS statistic against the target is satisfied by construction and is
reported only for completeness; read N_eff, `max_weight` and `tail_gap` instead.

---

## 6. What it can and cannot do

- **It cannot invent mechanisms.** Weights redistribute probability over trajectories the
  model produced. If the experimental kinetics require behaviour the model never samples,
  N_eff collapses — a result about the model, not a failure of the analysis.
- **It corrects only the t-correlated part of the error.** An observable independent of the
  first-passage time is unchanged by reweighting. That is the correct minimal-assumption
  behaviour, not a limitation to hide: the kinetics constrain what they constrain.
- **The clock is assumed shared** across systems. Check the spread of ln f; a mutation that
  changes the model's effective friction would break this.
- **Affinities follow automatically.** Matching both directions fixes
  K_D = k_off/k_on per system, so an affinity comparison after reweighting is a
  consistency check, not an independent test.
- **Reweighting is per system.** Cross-system statements come from comparing separately
  reweighted ensembles, not from reweighting one system onto another's data.

---

## 7. Usage

```bash
maxcal-target systems.csv --conc 0.017 --clock balanced --out target_out
```

`systems.csv`:

```
system,kon_exp,koff_exp,bind_times,unbind_times
EQVTAV_WT,2.6,22,times/EQVTAV_WT_bind.dat,times/EQVTAV_WT_unbind.dat
EQVTAV_L18A,2.4,10.4,times/EQVTAV_L18A_bind.dat,times/EQVTAV_L18A_unbind.dat
```

- `kon_exp` in µM⁻¹s⁻¹, `koff_exp` in s⁻¹; `--conc` is the ligand concentration in the
  simulation box, in M. For a first-order process (folding/unfolding) use
  `--system-kind first-order` and give both rates in s⁻¹.
- Each times file holds one first-passage time per line, in the model's own time units, in
  trajectory order.

| option | meaning |
|---|---|
| `--clock` | `geometric`, `decrease-only`, `balanced` (default) or a number (§3) |
| `--mode` | `target` (rate + Poisson shape, default) or `mean` (rate only) |
| `--conc`, `--system-kind` | concentration and whether binding is pseudo-first-order |

**Outputs**

| file | content |
|---|---|
| `summary.csv` | per system and direction: τ_model, τ_target, needed factor, N_eff and its expectation, weighted mean, KS, max weight, tail gap |
| `weights_<system>_<direction>.csv` | trajectory index, time, weight — join this to your structural analysis |
| `calibration.png` | needed factors and N_eff per system |
| `cdf_bind.png`, `cdf_unbind.png` | model, reweighted and target CDFs |

---

## 8. Worked example

`tests/synthetic.py: write_rate_dataset` builds twelve PDZ2-like systems (two peptides ×
WT plus five mutants) from published experimental k_on/k_off and model rates, with seeded
exponential times in place of the real per-trajectory data:

```bash
python -c "import sys; sys.path.insert(0,'tests'); import synthetic as S; print(S.write_rate_dataset('example_data'))"
maxcal-target example_data/systems.csv --conc 0.017 --clock balanced --out example_out
```

Fitted clock: **c ≈ 6 × 10⁻⁸** (model time × c = experimental time), i.e. the model is about
10⁷–10⁸ times faster in these units. What remains per system:

| system | f (binding) | N_eff | f (unbinding) | N_eff |
|---|---|---|---|---|
| EQVTAV WT | 0.97 | 50 | 0.84 | 48 |
| EQVTAV L18A | 0.79 | 47 | 1.13 | 48 |
| EQVTAV L25A | 1.66 | 38 | 0.59 | 43 |
| EQVTAV T35G | 0.62 | 43 | 0.34 | 32 |
| EQVTAV V44A | 0.69 | 45 | 0.66 | 43 |
| EQVTAV A53G | 0.84 | 49 | 0.64 | 45 |
| EQVSAV WT | 0.65 | 44 | 1.49 | 43 |
| EQVSAV L18A | 0.80 | 46 | 1.34 | 45 |
| EQVSAV L25A | 0.95 | 47 | 1.43 | 38 |
| EQVSAV T35G | 0.71 | 47 | 0.48 | 38 |
| EQVSAV V44A | 0.50 | 38 | 0.78 | 47 |
| EQVSAV A53G | 0.56 | 40 | 0.51 | 39 |

Every system is within a factor of ~1.7 of experiment once the clock is removed, and the
worst effective sample size is 32 of 50. The experimental kinetics are therefore reachable
*within* the mechanisms the model already samples. The affinity errors, mean |log₁₀ K_D|
0.19 for the model, go to 0 by construction after matching both directions.

**Using the weights.** For a per-trajectory descriptor O (encounter-complex lifetime, which
contacts form first, how much of a mutated side-chain contact is present at the
transition):

```python
import csv, numpy as np
rows = list(csv.DictReader(open("example_out/weights_EQVTAV_L25A_bind.csv")))
w = np.array([float(r["weight"]) for r in rows])       # trajectory order = input order
O = my_descriptor_per_trajectory                        # your own analysis
print("model:", O.mean(), " reweighted:", np.sum(w * O))
```

Bootstrap over trajectories for error bars, and report N_eff alongside: a large shift with
N_eff ≈ 40 of 50 is meaningful, the same shift with N_eff = 5 is one trajectory talking.

`notebooks/target_demo.ipynb` runs this example interactively: clock calibration, the
weights per system, the N_eff cost curve, a descriptor example and the affinity scatter.

---

## 9. Tests

- `tests/test_target.py` — the weights reproduce the target mean and CDF; they are flat
  when the target equals the model; `--mode mean` is an exact exponential tilt; the clock
  modes; the N_eff formula; recovery of a descriptor's target average from a mis-shaped
  model sample; and an end-to-end run whose needed factors scale with 1/c.
- `tests/test_target_regtest.py` — the twelve-system example above against
  `tests/regtest/reference_target.csv`, with sanity checks on the weighted means, the
  reachability of the factors and N_eff.

---

## References

1. Jaynes E. T. "Information theory and statistical mechanics." *Phys. Rev.* **106**,
   620–630 (1957). (Maximum entropy.)
2. Pressé S., Ghosh K., Lee J., Dill K. A. "Principles of maximum entropy and maximum
   caliber in statistical physics." *Rev. Mod. Phys.* **85**, 1115–1141 (2013).
3. Cover T. M., Thomas J. A. *Elements of Information Theory*, 2nd ed. Wiley (2006).
   (Minimum relative entropy and exponential tilting.)
4. Torrie G. M., Valleau J. P. "Nonphysical sampling distributions in Monte Carlo
   free-energy estimation: umbrella sampling." *J. Comput. Phys.* **23**, 187–199 (1977).
   (Importance reweighting of simulation ensembles.)
5. Ferrenberg A. M., Swendsen R. H. "New Monte Carlo technique for studying phase
   transitions." *Phys. Rev. Lett.* **61**, 2635–2638 (1988).
6. Kish L. *Survey Sampling*. Wiley (1965). (Effective sample size.)
7. Martino L., Elvira V., Louzada F. "Effective sample size for importance sampling based
   on discrepancy measures." *Signal Processing* **131**, 386–401 (2017).
8. Efron B. "Bootstrap methods: another look at the jackknife." *Ann. Stat.* **7**, 1–26
   (1979).
9. Lilliefors H. W. "On the Kolmogorov–Smirnov test for the exponential distribution with
   mean unknown." *J. Am. Stat. Assoc.* **64**, 387–389 (1969).
10. Gianni S., Haq S. R., Montemiglio L. C., Jürgens M. C., Engström Å., Chi C. N.,
    Brunori M., Jemth P. "Sequence-specific long range networks in PSD-95/Discs
    Large/ZO-1 (PDZ) domains tune their binding selectivity." *J. Biol. Chem.* **286**,
    27167–27175 (2011). (Source of the experimental rates used in the example.)
11. Scalone E. *et al.* "Multi-eGO: an in silico lens to look into protein aggregation
    kinetics at atomic resolution." *PNAS* **119**, e2203181119 (2022).
12. Fersht A. R., Matouschek A., Serrano L. "The folding of an enzyme. I. Theory of protein
    engineering analysis of stability and pathway of protein folding." *J. Mol. Biol.*
    **224**, 771–782 (1992). (Φ-value analysis, for interpreting reweighted ensembles
    across a mutant series.)
