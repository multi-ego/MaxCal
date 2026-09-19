"""maxcal - MaxCal correction of an underestimated barrier in first-passage trajectories.

Executables (each a module with a main()):
  maxcal-stitch     (maxcal.stitching) thinning barrier crossings -> Poisson kinetics, lambda_min, decision map
  maxcal-reweight   MaxCal weights on whole trajectories (memory-safe, no stitching)
  maxcal-committor  model/corrected committor, TS location, committor-based TSE frames
  maxcal-joint      forward + backward runs: kinetic DeltaG, TP symmetry, coupled tilt
  maxcal-target     reweighting onto experimental rates with a Poisson target

Importing the package re-exports the shared functions, so `import maxcal as m` gives
m.attempts, m.stitch, m.cv_curve, m.ts_location, ... as in the original single module.
"""
from .core import (load_traj, parse_traj, attempts, success_prob, log_weights, tilted_p,
                   cv_curve, first_crossing, weighted_km, ks_exp_weighted, null_D, D_rows,
                   lilliefors_p, geometric_test, geom_status, lag1_status, reweight_status,
                   stitch_status, add_common_args, load_trajectories, qts_grid, write_csv)
from .stitching import stitch, stitch_scan, ks_heatmap
from .committor import (committor_frames, isotonic, committor_profile, corrected_committor,
                        q_at, ts_location)

__version__ = "0.2.0"
