"""Regression test for maxcal-target on a seeded 12-system (PDZ2-like) dataset.

    MAXCAL_UPDATE_REF=1 pytest tests/test_target_regtest.py     # regenerate the reference
"""
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from synthetic import write_rate_dataset
from test_regtest import ROOT, REF, read, as_float

# everything here is deterministic given the seeded times: compare tightly
EXACT = ["needed_factor", "tau_model", "tau_target", "N_eff", "weighted_mean", "ks_D",
         "max_weight", "tail_gap"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("target_regtest")
    systems = write_rate_dataset(str(d))
    out = d / "out"
    r = subprocess.run([sys.executable, "-m", "maxcal.target", systems, "--conc", "0.017",
                        "--clock", "balanced", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return out, r.stdout


def test_outputs_and_sanity(run):
    out, stdout = run
    for f in ["summary.csv", "calibration.png", "cdf_bind.png", "cdf_unbind.png"]:
        assert (out / f).exists(), f
    rows = read(out / "summary.csv")
    assert len(rows) == 24                                   # 12 systems x 2 directions
    for r in rows:
        assert int(r["N"]) == 50
        assert 0 < float(r["N_eff"]) <= 50
        # the weights hit the experimental mean exactly, and match the exponential shape
        assert float(r["weighted_mean"]) == pytest.approx(float(r["tau_target"]), rel=1e-5)   # CSV is written with %.6g
        assert float(r["ks_D"]) < 0.2
    factors = np.array([float(r["needed_factor"]) for r in rows])
    assert 0.2 < factors.min() and factors.max() < 2.0       # a mild, reachable correction
    assert np.all(np.array([float(r["N_eff"]) for r in rows]) > 20)
    assert "clock factor" in stdout


def test_clock_is_consistent_across_systems(run):
    """One global clock: the needed factors are k_model/k_exp divided by a single c,
    so the spread of their logarithms is the model's own per-system error."""
    out, _ = run
    rows = read(out / "summary.csv")
    lf = np.log([float(r["needed_factor"]) for r in rows])
    assert abs(lf.mean()) < 0.6 and lf.std() < 0.6


def test_against_reference(run):
    out, _ = run
    new = os.path.join(out, "summary.csv")
    ref = os.path.join(REF, "reference_target.csv")
    if os.environ.get("MAXCAL_UPDATE_REF"):
        os.makedirs(REF, exist_ok=True)
        shutil.copy(new, ref)
        pytest.skip("reference updated")
    assert os.path.exists(ref), "no reference; run with MAXCAL_UPDATE_REF=1"
    got, exp = read(new), read(ref)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        assert (g["system"], g["direction"]) == (e["system"], e["direction"])
        for c in EXACT:
            np.testing.assert_allclose(as_float(g[c]), as_float(e[c]), rtol=1e-5, atol=1e-12,
                                       equal_nan=True, err_msg=f"{c} @ {e['system']} {e['direction']}")
