"""Regression test: full CLI run on a seeded Langevin data set, compared to a stored
reference summary.  To regenerate the reference after an intended change:

    MAXCAL_UPDATE_REF=1 pytest tests/test_regtest.py
"""
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from langevin import write_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, os.pardir, "maxcal_poisson.py")
REF = os.path.join(HERE, "regtest", "reference_summary.csv")
CLI = ["--qu", "0.3", "--qf", "0.8", "--nqts", "5", "--boot", "50",
       "--nstitch", "2000", "--nsub", "100", "--seed", "11"]

# deterministic columns (data + fixed-seed null): tight tolerance
EXACT = ["qts", "N", "n_done", "sum_k", "frac_k0", "p", "geom_p", "cv0", "ks_p0",
         "lam_star", "neff_star", "cv_lammax", "kmax"]
# stochastic but seeded columns: loose tolerance, robust to RNG-algorithm changes
LOOSE = {"cv_stitch0": 0.08, "lam_stitch": 0.5}


def read(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows


def as_float(v):
    return float(v) if v not in ("", "nan") else np.nan


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("regtest")
    write_dataset(str(d / "data"), n=100, seed=2024)
    out = d / "out"
    r = subprocess.run([sys.executable, SCRIPT, str(d / "data" / "q_*.xvg"),
                        "--out", str(out)] + CLI, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out, r.stdout


def test_outputs_exist(run):
    out, _ = run
    for f in ["summary.csv", "survival.png", "lambda_scan.png", "robustness_qts.png"]:
        assert (out / f).exists(), f
    assert list(out.glob("tse_frames_qts*.csv"))


def test_physics_sanity(run):
    out, stdout = run
    rows = read(out / "summary.csv")
    for r in rows:
        assert int(r["N"]) == 100 and int(r["n_done"]) == 100     # all fold, none censored
        assert 0.0 < float(r["p"]) <= 1.0
    p = [float(r["p"]) for r in rows]
    assert np.all(np.diff(p) > 0), "success probability must grow as Q‡ approaches Qf"
    ref = rows[len(rows) // 2]
    assert np.isfinite(as_float(ref["lam_stitch"])), "stitching must reach Poisson statistics"
    tse = read(next(out.glob("tse_frames_qts*.csv")))
    assert len(tse) == 100
    assert sum(float(t["weight_at_lambda_star"]) for t in tse) == pytest.approx(1.0, abs=1e-4)


def test_against_reference(run):
    out, _ = run
    new = os.path.join(out, "summary.csv")
    if os.environ.get("MAXCAL_UPDATE_REF"):
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        shutil.copy(new, REF)
        pytest.skip("reference updated")
    assert os.path.exists(REF), "no reference; run with MAXCAL_UPDATE_REF=1"
    got, exp = read(new), read(REF)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        for c in EXACT:
            np.testing.assert_allclose(as_float(g[c]), as_float(e[c]), rtol=1e-5,
                                       atol=1e-9, equal_nan=True, err_msg=f"{c} @ Q‡={e['qts']}")
        for c, tol in LOOSE.items():
            np.testing.assert_allclose(as_float(g[c]), as_float(e[c]), atol=tol,
                                       equal_nan=True, err_msg=f"{c} @ Q‡={e['qts']}")
