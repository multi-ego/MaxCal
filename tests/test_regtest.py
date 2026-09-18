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
# reason codes for NaNs: must match exactly
STATUS = ["geom_status", "lag1_status", "reweight_status"]
# stitch_status depends on the stochastic pass/fail near the threshold only through
# "ok" vs "no_pass"; pool availability is deterministic
POOL_STATUS = "stitch_status"


def check_nan_reasons(row):
    """Every NaN in a summary row must come with a non-ok reason code."""
    nan = lambda c: row[c] in ("", "nan")
    assert nan("geom_p") == (row["geom_status"] != "ok"), row
    assert nan("lag1_rho") == (row["lag1_status"] != "ok"), row
    if nan("lam_star"):
        assert row["reweight_status"] == "no_root", row
    if nan("lam_stitch"):
        assert row["stitch_status"] in ("empty_pool", "no_pass"), row


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
    for f in ["summary.csv", "survival.png", "lambda_scan.png", "robustness_qts.png",
              "committor.csv", "ts_location.csv", "ts_location.png"]:
        assert (out / f).exists(), f
    assert list(out.glob("tse_frames_qts*.csv"))
    assert list(out.glob("tse_committor_lam*_q0.50.csv"))
    com = read(out / "committor.csv")
    iso = np.array([as_float(r["q_model_isotonic"]) for r in com])
    iso = iso[np.isfinite(iso)]
    assert np.all(np.diff(iso) >= 0) and 0 <= iso.min() and iso.max() <= 1
    ts = read(out / "ts_location.csv")
    # default assumption (barrier on the model TS): the TS does not move with lambda
    at_default = [float(r["Q_TS"]) for r in ts if float(r["q_star"]) == 0.5]
    assert np.ptp(at_default) < 1e-9


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
    for r in rows:
        check_nan_reasons(r)
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
        for c in STATUS:
            assert g[c] == e[c], f"{c} @ Q‡={e['qts']}: {g[c]} != {e[c]}"
        assert (g[POOL_STATUS] == "empty_pool") == (e[POOL_STATUS] == "empty_pool")
        for c, tol in LOOSE.items():
            np.testing.assert_allclose(as_float(g[c]), as_float(e[c]), atol=tol,
                                       equal_nan=True, err_msg=f"{c} @ Q‡={e['qts']}")
