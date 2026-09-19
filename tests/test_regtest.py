"""Regression test: the three command-line tools on a seeded Langevin data set, compared
to stored reference summaries.  Regenerate after an intended change with

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
ROOT = os.path.join(HERE, os.pardir)
REF = os.path.join(HERE, "regtest")
COMMON = ["--qu", "0.3", "--qf", "0.8", "--qts", "0.40", "--nqts", "5", "--seed", "11"]

# deterministic columns (data + fixed-seed nulls): tight tolerance
EXACT = {"stitch": ["qts", "N", "n_done", "sum_k", "kmax", "frac_k0", "p", "geom_p",
                    "cv0", "ks_p0"],
         "reweight": ["qts", "N", "sum_k", "p", "cv0", "cv_lammax", "ks_p0", "lam_star",
                      "neff_star"]}
LOOSE = {"stitch": {"cv_stitch0": 0.08, "lam_stitch": 0.5}, "reweight": {}}
STATUS = ["geom_status", "lag1_status"]


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def as_float(v):
    return float(v) if v not in ("", "nan") else np.nan


def check_nan_reasons(row):
    """Every NaN in a summary row must come with a non-ok reason code."""
    nan = lambda c: row.get(c, "") in ("", "nan")
    assert nan("geom_p") == (row["geom_status"] != "ok"), row
    assert nan("lag1_rho") == (row["lag1_status"] != "ok"), row
    if "reweight_status" in row and nan("lam_star"):
        assert row["reweight_status"] == "no_root", row
    if "stitch_status" in row and nan("lam_stitch"):
        assert row["stitch_status"] in ("empty_pool", "no_pass"), row


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("regtest")
    data = str(d / "data")
    write_dataset(data, n=100, seed=2024)
    pattern = os.path.join(data, "q_*.xvg")
    out = {}
    for tool, extra in (("stitching", ["--nstitch", "2000", "--nsub", "100"]),
                        ("reweight", ["--boot", "50"]),
                        ("committor", [])):
        o = d / tool
        r = subprocess.run([sys.executable, "-m", f"maxcal.{tool}", pattern,
                            "--out", str(o)] + COMMON + extra,
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        out[tool] = o
    return out


def test_outputs_exist(run):
    for tool, files in (("stitching", ["summary.csv", "survival.png", "robustness_qts.png",
                                       "heatmap.csv", "heatmap.png"]),
                        ("reweight", ["summary.csv", "survival.png", "lambda_scan.png"]),
                        ("committor", ["committor.csv", "ts_location.csv", "ts_location.png"])):
        for f in files:
            assert (run[tool] / f).exists(), f"{tool}/{f}"
    assert list(run["reweight"].glob("weights_qts*.csv"))
    assert list(run["reweight"].glob("tse_frames_qts*.csv"))
    assert list(run["committor"].glob("tse_committor_lam*_q0.50.csv"))


def test_physics_sanity(run):
    rows = read(run["stitching"] / "summary.csv")
    for r in rows:
        assert int(r["N"]) == 100 and int(r["n_done"]) == 100      # all fold, none censored
        assert 0.0 < float(r["p"]) <= 1.0
        check_nan_reasons(r)
    assert np.all(np.diff([float(r["p"]) for r in rows]) > 0), "p must grow with Q‡"
    for r in read(run["reweight"] / "summary.csv"):
        check_nan_reasons(r)

    com = read(run["committor"] / "committor.csv")
    iso = np.array([as_float(r["q_model_isotonic"]) for r in com])
    iso = iso[np.isfinite(iso)]
    assert np.all(np.diff(iso) >= 0) and 0 <= iso.min() and iso.max() <= 1
    ts = read(run["committor"] / "ts_location.csv")
    at_default = [float(r["Q_TS"]) for r in ts if float(r["q_star"]) == 0.5]
    assert np.ptp(at_default) < 1e-9        # barrier on the model TS: TS does not move

    hm = read(run["stitching"] / "heatmap.csv")
    assert len(hm) == 6 * 21                                        # 5 grid values + --qts
    ksp = np.array([as_float(r["median_ks_p"]) for r in hm])
    assert np.all((ksp[np.isfinite(ksp)] > 0) & (ksp[np.isfinite(ksp)] <= 1))
    valid = sorted({float(r["qts"]) for r in hm if r["qts_valid"] == "1"})
    assert valid and max(valid) < 0.56

    tse = read(next(run["reweight"].glob("tse_frames_qts*.csv")))
    assert len(tse) == 100
    assert sum(float(t["weight_at_lambda_star"]) for t in tse) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("tool", ["stitching", "reweight"])
def test_against_reference(run, tool):
    new = os.path.join(run[tool], "summary.csv")
    ref = os.path.join(REF, f"reference_{tool}.csv")
    if os.environ.get("MAXCAL_UPDATE_REF"):
        os.makedirs(REF, exist_ok=True)
        shutil.copy(new, ref)
        pytest.skip("reference updated")
    assert os.path.exists(ref), f"no reference for {tool}; run with MAXCAL_UPDATE_REF=1"
    got, exp = read(new), read(ref)
    assert len(got) == len(exp)
    key = "stitch" if tool == "stitching" else "reweight"
    for g, e in zip(got, exp):
        for c in EXACT[key]:
            np.testing.assert_allclose(as_float(g[c]), as_float(e[c]), rtol=1e-5, atol=1e-9,
                                       equal_nan=True, err_msg=f"{c} @ Q‡={e['qts']}")
        for c in STATUS:
            assert g[c] == e[c], f"{c} @ Q‡={e['qts']}"
        for c, tol in LOOSE[key].items():
            np.testing.assert_allclose(as_float(g[c]), as_float(e[c]), atol=tol,
                                       equal_nan=True, err_msg=f"{c} @ Q‡={e['qts']}")
