"""End-to-end test of the reweighting path (finite lambda*), on hand-built trajectories
where the weighted CV crosses 1 (see synthetic.py).  The Langevin regression data never
reach CV = 1, so without this test the finite-lambda* branch (bootstrap CI, N_eff,
KS at lambda*, TSE weights) would not be exercised end to end."""
import csv
import subprocess
import sys

import numpy as np
import pytest

import maxcal as m
from synthetic import write_reweight_set
from test_regtest import ROOT, check_nan_reasons

QU, QF, QTS = 0.3, 0.8, 0.45


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("reweight")
    specs = write_reweight_set(str(d / "data"))
    out = d / "out"
    r = subprocess.run([sys.executable, "-m", "maxcal.reweight", str(d / "data" / "s_*.xvg"),
                        "--qu", str(QU), "--qf", str(QF), "--qts", str(QTS), "--nqts", "1",
                        "--boot", "100", "--seed", "3", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    with open(out / "summary.csv") as fh:
        rows = {round(float(x["qts"]), 3): x for x in csv.DictReader(fh)}
    return out, rows, specs, d


def test_reweighting_succeeds(run):
    _, rows, _, _ = run
    r = rows[QTS]
    assert r["reweight_status"] == "ok" and float(r["reweight_ok"]) == 1
    lam, lo, hi = float(r["lam_star"]), float(r["lam_lo"]), float(r["lam_hi"])
    assert 0.5 < lam < 2.0
    assert lo <= lam <= hi                       # bootstrap CI brackets the estimate
    assert float(r["neff_star"]) >= 30
    assert 0.0 < float(r["ks_p_star"]) <= 1.0


def test_weighted_cv_is_one_at_lambda_star(run):
    _, rows, _, d = run
    lam = float(rows[QTS]["lam_star"])
    trajs = []
    for path in sorted((d / "data").glob("s_*.xvg")):
        q, dt = m.load_traj(str(path), None)
        fold, fails = m.parse_traj(q, QU, QF)
        trajs.append(dict(path=str(path), q=q, dt=dt, n=q.size, fold=fold, fails=fails))
    A = m.attempts(trajs, QTS, 0.0, False)
    cv, _ = m.cv_curve(np.array([lam]), A["k"], A["T"], A["done"],
                       m.success_prob(A["k"], A["done"]))
    assert cv[0] == pytest.approx(1.0, abs=0.02)   # grid interpolation error only


def test_tse_weights_follow_the_tilt(run):
    out, rows, _, _ = run
    lam = float(rows[QTS]["lam_star"])
    with open(next(out.glob(f"weights_qts{QTS:.3f}.csv"))) as fh:
        tse = list(csv.DictReader(fh))
    w = np.array([float(t["weight_at_lambda_star"]) for t in tse])
    k = np.array([int(t["k_failed"]) for t in tse])
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    p = rows[QTS]["p"]
    c = 1.0 / (1.0 - float(p) * (1.0 - np.exp(-lam)))
    assert w[k == 3].mean() / w[k == 0].mean() == pytest.approx(c ** 3, rel=1e-3)


def test_statuses_explain_nans_beyond_attempts(run):
    _, rows, _, _ = run
    r = rows[0.55]                               # no excursion reaches 0.55: no attempts
    assert r["geom_status"] == "too_few_bins"
    assert r["lag1_status"] == "too_few_pairs"
    assert r["reweight_status"] == "no_root"
    assert rows[QTS]["lag1_status"] == "constant_cycles"   # identical cycles by construction
    for row in rows.values():
        check_nan_reasons(row)
