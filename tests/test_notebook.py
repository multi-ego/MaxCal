"""The demo notebook must run top to bottom (slow: pytest --runslow).
Executed in static mode (MAXCAL_DEMO_STATIC=1), exactly as the committed copy is rendered."""
import os

import pytest

pytestmark = pytest.mark.slow

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, os.pardir, "notebooks", "demo.ipynb")


def test_demo_notebook_executes(monkeypatch):
    monkeypatch.setenv("MAXCAL_DEMO_STATIC", "1")
    nb = nbformat.read(NB, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(NB)}}).execute()
    n_fig = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.outputs
                if "data" in o and "image/png" in o["data"])
    assert n_fig >= 10
