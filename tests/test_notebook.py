"""The demo notebooks must run top to bottom (slow: pytest --runslow).
Executed in static mode (MAXCAL_DEMO_STATIC=1), exactly as the committed copies are
rendered."""
import os

import pytest

pytestmark = pytest.mark.slow

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, os.pardir, "notebooks")


@pytest.mark.parametrize("name,min_figures", [("demo.ipynb", 10), ("target_demo.ipynb", 5)])
def test_demo_notebook_executes(monkeypatch, name, min_figures):
    monkeypatch.setenv("MAXCAL_DEMO_STATIC", "1")
    NB = os.path.join(NB_DIR, name)
    nb = nbformat.read(NB, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(NB)}}).execute()
    n_fig = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.outputs
                if "data" in o and "image/png" in o["data"])
    assert n_fig >= min_figures
