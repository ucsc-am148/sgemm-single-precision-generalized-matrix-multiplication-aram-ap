"""Run sanity_check.py on a Modal A100 40GB — same hardware the autograder
uses. Lets you iterate on your kernels without burning graded submissions.

One-time setup:
    pip install modal
    modal token new            # browser auth, links your Modal account

Run (from the repo root, after editing kernels.py):
    modal run modal_sanity.py

Output streams back to your terminal. Each run costs about $0.01 of the
$30/month free credit Modal gives every account.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "numpy==2.4.4",
        "numba==0.65.0",
        "numba-cuda==0.30.0",
        "cupy-cuda12x==14.0.1",
    )
    .env({"NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS": "0"})
    # add_local_* must be the LAST build steps.
    .add_local_file(str(HERE / "kernels.py"),       "/app/kernels.py")
    .add_local_file(str(HERE / "sanity_check.py"),  "/app/sanity_check.py")
    .add_local_file(str(HERE / "kernel6_stretch.cu"), "/app/kernel6_stretch.cu")
)

app = modal.App("sgemm-student-sanity", image=image)


@app.function(gpu="A100-40GB", timeout=300)
def run_sanity():
    import sys
    sys.path.insert(0, "/app")
    import sanity_check
    sanity_check.main()


@app.local_entrypoint()
def main():
    run_sanity.remote()
