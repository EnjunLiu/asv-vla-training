# ASV VLA Training

This package contains the offline PC data registry, split, feature-cache, and
single-point policy training tools. It consumes recorded images, offline
`Entities` supervision, and real-ASV ego state; it does not start ROS or send
UE5 control commands. `Entities` remain privileged offline labels only.

The importable packages are `asv_training` and the PC-executed pure algorithm
contracts in `asv_vla`, both under `src`. ROS nodes remain in the independent
Jetson workspace. The active config entry points are kept under `configs`:

- `model_small_v3.yaml` defines the current policy contract.
- `sine_near_collection_plan_v1.json` defines the current collection plan.

Older experiment snapshots are retained under
`configs/research_archive/` for research traceability. They are not
part of the active workflow and are not selected by default commands. Tests
are in `tests`, and model/data files remain external to this
repository. Install the package in a PC Python environment with
`pip install -e '.[test]'`, or run in place:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
python -m pytest -q -p no:cacheprovider tests
```

Training outputs must use a new external directory. The deployment boundary is
the two-dimensional body-frame desired displacement `[desired_x, desired_y]`;
this package never emits direct thruster commands. CUDA/Torch availability is
environment-specific and is not represented by the source import alone.
