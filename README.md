# ASV VLA Training

PC-side training and offline evaluation for the ASV HIL stack. Consumes recorded episodes (images, offline entity labels, ego state), trains perception / policy artifacts, and scores candidates before Jetson deployment.

Part of [asv-hil-platform](https://github.com/EnjunLiu/asv-hil-platform). Runtime inference lives in [`asv-jetson-ws`](https://github.com/EnjunLiu/asv-jetson-ws); the UE5 scene is [`asv-unreal-simulation`](https://github.com/EnjunLiu/asv-unreal-simulation).

## Layout

| Path | Role |
| --- | --- |
| `src/train_final.py` | Main training entry |
| `src/perception.py` / `src/decision.py` | Perception and policy contracts shared with Jetson |
| `scripts/` | Dataset / export helpers |
| `tests/` | Contract and workflow tests |
| `experiments/` | Candidate run dirs (weights ignored; keep scripts / notes) |
| `data/` | Local episodes and embeddings (**not committed**) |

Active closed-loop candidate (local): `experiments/chase_standoff_candidate`.

## Contract

- Policy output: body-frame desired displacement `[desired_x, desired_y]` in meters
- Never emit thruster PWM from this package
- `/ue/entities` (or equivalent) is offline supervision only—not online privileged truth
- Language-conditioned standoff uses the same embedding space as Jetson (`qwen_final_embeddings.npz` locally)

## Usage

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
python -m pytest -q -p no:cacheprovider tests
python run_training.py
# or a candidate-specific entry, e.g.
python train_chase_standoff.py
```

Weights (`.pt`, `.npz`, …), episode dumps under `data/`, and large caches stay outside git. Deploy accepted artifacts to Jetson `models/` with a matching `manifest.yaml`.

## Related

Portfolio overview and HIL diagram: [asv-hil-platform](https://github.com/EnjunLiu/asv-hil-platform).
