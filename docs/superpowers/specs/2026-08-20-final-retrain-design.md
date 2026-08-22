# Final Retrain Collection and Training Design

## Goal

Collect fresh UE5 expert episodes without overwriting prior data, train the perception and decision models from that isolated dataset, and preserve evidence needed for a later real Jetson CUDA closed-loop validation.

## Scope

- Add one new PowerShell collection entry point for the final dataset.
- Add one new Python training entry point that accepts explicit data and experiment paths.
- Keep existing collection scripts, datasets, and experiments unchanged.
- Keep the four-node ROS architecture unchanged.

## Data flow

UE5 expert motion -> Jetson bridge -> rosbag (`/ue/asv_state`, `/ue/camera_frame`, `/ue/entities`) -> synchronized episode export -> `data/episodes/final_retrain` -> training -> `experiments/final_retrain`.

The collector uses a 215-second UE5 runtime so that startup and warmup overhead still leave at least 179 seconds of recorded data. Remote processes are managed by explicit PID files and are stopped in `finally` cleanup.

## Safety boundaries

- The destination directories must be absent or empty; the collector refuses an existing non-empty destination.
- No old directory is deleted or overwritten.
- Jetson image bytes are exported directly from `CameraFrame.data`; no exposure, gamma, contrast, or brightness operation is introduced.
- Expert collection uses UE5 executed pose deltas as labels. It is distinct from final policy-driven validation.
- CPU fallback is not enabled.

## Training

The new training entry point reuses the existing `train_final` implementation and receives `--data-root`, `--run-root`, and `--embedding-path`. It writes only the requested run directory and generates `perception.npz`, `policy.pt`, `metrics.json`, and `tracking_result.png` there.

## Acceptance

- Contract tests pass.
- A single fresh trial exports a synchronized episode with one run identity and at least 179 seconds of usable data.
- Full collection creates the declared train/validation/test slots without touching old directories.
- Training completes in the new experiment directory and produces both model files and the result figure.
- Deployment and final three-scene validation remain separate, evidence-based steps.
