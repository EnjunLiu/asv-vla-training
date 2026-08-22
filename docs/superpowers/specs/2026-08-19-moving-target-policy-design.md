# Moving-Target Policy Retraining Design

## Goal

Train and validate a policy that follows a moving red or blue target at a
requested 3 m or 4 m standoff. Completion requires a real UE5 180 s trajectory
plot comparable to `track_world_single_point_policy_dominant_2x3.png`; loading a
model, producing nonzero actions, or converging in the old fixed-target rollout
is not sufficient.

## Acceptance Criteria

The final evaluation uses three unseen UE5 runs: RED 3m, BLUE 3m, and RED 4m.
Each run must use a seed absent from training and must record target and ASV
world positions from the same run.

- UE5 runtime: at least 180 s per scenario.
- Target displacement: at least 50 m in world coordinates.
- Policy-driven valid commands: at least 90% after the first 10 s.
- No divergence: absolute standoff error never exceeds 8 m after the first 10 s.
- Steady-state MAE over the final 120 s: at most 0.5 m.
- Final 10 s mean absolute error: at most 0.3 m.
- The result figure has a 2x3 layout: world trajectories on the top row and
  signed standoff error versus time on the bottom row.
- The final evidence comes from one fresh UE5/Jetson run per scenario and
  includes CUDA perception, CUDA policy, valid displacement, and UE5 apply
  markers from that same run.

## Data Collection Architecture

UE5 gains an opt-in expert collection mode inside the existing
`SceneAutomationSubsystem`; no new ROS node is added. The expert uses the UE5
target world position and velocity only during data collection. It produces a
bounded incremental world-frame displacement that combines target-velocity
feed-forward with radial standoff correction. The command is applied through
the same kinematic movement boundary used by closed-loop evaluation, so the
camera observes the ASV while it is actually following the moving target.

Jetson runs the existing bridge and records `/ue/camera_frame`, `/ue/asv_state`,
`/ue/entities`, and `/control/desired_displacement` with `ros2 bag`. The expert
command is included in UE5 diagnostics and recovered from consecutive ASV world
poses; the exporter rejects samples whose run identity or frame order is
inconsistent. Bags are copied to `D:/asv-vla-training/data/raw` and converted to
episodes under `D:/asv-vla-training/data/episodes/moving_target`.

## Expert Controller

For target position `p_t`, target velocity `v_t`, ASV position `p_a`, requested
standoff `d`, and control interval `dt`:

```text
r = p_t - p_a
e = norm(r) - d
v_cmd = v_t + k_p * e * normalize(r)
delta = clamp_norm(v_cmd * dt, 0.30 m)
```

The controller has a small deadband around the requested distance and limits
per-step acceleration to avoid abrupt labels. RED 3m, BLUE 3m, and RED 4m are
collected independently. Training, validation, and test use disjoint UE5 seeds.

## Training

Perception is trained from the new raw JPEGs and synchronized UE entity truth;
no exposure, gamma, brightness, or contrast preprocessing is permitted. Policy
training uses the expert displacement as the label and the current ASV surge
velocity/yaw rate as ego input. Synthetic distance scaling and frame-index
derived fake ego values are removed from the final policy dataset.

Training outputs use stable names only:

```text
experiments/moving_target/perception.npz
experiments/moving_target/policy.pt
experiments/moving_target/metrics.json
experiments/moving_target/tracking_result.png
```

## Evaluation And Failure Handling

Offline tests verify synchronization, expert controller direction/bounds,
split isolation, and metric calculations. A model failing any offline gate is
not deployed. UE5 evaluation is serialized one scenario at a time. Missing
identity, stale messages, invalid CUDA inference, invalid actions, or absent UE5
apply markers fail the run rather than falling back to cached data or CPU.

The final 2x3 plot is generated only from parsed fresh UE5 world-position logs;
the plotting code must not synthesize target trajectories or exponential error
curves.
