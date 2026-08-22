# Moving-Target Policy Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect real UE5 expert-follow trajectories, train the perception and policy models from synchronized data, and pass three fresh 180 s moving-target closed-loop evaluations.

**Architecture:** Add an opt-in expert controller to the existing UE5 scene subsystem, record the existing ROS topics without adding a node, export identity-matched episodes on the training PC, then train and evaluate with real world trajectories. The final plot is produced from UE5 logs only.

**Tech Stack:** UE 5.6 C++, ROS 2 Humble rosbag2, Python 3, NumPy, PyTorch, pytest, matplotlib.

---

### Task 1: UE5 Expert Controller Contract

**Files:**
- Modify: `D:/asv-unreal-simulation/Tests/test_scene_automation.py`
- Modify: `D:/asv-unreal-simulation/Source/EDGE/SceneAutomationSubsystem.h`
- Modify: `D:/asv-unreal-simulation/Source/EDGE/SceneAutomationSubsystem.cpp`

- [ ] Add a failing static contract test requiring opt-in `ExpertFollowColor`,
  `ExpertStandoffM`, bounded 30 cm displacement, acceleration limiting, and
  `SCENE_EXPERT_APPLY` diagnostics.
- [ ] Run `python -m pytest Tests/test_scene_automation.py -v` and verify the new
  test fails because the expert mode is absent.
- [ ] Implement the minimal command-line parsing and expert controller inside
  `USceneAutomationSubsystem`; ordinary runs must remain unchanged.
- [ ] Re-run the UE tests and verify all tests pass.
- [ ] Build `EDGEEditor Win64 Development` with UE 5.6 and verify compilation.

### Task 2: Synchronized Bag Export

**Files:**
- Create: `D:/asv-vla-training/src/export_moving_target_bag.py`
- Create: `D:/asv-vla-training/tests/test_export_moving_target_bag.py`

- [ ] Add failing tests for exact identity matching, monotonic frame indexes,
  raw JPEG preservation, expert action extraction, and rejection of incomplete
  samples.
- [ ] Run the exporter tests and verify failure because the exporter is absent.
- [ ] Implement a small exporter that consumes decoded rosbag records and writes
  the existing episode manifest/frame JSON layout with an explicit `action`.
- [ ] Run exporter tests and the full training test suite.

### Task 3: Real-Action Policy Dataset

**Files:**
- Modify: `D:/asv-vla-training/src/train_final.py`
- Modify: `D:/asv-vla-training/tests/test_final_training_contract.py`

- [ ] Add failing tests proving policy rows use recorded actions and recorded
  ego state, do not derive ego from `frame_index`, and do not use synthetic
  distance scales.
- [ ] Run the focused test and verify the old synthetic dataset fails it.
- [ ] Extend `EpisodeRecord` with a checked two-dimensional action and replace
  synthetic policy labels with recorded labels.
- [ ] Re-run focused and full tests.

### Task 4: Real Trajectory Metrics And Plot

**Files:**
- Modify: `D:/asv-vla-training/run_training.py`
- Modify: `D:/asv-vla-training/tests/test_run_training_evaluation.py`

- [ ] Add failing tests for 180 s duration, moving-target displacement, steady
  MAE, final-window MAE, policy-driven ratio, and rejection of a fixed target.
- [ ] Run the focused test and verify the old fixed-target rollout fails.
- [ ] Replace `rollout_plot()` with a parser/plotter for actual UE5 target and
  ASV world traces; remove all synthetic trajectory updates.
- [ ] Re-run focused and full tests.

### Task 5: Headless Expert Collection

**Files:**
- Create: `D:/asv-vla-training/scripts/collect_moving_target.ps1`
- Output: `D:/asv-vla-training/data/raw`
- Output: `D:/asv-vla-training/data/episodes/moving_target`

- [ ] Start only the Jetson bridge and confirm the expected four recording
  topics and run identity fields are present.
- [ ] Run UE5 offscreen expert collection for RED 3m, BLUE 3m, and RED 4m using
  disjoint train/validation/test seeds.
- [ ] Record each run with rosbag2, copy it to D:, export it, and verify JPEG
  brightness statistics and identity completeness.
- [ ] Reject and repeat any run shorter than 180 s or with target displacement
  below 50 m.

### Task 6: Train And Gate Models

**Files:**
- Modify: `D:/asv-vla-training/run_training.py`
- Output: `D:/asv-vla-training/experiments/moving_target`

- [ ] Train perception and policy using only training seeds.
- [ ] Evaluate validation seeds and tune only from validation results.
- [ ] Run the untouched test seeds and require the offline action and perception
  gates before deployment.
- [ ] Save stable model names, metrics, hashes, split lists, and training inputs.

### Task 7: Jetson Deployment And Real Closed Loop

**Files:**
- Modify: `/home/jetson/jetson_asv_ws/models/manifest.yaml`
- Output: `D:/asv-vla-training/experiments/moving_target/tracking_result.png`

- [ ] Copy only `perception.npz` and `policy.pt` to a temporary Jetson model
  directory and verify SHA-256 before atomic replacement.
- [ ] Build and test the four Jetson packages, then launch the four-node stack.
- [ ] Run RED 3m, BLUE 3m, and RED 4m UE5 scenarios for 180 s with unseen seeds.
- [ ] Require same-run CUDA readiness, valid displacement, and
  `SCENE_EXEC_APPLY` evidence.
- [ ] Parse fresh UE5 logs, calculate all acceptance metrics, and generate the
  final 2x3 figure from actual trajectories.
- [ ] Update both workspace context documents with paths, hashes, seeds,
  metrics, and the evidence boundary.
