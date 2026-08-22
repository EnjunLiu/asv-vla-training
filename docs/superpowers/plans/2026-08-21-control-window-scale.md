# Control Window Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a new three-scene candidate whose 0.5 s policy labels match the recorded UE5 action time scale and whose real closed-loop result is comparable to the world-coordinate reference chart.

**Architecture:** Keep the existing perception and decision model contracts unchanged. Fix only the training label adapter so each recorded frame action contributes according to its overlap with the 0.5 s control window, then bound the resulting body-frame displacement to the existing 0.50 m contract. Preserve the frozen prototype and old experiments while training a separate candidate.

**Tech Stack:** Python, NumPy, PyTorch, pytest, ROS 2, UE5, Jetson CUDA.

---

### Task 1: Reproduce and lock the time-scale contract

**Files:**
- Modify: `D:/asv-vla-training/tests/test_final_training_contract.py`
- Test: `D:/asv-vla-training/tests/test_final_training_contract.py`

- [ ] **Step 1: Replace stale 0.6 s expectations**

Change the existing fixed-step expectations from `[0.36, -0.12]` to `[0.30, -0.10]`, because a 0.5 s window contains 2.5 recorded 0.2 s intervals.

- [ ] **Step 2: Add a fractional final-interval test**

Use interval actions `[0.1, 0.2, 0.3, 0.4]` at timestamps `0.0, 0.2, 0.4, 0.6, 0.8` and assert the first label is `0.1 + 0.2 + 0.5 * 0.3 = 0.45`.

- [ ] **Step 3: Add a norm-bound test**

Use three `[0.4, 0.0]` intervals and assert the accumulated label has norm `0.50 m`, not `1.20 m`.

- [ ] **Step 4: Run the focused tests and record the expected failures**

Run `python -m pytest -q tests/test_final_training_contract.py -k "control_window or recorded_action or teacher_labeled"`. The failures must identify the old full-interval accumulation and missing norm bound.

### Task 2: Implement minimal fractional integration

**Files:**
- Modify: `D:/asv-vla-training/src/train_final.py`
- Test: `D:/asv-vla-training/tests/test_final_training_contract.py`

- [ ] **Step 1: Integrate action intervals by time overlap**

For each candidate record, use its timestamp and the next record timestamp as the recorded action interval. Add `action * overlap_duration / interval_duration` for the portion inside the policy window.

- [ ] **Step 2: Preserve incomplete and invalid-window behavior**

Return `None` when the next frame is unavailable, the window is not fully covered, an overlapping action is missing, or interval timestamps are invalid.

- [ ] **Step 3: Enforce the existing 0.50 m body-frame contract**

After integration, scale only vectors whose norm exceeds `0.50 m`; do not change model dimensions, ROS messages, control period, or node topology.

- [ ] **Step 4: Run focused and full training-contract tests**

Run `python -m pytest -q tests/test_final_training_contract.py` and then the relevant perception/evaluation contract tests. All must pass.

### Task 3: Train and register a separate candidate

**Files:**
- Create: `D:/asv-vla-training/experiments/control_window_candidate/`
- Modify: `D:/asv-vla-training/WORKSPACE_CONTEXT.md`

- [ ] **Step 1: Recompute dataset statistics**

Verify record count, slot split, label count, median/p95 action norm, and maximum norm after the fix. Store the command output with the candidate metadata.

- [ ] **Step 2: Train perception and decision heads without changing their contracts**

Use the real `moving_target_valid` dataset and save configuration, source hash, data hash, model hashes, and metrics under the separate candidate directory.

- [ ] **Step 3: Run offline evaluation only as a diagnostic**

Report it separately from online evidence and do not use replay plots as closed-loop acceptance.

### Task 4: Deploy and verify the real closed loop

**Files:**
- Modify: `/home/jetson/jetson_asv_ws/models/manifest.yaml`
- Modify: `D:/asv-vla-training/WORKSPACE_CONTEXT.md`
- Modify: `D:/asv-unreal-simulation/WORKSPACE_CONTEXT.md`
- Modify: `/home/jetson/jetson_asv_ws/WORKSPACE_CONTEXT.md`

- [ ] **Step 1: Copy only explicit model files**

Compare SHA-256 before and after deployment; do not overwrite the frozen prototype experiment.

- [ ] **Step 2: Run RED 3m, BLUE 3m, and RED 4m separately**

Each run must have a fresh `run_id`, model hash, CUDA `POLICY_READY`, valid non-fail-closed commands, UE5 `SCENE_EXEC_APPLY`, and world-coordinate target/ASV logs.

- [ ] **Step 3: Generate one unified world-coordinate result figure**

Use only same-run UE5 logs and label any missing evidence as blocked rather than smoothing or substituting replay output.

