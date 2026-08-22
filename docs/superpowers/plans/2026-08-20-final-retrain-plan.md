# Final Retrain Collection and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated fresh-data collection and training workflow for the final UE5/Jetson experiment.

**Architecture:** A new collector owns only UE5 launch, Jetson bag recording, export, and archive transfer. A new training wrapper passes explicit paths into the existing training functions. Existing data and scripts remain untouched.

**Tech Stack:** PowerShell, SSH/SCP, ROS 2 rosbag2, Python, PyTorch, NumPy, Matplotlib, pytest.

---

### Task 1: Lock the workflow contract

**Files:**
- Create: `D:/asv-vla-training/tests/test_final_retrain_workflow.py`

- [ ] Write tests requiring the new collector to use `final_retrain`, refuse overwrite, and run UE5 for at least 215 seconds.
- [ ] Run the focused test and confirm it fails because the new entry points do not exist.

### Task 2: Add the isolated collector

**Files:**
- Create: `D:/asv-vla-training/scripts/collect_final_retrain.ps1`

- [ ] Implement explicit destination checks and the final 18-slot scenario list.
- [ ] Start exactly one Jetson bridge and one rosbag recorder with remote PID files.
- [ ] Run each UE5 scenario for 215 seconds, export synchronized frames, archive, and copy to D:.
- [ ] Stop remote processes in `finally` and preserve partial evidence for diagnosis.
- [ ] Run the focused contract test.

### Task 3: Add parameterized training entry point

**Files:**
- Create: `D:/asv-vla-training/run_final_retrain.py`

- [ ] Parse explicit data, run, and embedding paths.
- [ ] Reuse the existing split, perception training, policy training, metrics, and plot functions.
- [ ] Refuse missing or overlapping train/validation/test slots before writing outputs.
- [ ] Run the focused workflow tests and existing training tests.

### Task 4: Trial collection

- [ ] Run one RED 3m trial into a separate temporary remote directory.
- [ ] Verify bag duration, synchronized frame count, one run identity, direct JPEG bytes, and image statistics.
- [ ] Continue only if usable duration is at least 179 seconds.

### Task 5: Full collection and training

- [ ] Run the fresh collection into `D:/asv-vla-training/data/episodes/final_retrain`.
- [ ] Run `run_final_retrain.py` into `D:/asv-vla-training/experiments/final_retrain`.
- [ ] Verify model files, metrics, plot, and recorded source hashes.

### Task 6: Deployment and closed-loop validation

- [ ] Copy only the explicitly registered model artifacts to a temporary Jetson staging directory.
- [ ] Verify SHA-256 before and after deployment and rebuild/test Jetson if required.
- [ ] Run RED 3m, BLUE 3m, and RED 4m policy-driven UE5 scenarios.
- [ ] Generate the final 2x3 figure only from same-run closed-loop traces and report acceptance evidence.
