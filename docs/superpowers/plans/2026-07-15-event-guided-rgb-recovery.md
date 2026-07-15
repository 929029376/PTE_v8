# Event-Guided RGB Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace always-on heavy SRBT recovery with a detached lightweight visibility gate and an ABSENT-only Event-proposal/RGB-verification recovery expert while preserving normal expert outputs and parameters.

**Architecture:** Normal TRACK frames keep the current selected expert path. A small gate and hysteretic controller decide TRACK/SUSPECT/ABSENT/VERIFY. ABSENT frames generate full-frame Event proposals, batch only the candidate RGB/Event crops through the existing encoder/redetection head, and require clean-RGB identity verification before resuming.

**Tech Stack:** Python, PyTorch, pytest, existing PETTrack ViT+Hopfield, THOR, CenterPredictor, FELT causal SequenceVal.

## Global Constraints

- Do not change normal expert architecture or weights during recovery training.
- Reuse `srbt_controller.py`, `redetection.py`, THOR, and the existing redetection localization head.
- Remove superseded hazard, future-teacher, and always-on spatial-field runtime paths; do not retain duplicate implementations.
- Test each task independently before starting the next task.
- Record every local modification in `LOCAL_WORKLOG.md`; never upload tests, plans, specifications, or the worklog.
- Use official FELT data only; ground truth is scoring/training supervision, never a Test input.

---

### Task 1: SRBT-Lite Gate and Four-State Controller

**Files:**
- Modify: `lib/models/layers/srbt_controller.py`
- Modify: `lib/config/pet_track/config.py`
- Test: `tests/srbt/test_belief_controller.py`

**Interfaces:**
- Produces: `VisibilityGate.forward(pooled_feature, response_stats) -> Tensor[B,2]`
- Produces: `VisibilityController.step(present_probability, identity_score=None, localization_score=None) -> ControllerAction`
- Produces states: `TRACK`, `SUSPECT`, `ABSENT`, `VERIFY`

- [ ] Write failing tests proving: one weak frame stays TRACK without memory writes; two weak frames enter SUSPECT; four weak frames enter ABSENT; one strong recovery frame enters VERIFY; two verified frames resume TRACK.
- [ ] Run `F:\Anaconda\envs\attrack\python.exe -m pytest tests\srbt\test_belief_controller.py -q -p no:cacheprovider` and confirm failure against the old three-action controller.
- [ ] Replace the old duration/hazard decision logic with the four-state hysteresis controller. Keep `ControllerAction` as the tracker boundary and make non-TRACK actions output absent with score zero.
- [ ] Add a minimal two-layer `VisibilityGate` using pooled feature plus response peak, entropy, and RGB/Event agreement. Validate exact batch shapes and finite values.
- [ ] Add configuration defaults: `SUSPECT_FRAMES=2`, `ABSENT_FRAMES=4`, `VERIFY_FRAMES=2`, `THETA_PRESENT=0.70`, `THETA_RECOVER=0.75`.
- [ ] Run the focused test and complete `tests/srbt` suite; proceed only when both pass.

### Task 2: Deterministic Full-Frame Event Proposals

**Files:**
- Create: `lib/models/layers/event_recovery.py`
- Test: `tests/srbt/test_event_guided_recovery.py`

**Interfaces:**
- Produces: `EventProposalExtractor.forward(event_frame, background=None) -> dict`
- Return keys: `heatmap: Tensor[B,1,H,W]`, `centers: Tensor[B,K,2]`, `scores: Tensor[B,K]`, `valid: BoolTensor[B,K]`

- [ ] Write failing tests with synthetic event images proving background suppression, deterministic Top-K ordering, spatial NMS, finite zero-event output, and original-image normalized coordinates.
- [ ] Run the focused test and confirm import failure.
- [ ] Implement activity as channelwise absolute magnitude, optional background subtraction, median/MAD normalization, max-pool NMS, and `torch.topk`; add no dependency.
- [ ] Keep `K`, NMS radius, and minimum robust score configurable. Invalid shapes raise `ValueError`.
- [ ] Run focused and complete suites before proceeding.

### Task 3: Independent RGB Clean-Template Verifier

**Files:**
- Modify: `lib/models/layers/event_recovery.py`
- Modify: `lib/models/pet_track/pet_track.py`
- Test: `tests/srbt/test_event_guided_recovery.py`

**Interfaces:**
- Produces: `RGBIdentityVerifier.forward(template_rgb_tokens, candidate_rgb_tokens) -> Tensor[B,K]`
- Produces: `PETTrack.rgb_identity_tokens(rgb_image, template: bool) -> Tensor[B,N,C]`

- [ ] Write failing tests proving identical RGB tokens score above shuffled negatives, Event tokens cannot change identity score, and verifier gradients do not reach supplied frozen tokens.
- [ ] Implement normalized template/candidate projections and cosine similarity. Detach input tokens inside the verifier so recovery loss cannot update the shared encoder.
- [ ] Extract RGB-only patch tokens using existing `_z_feat`/`_x_feat`; do not concatenate Event template tokens into identity verification.
- [ ] Add verifier parameters to the recovery optimizer group only.
- [ ] Run focused optimizer-coverage and complete SRBT tests.

### Task 4: Event-ROI Recovery Inference

**Files:**
- Modify: `lib/models/layers/redetection.py`
- Modify: `lib/models/pet_track/pet_track.py`
- Modify: `lib/test/tracker/pet_track.py`
- Test: `tests/srbt/test_event_guided_recovery.py`
- Test: `tests/srbt/test_memory_gating.py`

**Interfaces:**
- Consumes: Event proposals and RGB verifier from Tasks 2-3.
- Produces: `PETTrack.recover_from_candidates(clean_rgb, clean_event, candidate_rgb, candidate_event, event_scores) -> dict`
- Return keys: `boxes`, `identity_scores`, `localization_scores`, `event_scores`, `accepted`.

- [ ] Write failing tests proving ABSENT searches Event proposals outside the last local crop, batches Top-K candidates once, rejects high-Event/low-RGB candidates, and keeps THOR frozen during VERIFY.
- [ ] Change the tracker so TRACK uses the original output unchanged; SUSPECT freezes memory; ABSENT computes full-frame Event proposals every frame; VERIFY uses candidate-centered crops.
- [ ] Pass the real Event heatmap into redetection instead of `prior_H=None`.
- [ ] Reuse the existing redetection localization head for each candidate batch and require both identity and localization thresholds.
- [ ] Preserve Top-K hypotheses across frames and resume only after `VERIFY_FRAMES` confirmations.
- [ ] Run focused tracker, memory, output-contract, and complete SRBT tests.

### Task 5: Remove Superseded Heavy SRBT Runtime

**Files:**
- Modify: `lib/models/pet_track/pet_track.py`
- Modify: `lib/train/actors/pet_track.py`
- Modify: `lib/models/layers/pet_losses.py`
- Modify: `lib/train/data/sampler.py`
- Modify: `lib/config/pet_track/config.py`
- Test: `tests/srbt/test_srbt_model_integration.py`
- Test: `tests/srbt/test_train_test_contract.py`

**Interfaces:**
- Produces model outputs: `presence_logits`, `presence_score`, normal expert outputs, and recovery outputs only when requested.
- Removes runtime dependence on hazard, future teacher, always-on field, and old duration-triggered redetection.

- [ ] Write failing contract tests asserting TRACK forward does not call future encoding, spatial hypotheses, or redetection; ABSENT recovery is the only heavy recovery path.
- [ ] Replace current SRBT forward with gate statistics computed from existing shared features and normal response.
- [ ] Remove future observation sampling and obsolete hazard/field/teacher losses from the production configuration and actor.
- [ ] Keep official present/absent supervision for the visibility gate and absent-to-present boxes for recovery training.
- [ ] Update checkpoint loading to accept the new recovery modules while strictly validating all retained normal-expert keys.
- [ ] Run checkpoint, sampling, model-integration, contract, and complete suites.

### Task 6: Recovery Training Isolation

**Files:**
- Modify: `lib/train/base_functions.py`
- Modify: `lib/train/actors/pet_track.py`
- Modify: `experiments/pet_track/felt_pet_track.yaml`
- Test: `tests/srbt/test_all_student_parameters_trainable.py`
- Test: `tests/srbt/test_local_experts.py`

**Interfaces:**
- Adds phase: `TRAIN.EXPERT_PHASE=recovery`
- Trainable parameters: visibility gate, RGB verifier, recovery localization/adaptation only.

- [ ] Write failing tests that hash/freeze shared encoder and every normal expert, then verify only recovery parameters receive gradients.
- [ ] Add the recovery optimizer phase and fail fast if any frozen parameter enters an optimizer group.
- [ ] Add gate BCE/focal loss, RGB identity BCE/contrastive loss, positive bbox GIoU/L1, and true-candidate ranking loss.
- [ ] Configure balanced absent/reappearance sampling without challenge labels.
- [ ] Run focused gradient-isolation tests and complete suite.

### Task 7: Causal Sequence Validation and Server Deployment

**Files:**
- Modify: `lib/train/sequence_validation.py`
- Modify: `tracking/supervisord_local_experts_v8.conf`
- Modify: `LOCAL_WORKLOG.md`
- Test: `tests/srbt/test_sequence_validation.py`

**Interfaces:**
- Adds diagnostics: Event Recall@1/3/5, RGB false-accept rate, recovery latency, IoU@1/3/5 after reappearance, visible-sequence retention, TRACK FPS, recovery FPS.

- [ ] Write failing metric tests with synthetic causal sequences; ground truth is consumed only after tracker outputs are produced.
- [ ] Add the diagnostics without changing official FELT metric definitions.
- [ ] Run `F:\Anaconda\envs\attrack\python.exe -m pytest tests\srbt -q -p no:cacheprovider`, production `py_compile`, YAML load, and `git diff --check`.
- [ ] Record frozen-parameter hashes and local verification in `LOCAL_WORKLOG.md`.
- [ ] Stop the old remote run only after preserving its latest completed checkpoint and key logs.
- [ ] Synchronize production/config files only, run remote compilation and a one-sequence smoke test, then start recovery training from the best specialist checkpoint.
- [ ] Monitor GPU, OOM/NaN, gate false-absent rate, proposal recall, identity false accepts, and recovery IoU. Do not run Test until two consecutive SequenceVal points improve.

## Completion Gate

- Normal expert outputs match the pre-change checkpoint when the gate remains TRACK.
- Frozen normal parameters have identical hashes before and after recovery training.
- Complete local SRBT tests pass.
- Event proposal Recall@5 covers reappearance targets before recovery training starts.
- Two consecutive causal validation points improve reappearance metrics without visible-sequence regression.
- Final claims use the official FELT toolkit.
