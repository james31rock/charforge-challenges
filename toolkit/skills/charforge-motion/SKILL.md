---
name: charforge-motion
description: Apply deterministic contact, knee, root and arm corrections to licensed motion.
---

# charforge-motion

## Inputs

Rig measurements, separately licensed motion, explicit source contacts and cycle timing.

## Outputs

motion.json with source hashes, target samples, contact residuals and QA/gait diagnostics.

## Procedure

1. Use toolkit.rig.math_core for two-bone IK, flexion-gated knee poles, shoulder counterrotation and authored-anchor locks.
2. Use foot_math for measured ball pivots and gait_planning for smooth reach-bounded root and swing clearance.
3. Preserve unresolved reach residuals, source contact windows and canonical units.
4. Evaluate actual deformed surfaces with QA and compare traces through toolkit.gait. Inspect complete motion; no mocap acquisition or complete retarget service is bundled.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
