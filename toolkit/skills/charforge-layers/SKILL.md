---
name: charforge-layers
description: Fit an inner collision body and partition visible shells.
---

# charforge-layers

## Inputs

Canonical mesh, rig landmarks, explicit height and a separately licensed base OBJ or observed proxy arrays.

## Outputs

layers.json, landmarks.json, rig.json and candidate proxy/shell geometry.

## Procedure

1. Read toolkit/layers/README.md and supply the documented base convention; no base data is bundled.
2. Use tps, fit_hip_from_leg and voxel_surface for deterministic fitting and conservative proxy construction.
3. Run the optional fit.py adapter only on a new output copy. Inspect shell seams and hidden-body coverage.
4. Score the public mannequin with the unchanged garment scorer. Open shells and uncertain labels remain unresolved.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
