---
name: charforge-preflight
description: Predict structural motion failures from a native rig.
---

# charforge-preflight

## Inputs

Candidate rig and declared canonical units.

## Outputs

preflight.json with eight predictor families and measured metrics.

## Procedure

1. Run toolkit/preflight/preflight.py through the documented headless Blender command.
2. Read toolkit/preflight/FAILURE-PATTERNS.md and inspect each likely or missing-evidence result.
3. Fix measured anatomy or weights, then rerun the exact same diagnostics and professional control.
4. Keep advisory predictions separate from motion acceptance; no-error exit is not a quality pass.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
