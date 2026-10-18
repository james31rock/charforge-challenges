---
name: charforge-rig
description: Fit a candidate humanoid skeleton, weights and correctives.
---

# charforge-rig

## Inputs

Mesh-only data and measurements.json with explicit orientation and height.

## Outputs

landmarks.json, rig.json and candidate rig.blend.

## Procedure

1. Read toolkit/rig/README.md, configure CHARFORGE_SCRATCH and select a new output child.
2. Run the documented headless toolkit/rig/blender_rig.py command with -Y and --python-exit-code 1.
3. Inspect pivot locations, native evaluated surfaces, weight domains and stress poses.
4. Run preflight before motion. Preserve missing-landmark and collision warnings; helpers are experimental until target-engine checks pass.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
