---
name: charforge-mesh-qa
description: Check scale, geometry and observable landmarks before binding.
---

# charforge-mesh-qa

## Inputs

mesh.json, original views, positive height_cm and mesh arrays.

## Outputs

mesh-qa.json and measurements.json conforming to toolkit/rig/measurement.schema.json.

## Procedure

1. Normalize to metres, Z up, X left and Y back using a recorded transform.
2. Inspect mesh connectivity, degenerate triangles, open boundaries, silhouettes and distinct semantic parts. Preserve original files.
3. Use toolkit.rig.geometry section helpers for observations; flag clothed or unsupported cross-sections.
4. Measure visible joints and leave uncertain hidden joints explicitly uncertain. Do not use clothing width as hip anatomy.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
