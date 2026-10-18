---
name: charforge-face
description: Build and validate a layered face while preserving design.
---

# charforge-face

## Inputs

Approved registered edits, neutral drawing, inspected head mesh and cue/timing data.

## Outputs

face.json, binding.json, feature atlases, native UV cleanup and C8/likeness reports.

## Procedure

1. Follow toolkit/face/README.md: register, reconstruct skin, isolate features, erase original native UV ink, bind and animate.
2. Freeze the failed baseline before cleanup. Never hide static duplicate ink behind an opaque base plate.
3. Run C8/10 plus mouth_likeness and inspect source/front/oblique views and all expression changes.
4. Use the unchanged challenge scorer for challenge submissions. A local C8 pass does not establish global likeness or engine compatibility.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
