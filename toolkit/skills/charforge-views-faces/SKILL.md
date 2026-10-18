---
name: charforge-views-faces
description: Complete a consistent turnaround and approved expression/mouth edits.
---

# charforge-views-faces

## Inputs

Reviewed intake.json and the original character reference drawings.

## Outputs

views.json with turnaround, expression and mouth-state image hashes and manual likeness review.

## Procedure

1. Use an explicitly chosen image editor or generation service with the original reference attached to every edit.
2. Keep scale, pose, camera, face anchors and outfit identity consistent; include neutral/closed mouth and independently approved expressions.
3. Place the neutral drawing at face-grids/master.png and approved states under face-targets using expression-<state>.png and mouth-<state>.png names as documented by toolkit/face/layers.py. No service client is bundled.
4. Inspect every state before extracting layers. Reject changed smile width, line weight, silhouette or added features rather than treating a complete grid as a pass.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
