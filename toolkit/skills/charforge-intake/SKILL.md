---
name: charforge-intake
description: Declare views and physical scale before modelling.
---

# charforge-intake

## Inputs

Views of the same licensed cartoon character on a plain background and explicit height_cm.

## Outputs

intake.json with source hashes, view labels/angles, crop rectangles, mask files, measured silhouettes, confidence and review status.

## Procedure

1. Label view direction explicitly; a symmetric silhouette or filename does not prove front/back.
2. Prepare masks with a separately supplied image tool, inspect white clothing and hair boundaries, and retain the original pixels. This toolkit does not include an automatic intake service.
3. Keep clothed silhouette ratios separate from anatomical joint observations. Unknown hidden anatomy stays null.
4. Record coordinate orientation, positive supplied height and any missing angles. Pass reviewed data to the view and mesh stages.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
