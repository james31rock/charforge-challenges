---
name: charforge-gait-compare
description: Compare explicit reference and candidate gait geometry.
---

# charforge-gait-compare

## Inputs

Two valid charforge-gait-trace records with known phase timing and rest skeletons.

## Outputs

Diagnostic comparison.json and reviewed worst-phase observations.

## Procedure

1. Run python -B -m toolkit.gait reference.json candidate.json --out comparison.json.
2. Inspect normalized pelvis, leg, torso and arm descriptors; near-straight knee heading stays unavailable.
3. Use non-periodic mode for an open capture and keep declared contact separate from proxy contact.
4. Treat error ranking as a repair aid, never a universal production threshold.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
