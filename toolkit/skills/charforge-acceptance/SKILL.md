---
name: charforge-acceptance
description: Evaluate measured motion and face evidence before delivery.
---

# charforge-acceptance

## Inputs

Hash-matched measured traces, contact intent, independent calibration, rendered motion and face reports.

## Outputs

acceptance.json with per-check PASS/FAIL/BLOCKED, coverage, source hashes and visual review.

## Procedure

1. Run python -B -m toolkit.qa trace.json --out report.json and retain all failed or unavailable checks.
2. Run face C8 and mouth likeness where applicable. Keep the challenge scorer version frozen.
3. Inspect worst frames plus every motion/expression change. Check actual source identity, skin contact, hair collision, sleeve shape and foot contact.
4. Do not assert four-body or animal acceptance without complete evidence on those bodies. Unknown or unsupported stages remain BLOCKED.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
