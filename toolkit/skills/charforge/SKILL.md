---
name: charforge
description: Use the toolkit to plan and review a reproducible character pipeline.
---

# charforge

## Inputs

Original or licensed cartoon views, explicit positive height in centimetres, measured anatomy and separately obtained mesh/motion data.

## Outputs

A run folder with intake.json, views.json, mesh.json, mesh-qa.json, rig.json, face.json, motion.json, secondary.json, acceptance.json and export.json.

## Procedure

1. Install toolkit/requirements.txt and read toolkit/README.md and toolkit/LICENSES.md.
2. Follow the stage skills in order. Bind each manifest to input/output hashes, units, configuration, status and uncertainty. Retain the failed baseline before repair.
3. Use the included deterministic modules for rig math, physics and diagnostics. Image generation and final engine integration require separately supplied tools and explicit data rights.
4. Inspect contact sheets and movies before acceptance. Missing anatomy, unavailable measurements or uncalibrated gates remain BLOCKED. A successful file save is not acceptance.
5. Run the toolkit test suite and the unchanged challenge verifier after changes. Preserve professional controls and seeded counterexamples. No single command currently certifies every stage or arbitrary anatomy.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
