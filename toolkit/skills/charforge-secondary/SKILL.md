---
name: charforge-secondary
description: Simulate body-scaled hair and outfit chains deterministically.
---

# charforge-secondary

## Inputs

Explicit height/mass proxy, measured feature lengths, ordered roots/tips, actual body targets and collider samples.

## Outputs

secondary.json with resolved presets, fixed-step trajectory, numerical diagnostics and inspected surface evidence.

## Procedure

1. Resolve presets with toolkit.secondary.physics.resolve_preset and build chains with topology.from_roots_tips.
2. Advance Lattice in fixed substeps, with stable constraint order and target-space drive velocity.
3. Use assess for length, response, settling and sampled collisions. Inspect actual hair and outfit triangles separately.
4. Bake or use runtime springs, with a single authority per chain. Do not claim engine parity from numerical proxy tests.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
