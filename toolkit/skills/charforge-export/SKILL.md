---
name: charforge-export
description: Review Blender and engine exports after upstream acceptance.
---

# charforge-export

## Inputs

Accepted rig, motion, face and secondary evidence plus explicit distribution rights.

## Outputs

export.json with per-format hashes, round-trip checks, units, influence budget and import instructions.

## Procedure

1. Use the separately installed Blender application to export the requested BLEND, GLB or FBX. This toolkit does not include a turnkey exporter or a VRM extension writer.
2. Reimport each format and compare geometry, skeleton, animation, face states and scale. Preserve per-format failures.
3. Validate Unity and Unreal imports in those applications, including influence limits and baked versus runtime spring authority.
4. Create character-sized collision bodies and engine physics configuration explicitly. A Blender export does not create an Unreal physics asset or prove runtime visual parity.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
