---
name: charforge-mesh
description: Obtain a mesh from reviewed views and retain generation provenance.
---

# charforge-mesh

## Inputs

Reviewed views.json, physical height and explicit input rights.

## Outputs

mesh.json with mesh/texture hashes, tool/version, source-view references, part labels and unresolved geometry.

## Procedure

1. Use a separately installed or authorized image-to-3D tool; this skill does not upload files or install a model.
2. Export self-contained geometry and textures. Record orientation, model units, source images and known fused parts.
3. Treat hidden body completion, hair separation, likeness and topology as unresolved until measured.
4. Pass the result to mesh QA without a pre-existing rig when testing the procedural fitter.

## Limits

Use original or appropriately licensed inputs. Keep manifests and reviewed evidence in your run folder. Tools in this distribution are experimental; preserve uncertainty and failed checks. This skill neither acquires assets nor certifies arbitrary-body production quality.
