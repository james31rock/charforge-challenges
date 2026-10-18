# Procedural rig and motion math

`geometry.detect(vertices, triangles, height_m, observations)` fits section-centre landmarks; explicit observations override heuristics. `geometry.skeleton` produces canonical humanoid bones. `weights.diffuse` solves a screened mesh-graph diffusion problem with anatomical branch exclusions and normalized sparse influences. `blender_rig.helpers` adds body-scaled bend support and twist helpers.

```powershell
& $env:BLENDER -b --factory-startup -Y -t 2 --python-exit-code 1 --python toolkit/rig/blender_rig.py -- --measurements measurements.json --out $run
```

`measurements.json` follows `measurement.schema.json`: `schema_version: 1`, `character_id`, mandatory `height_cm`, `mesh_path`, and explicit `orientation.up`/`orientation.forward` vectors. Optional anatomical landmarks have a declared source/canonical space and confidence. `$run` must be an empty child of `CHARFORGE_SCRATCH`. The mesh must have no existing armature or weights. Outputs are `landmarks.json`, `rig.json` and a candidate `rig.blend`.

| File | Math and use |
|---|---|
| `math_core.py` | Sphere-intersection two-bone IK, flexion-gated knee pole, cyclic low-pass, opposite shoulder rotation, scaled arm swing, authored-anchor contact lock and girth-based correctives |
| `foot_math.py` | Sole-region measurements and ball-pivot toe rotation; preserves the bind sole frame |
| `gait_planning.py` | Fourier root curves with constrained quadratic reach bounds, triangle-section shoe width and support-aware swing clearance |
| `natural_gait.py` | Circular phase error, reference hip curves and smooth swing-floor correction |
| `silhouette.py` | Foreground interval evidence, normalized body envelopes and branch-distance priors |

Use the IK/contact primitives with independently declared heel/ball/toe anchors. Preserve unreachable-target residuals; do not clamp them into a passing contact score. Root and arm changes must be remeasured with QA. The public garment body's topology exercises landmark detection, skeleton construction and weight normalization; analytic cases cover extreme proportions and invariance.

Limits: humanoid heuristics, no finger/face articulation, up to eight influences, uncertain joints beneath fused clothes. Corrective drivers and preserve-volume skinning need target-engine baking and validation. A normalized weight vector does not prove deformation quality. No licensed motion or end-to-end retargeting service is bundled.

For a separate native construction test, run `toolkit/tests/blender_smoke.py` with the same headless Blender flags and `--out <run>`. It reads the public mannequin, creates a candidate, checks finite deformed vertices and writes three CPU review views. This intentionally simple fixture has no articulated hand or shoe design; visible bend quality remains diagnostic. Remove its generated BLEND files after review.
