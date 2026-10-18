# Inner body and visible layers

`core.tps` implements 3D polyharmonic interpolation with kernel `r` and an affine polynomial tail. `fit_hip_from_leg` extrapolates the visible knee/ankle column with a bounded slope instead of using skirt width. `voxel_surface` extracts a manifold half-isovalue surface with conforming tetrahedra and shared edge vertices.

The optional `fit.py` Blender adapter fits an explicitly supplied CC0 base, clips a conservative inner volume and partitions the original visible shells. It reuses the rig module, preserves shared seam weights and writes `layers.json`, `landmarks.json`, `rig.json`, `proxy.npz` and `layers.blend` plus review images.

```powershell
& $env:BLENDER -b --factory-startup -Y -t 2 --python-exit-code 1 --python toolkit/layers/fit.py -- --input proxy-input.json
```

Input fields: `character_id`, `height_cm`, `source`, `landmarks`, `mpfb_base`, `out`, optionally `canonical_source`, `mesh_objects` and `s0_palette`. `out` is a child of `CHARFORGE_SCRATCH` whose leaf equals `character_id`. The source is a mesh BLEND or FBX. Landmarks come from the rig module. `mpfb_base` is a separately acquired, licensed OBJ with a `body` group and `joint-*` anchors; the parser expects decimetre Y-up coordinates and converts them to metre Z-up. No base mesh or external add-on code is bundled. Use the core array functions for another base convention.

Voxel centres must be inside both the fitted body and visible source, with clearance for a full cell diagonal and margin. This is a collision proxy, not recovered hidden anatomy. Visible shell labels use geometry and explicitly declared palette evidence; unknown semantics remain review-required. Open shells and fused folds can remain defects. The full base-specific adapter is experimental; tests validate the mathematical core on the public mannequin, not anatomical correctness of a newly generated proxy.

Challenge use: compare proposed hidden-body completion to `garment-separation/assets/oracle.npz` and run the unchanged scorer. Sparse sample occupancy is only a test of closed-surface extraction, not a production body reconstruction.
