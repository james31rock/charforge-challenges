# Layered 2D face on a 3D head

The method registers approved expression/mouth edits to a neutral drawing, reconstructs a shared feature-free skin field, isolates feature alpha, removes original animated ink from native UV texels and drives one active atlas state per feature on the original head surface. No neural model or supplied face art is bundled.

| Step | Tool and contract |
|---|---|
| Register and isolate | `layers.extract(character, out)` reads `face-grids/master.png` and approved `face-targets/*.png` files; `matte_v2.reconstruct` uses the registered temporal skin evidence |
| Inspect geometry | Blender `inspect_mesh.py --source head.blend --out <inspection>` writes geometry/UV arrays and textures |
| Bind atlases | `surface.prepare(folder, character)` uses `layers.json`, `aligned.npz`, `base.png`, `original-front.png`, `camera.json` and the inspection; outputs `binding.json` and atlases |
| Remove static ink | `erase_static.erase(source, inspect, skin, out, freeze)` requires a frozen failed-baseline report and a supplied continuous skin field; cleans original UV support without hiding ink behind a covering plate |
| Animate | `timing.make_track` or `timing_v2.make_smooth_track` consumes PCM16 WAV and phonetic cue JSON; `mouth_morph`/`mouth_transport` interpolate one contour while preserving endpoint artwork |
| Install | Blender `blender_face.py` installs the cleaned static texture, atlas shader and controls with constant state keys |
| Measure | `c8_gate` checks static ink, one active source per feature, observed connected groups, missing core pixels and independent depth-supported occlusion; `mouth_likeness` checks style relative to the approved neutral mouth |

Run ordinary modules with `python -B -m toolkit.face.<module> --help`. Run native adapters with `blender -b --factory-startup -Y -t 2 --python-exit-code 1 --python toolkit/face/<script>.py -- <arguments>`. Always work on a new output copy. Runtime data formats retain the stage-relative layout above; no separate application installation is required.

Source/front registration is affine from measured eyes and mouth. UV attachment is defined in rest geometry rather than camera-facing billboards. C8 uses the union of all approved feature supports as fixed regions. Occlusion needs one original stroke plus independent depth/hair evidence and supported visible pixels; a missing feature alone is not evidence of occlusion. The current movie adapter expects 240 frames at 24 fps in two 512-pixel review views with its documented camera curve; use the array APIs for other layouts.

Mouth likeness measures width and opening relative to neutral width, line thickness from distance-transform ridges, cheek centre and upper-lip anchors in interocular units, and excessive white area for an EE mouth. Bounds depend on expression/viseme. This is a falsifiable style check, not proof of identity.

The public atlas integration test loads all states, rejects a widened mouth and rechecks a frame from the failing brow movie. Synthetic tests challenge duplicate brows, unexplained disappearance, independently supported narrow fragments, silent-mouth closure and exact endpoint preservation. The frozen challenge continues using its own C8/6; this module is C8/10.

Limits: skin and feature segmentation assumes the palette/contrast of the tested cartoon style. Dark skin, unusual eye designs, profile views and new hairstyles need calibration. Temporal skin reconstruction can damage boundaries; keep original UV/geometry and hair-mask comparisons. Human likeness and motion review remain required. No general lip-sync, hair repair or engine facial export acceptance is implied.
