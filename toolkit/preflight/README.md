# Rig preflight

The Blender adapter inspects rest geometry, weights and a few FK stress poses. It predicts eight failure families before motion generation. It does not alter or save the input file.

```powershell
& $env:BLENDER -b --factory-startup -Y -t 2 --python-exit-code 1 --python toolkit/preflight/preflight.py -- --input hand-plant/assets/rig.blend --output preflight.json
```

Inputs: BLEND, GLB/GLTF or VRM container; a rig with supported canonical, Rigify, VRM or common humanoid bone names. Output: measured metrics, likely/clear predictions and knee coverage. `--batch` accepts a JSON list of `{"name":"example","path":"rig.blend"}` records. Use canonical metres/Z-up geometry; unit normalization is a prior stage.

Read [FAILURE-PATTERNS.md](FAILURE-PATTERNS.md) and its JSON counterpart for the math and provisional thresholds. The integration test runs the public professional humanoid fixture unchanged and requires eight predictions with no false likely result.

These are heuristic predictors, not acceptance gates. Millimetre thresholds, weight-bleed cutoffs and triangle-intersection counts require wider calibration. Missing anatomy or invalid ray fans must not be interpreted as a clean rig. The original predictor includes one-sided measurements and conservative static approximations; inspect both sides and animation separately. A no-error process exit only means diagnostics ran.
