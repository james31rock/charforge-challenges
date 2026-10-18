# Gait geometry comparison

`core.load`, `measure` and `compare` operate on `charforge-gait-trace` JSON (`trace.schema.json`). `naturalness.measure_full` adds shoulder, pelvis, torso, arm and phase descriptors; `rank` orders deviations for investigation.

```powershell
python -B -m toolkit.gait reference.json candidate.json --out comparison.json
```

Supply a rest skeleton, full ordered joint samples, explicit cycle duration and phase origin. `phase_method` must describe how timing was determined. Declare `periodic: false` for non-cyclic captures; do not create a false seam. All traces use metres and Z up. Native extraction and reference acquisition are caller responsibilities.

Lengths are normalized by measured leg length. Angles use a moving pelvis frame. Signals are resampled over phase before RMSE; pelvis-height offsets are removed. Phase jerk separates geometric roughness from cadence-dependent time jerk. Knee heading is undefined near extension and remains null. A declared stance fraction is distinct from a sampled fraction or a velocity proxy.

Use this to compare a proposed walk repair against unchanged reference geometry. The public integration test exercises these calculations on the hand challenge's pose cache and a scaled/translated copy; that non-walk clip is a geometry test, not gait validation. Analytic tests cover phase shifts, root pops and four proportion sets.

Outputs always say `diagnostic-not-acceptance`. Clinical adult norms do not establish chibi or child thresholds. Joint-centre angles are not anatomical joint-coordinate measurements, and estimated COM is not force-plate evidence. A smaller RMSE alone does not prove a better-looking walk.
