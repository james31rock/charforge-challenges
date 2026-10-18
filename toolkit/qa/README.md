# QA checks

`core.evaluate(trace)` consumes `qa-trace/1` (`trace.schema.json`) and returns `qa-report/1`. `hands.evaluate_hands` isolates C7. `geometry.py` uses Blender's BVH for triangle crossing, closed-volume depth and edge strain.

```powershell
python -B -m toolkit.qa trace.json --out report.json
```

Exit 0 = PASS, 1 = FAIL, 3 = BLOCKED. The trace must contain full-rate consecutive frames, positive height/hip width/leg length/fps, a fixed floor and finite world-space points. Contact intervals are authored or source-derived, never inferred to make a candidate pass. The calibration field is an evidence assertion by the caller; this evaluator cannot certify it independently.

| Check | Measurement |
|---|---|
| C1 | Hand volume penetration or confirmed transverse edge crossing; threshold 0.0005 stature |
| C2 | Hair/body penetration and head-local tip response divided by head acceleration |
| C3 | Sustained toe walking, fixed material-point skate, floor penetration and declared-stance gaps |
| C4 | Foot/knee separation in the pelvis frame, normalized by hips and leg length |
| C5 | Rest-defined section/volume retention and triangle-edge stretch |
| C6 | C1 restricted to declared landing frames |
| C7 | Skin-palm plane, gap, signed wrist extension, patch skate, penetration, fingers and neutral hand rest |
| C8 | [Face](../face/README.md): source-bound static ink and independently supported active features |

`test_core.fixture()` and `test_hands.planted()` provide original analytic cases; test methods mutate one failure at a time. `make_falsifier.py` and `make_c7_fixture.py` are optional Blender fixture generators. Run them through `blender -b --factory-startup -Y --python-exit-code 1 --python <script> -- --out <run>`. Generated geometry is original and contains no external assets.

The public hand challenge's `extract.py` measures fixed skin patches from its native rig; the integration test feeds those traces through the exported C7 evaluator and checks both baseline and control. General C1–C6 native extraction is deliberately an adapter contract: supply measured geometry fields, semantic part labels and coverage flags. No hidden rig adapter or motion cache is required by the numerical package.

Missing closed volumes, unknown contact intent, unexcited hair or absent calibration block acceptance. Finite samples cannot prove that all triangles never collide; static support is not dynamic balance. C5 stretch is not silhouette approval. Inspect worst-frame images separately.

`hair.strip_spacing` adds provisional material-patch spacing guards for challenge
4. Variance, minimum and maximum normalized spacing are all required: variance
alone cannot reject uniform collapse. The headless adapter in
`stylised-hair/measure.py` measures geometry before invoking C2; entrants submit
coordinates, not self-reported metrics. The synthetic positive control does not
certify the real character or calibrate these spacing bands on professional rigs.
