# C7: plant a cartwheel palm

Retarget CMU 49_06 onto the supplied pixiv native rig. During source hand-support
episodes, the palm skin must lie flat, the wrist must extend, fingers must stay
flat and material points must not skate or penetrate the floor.

```powershell
python score.py baseline/motion.npz
python score.py tests/control/poses.npz --control
```

`assets/rig.blend` is the fixed packed native imported rig; `sample.vrm` is the
original. The current failed retarget is `baseline/motion.npz`. The source BVH
and converted 24 Hz cache are in `assets/`. `contact-plan.json` fixes left
[46,63) and right [42,48), zero-based half-open, across all 97 samples. These are
reviewed source annotations. Automatic detection found only left [61,63), no
right; raw source hand landmarks float 89–181 mm above the inferred floor.
They are not measured physical contact or reliable finger motion.

## Why the problem remains open

Point locks preserve curled wrists. Flattening the palm changes reach: an early
arm-only solve left up to 113.17 mm residual on this rig. Smooth whole-clip root
and shared anchors removed target reach error but did not plant evaluated skin.
Skin-plane alignment plus two-link IK still failed wrist extension and ground
clearance. The packaged second attempt adds anatomical elbow-plane selection
and finite-difference hand orientation/height refinement, but rotates during
support. No contact correction is accepted merely because its bone target lands.
Discontinuous captures were replaced; per-frame root snapping was rejected.

## Exact pass criteria

All eight checks must PASS; missing skin patches/fingers/source coverage FAIL.
`h` is fixed native neutral mesh height; distances are metres, angles degrees.

| Check | Maximum / band |
|---|---|
| Palm-facing skin-plane tilt | 15 degrees |
| Absolute height of all four fixed palm patches | 0.006 h |
| Anatomical wrist extension | 60–90 degrees with 5-degree tolerance |
| Maximum 3D speed of a fixed palm patch within support | 0.08 h / second |
| Entire hand/finger skin depth below floor Z=0 | 0.001 h |
| Sagittal finger bend/deviation | 20 degrees |
| Neutral wrist deviation, action disabled | 10 degrees |
| Neutral finger bend, action disabled | 20 degrees |

The neutral/open-hand profile is not a clinical natural-pose limit. Thumb
opposition is unmeasured. Tolerances are engineering thresholds inherited from
the original gate, not claims of literally zero numerical penetration.

Use the same `names`, `basis`, `world`, `times` arrays as the baseline. Matrices
must be finite rigid transforms; no scale or reflection. Rig geometry, weights,
rest pose, floor and source labels are fixed. Joint attachment preservation is
checked separately (maximum joint-head length change 0.0001 h for upper/lower
arms and legs). The synthetic control is a skinned two-hand fixture, not a
solved cartwheel and not an eligible entry. Full-body contacts, entry/exit pops,
root smoothness and recognisable source motion need review before acceptance.
Submit the output plus the script/patch that produced it; see `submit/` and
the repository contribution rules.
