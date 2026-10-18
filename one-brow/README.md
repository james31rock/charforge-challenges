# C8: one brow per side

Pippa is an original test-cast character, supplied as a packed head and layered
face. Repair the face so every animated feature is absent from every static
source, then appears exactly once per side in the active composite. Preserve
the bangs and correct depth occlusion at front and 35-degree views.

```powershell
python score.py baseline
python score.py tests/bruno --control-bruno
```

`baseline/face.blend` and its movie are the frozen failed demo. `assets/face.blend`
is a later partial repair supplied for study. Its surprised far brow is physically
occluded at 35 degrees; C8/6 supersedes the older rule that rejected all such
fragments. The historical failed demo remains the challenge baseline. `assets/reference` contains the fixed feature atlases,
registration, timeline and feature-free reference. Maps are derived from the
original mesh, not from a candidate's reported regions. Bruno is a separately
registered known-good control using the identical code, not a Pippa solution.

## What failed

Fixed proportional windows cut moving brows and include hair or cheek ink.
Connected components miss hair-connected strokes; a 138-pixel compact brow was
confused with an 802-pixel eyelid. Temporal skin reconstruction retained
antialiased ghosts. Covering the original texture with an opaque feature-free
plate looked clean while the baked UV atlas still contained brows. Native UV
erasure removed those marks but altered the hair boundary; far brows still
vanish at three-quarter view. Vector redraw and generic landmark detectors
were rejected because they do not preserve this artwork. Prior unqualified
acceptance was withdrawn when the raw static material was finally exposed.

## Exact pass criteria (C8/6)

All four raw/composite static views and the submitted 512x512 static plate:
**zero** feature-coloured pixels within the fixed brow/eye/mouth regions.
Each brow and eye: **one** significant connected feature at every required
view/frame. No significant unassigned ink in the 240-frame movie. All 18 expression/blink views must pass the same depth-aware feature check. Head topology
and positions must match the supplied head. A sub-resolution brow is allowed only when its source and frontal strokes are
singular, at least 50% of frontal samples are independently depth-hidden and
observed ink matches the z-buffer prediction. Empty ink fails when at least four
pixels are predicted. Painting another brow over the bangs is not a solution.

The occlusion correction was reviewed while packaging: hidden is not missing.
It preserves static-zero and duplicate gates. `assets/gate-freeze.json` pins the
source snapshot, so later upstream edits cannot silently change this challenge.

Frozen pixel rules: luminance <120; alpha-core >0.12; ROI dilation 0.045 times
interocular distance; component area >=max(4,0.0005*d*d), stroke width
>=max(3,0.07*d); 2-pixel raster allowance; eye colour distance >35 and eye join
radius max(1,round(0.01*d)). Exclusive active regions separate adjacent eye/brow
ink. `vendor/pixels.py` is the exact check subset; no threshold is retuned here.

The one-command scorer opens `face.blend` with auto-execution disabled and
renders all 22 required stills itself. Supply `feature-free.png` and a 24 fps,
240-frame `face-demo.mp4` with the original two-view layout: 512px crops at
y=56..567, front/turn on the left and fixed 35 degrees on the right. Preserve
the timeline in `assets/reference/track.json`; neutral is explicit. Preserve
the named face controls and S5 material labels used by the bundled renderer.
Outside scripts are reviewed, not run by the scorer. Maintainers must reproduce
the movie from the submitted recipe; a video file alone cannot prove its source.
Hair-boundary fidelity, skin seams, proper depth occlusion, mouth smoothness and
global likeness additionally require visual review. This is not full face acceptance.

Include `binding.json` and every referenced atlas (including the visibility mask),
using relative paths within your submission. Fixed geometry, camera, landmarks
and registration must remain unchanged. The static ROIs stay frozen; active
drawings are read from your supplied atlases for the occlusion proof. Preserving
one source stroke and frontal visibility prevents an empty atlas from winning.
