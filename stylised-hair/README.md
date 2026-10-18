# Challenge 4: Stylised hair physics

Make Pippa's anime bob swing visibly on a walk and a jump, stay clear of the
head, neck, shoulders and arms, and retain distinct, evenly spaced sections.
The challenge is open. There is no accepted real-character solution yet.

The input is the public test-cast character, not a story character. The packaged
mesh includes the head, complete hair surface and body needed for collisions.
`assets/walk.npz` and `assets/jump.npz` freeze the body and head drive at 24 fps:
119 walk frames and 113 jump frames, including 48-frame settling tails.
These are derived from CMU 08_03 and 16_01. See [asset terms](assets/LICENSES.md).

## Run

Use the repository's Python dependencies and Blender 5.x. Set `BLENDER` to the
executable and `CHARFORGE_SCRATCH` to a disposable directory outside the repo.

```powershell
python score.py baseline
python score.py tests/control --control
python score.py submit/my-entry --json result.json
python test_controls.py
```

The first command must FAIL; the second must PASS. Exit 0 means PASS and exit 1
means FAIL, including malformed inputs. Both clips must pass every check.
The synthetic control uses simplified geometry and analytic motion; it proves
the measurement path can pass. It is not a solution, a professional calibration,
or an entry eligible for the scoreboard. `--control` always reports that scope.

## What makes this hard, and what failed

Eleven iterations explored spring strips, seam pinning, neighboring-strip
constraints, capsule contact, mesh contact corrections and actual VRM-style
hierarchical spring chains. Some strip versions passed contact and response
numbers while visibly folding the nape or bunching hair. Collision-free guide
endpoints do not guarantee collision-free thick, weighted hair surfaces.

A subsequent audit confirmed six of ten proposed mechanisms, refuted two and
left two unresolved. Confirmed gaps included sparse three-guide binding,
missing center-space damping, unmeasured inward section thickness, averaging
per-guide results into one bob rotation, length projection after collision, and
independent strand translations without parent-relative angular limits.
Claims of a universal timestep bug and a particular seam-weight clamp were
refuted against the implementation and runtime tests. Predicted passes were
not evidence.

The retained baseline is the later eight-guide experiment: surface-based
guides, adjacent-sector skin weights, measured section envelopes, translation
damping, spherical angle limits and iterative fixed-length capsule contacts.
It still fails. The walk has a folded shelf at the nape; the jump has weak
follow-through. The original measured maximum depths were 6.4004 and 8.9960 mm;
response was 0.0137856 and 0.00623599 s². This package remeasures the actual baked
vertices, stored as float32; see [current scores](baseline/score.json).
No solver parameters are changed to manufacture a baseline failure.

## Exact pass criteria

The scorer imports `toolkit/qa/core.py` and `toolkit/qa/geometry.py` directly.
The companion spacing function is in `toolkit/qa/hair.py`.

| Check, on both full-rate clips | Pass band |
|---|---|
| C2 hair/body closed-volume depth | maximum ≤ 0.0005 × stature = 0.525 mm |
| C2 response: head-local tip range / RMS head acceleration | ≥ 0.01 s²; excitation ≥ 0.25 stature/s² |
| Variance across adjacent strip spacing ratios, at every frame | ≤ 0.04 |
| Smallest adjacent spacing / corresponding rest spacing | ≥ 0.65 |
| Largest adjacent spacing / corresponding rest spacing | ≤ 1.60 |
| Fixed scalp/root attachment displacement from drive | ≤ 0.001 × stature |
| Displacement from baseline hair envelope | ≤ 0.35 × stature |

Response uses the RMS of the two fixed tip patches' coordinate-wise peak-to-peak
ranges, divided by RMS head translation acceleration from centered second
differences. Tip positions are transformed into the head's local frame, so
rigid head motion cannot pass as secondary motion. Excitation below the band is
unavailable and fails; changing drive files is forbidden.

Eight fixed distal material patches define strip centres. They follow the
angular order of the bob, with adjacent pairs along that order; the front gap
is not closed into a ring. Each distance is divided by its own bind distance.
Uniform clumping can have zero variance, hence the mandatory minimum ratio.
These new spacing bands are provisional benchmark guards, not measurements
calibrated on a professional hairstyle. They do not detect every possible fold
or nonadjacent self-contact. Contact is sampled at 8,138 fixed moving-hair
vertices against the frozen body volume, including its arms and shoulders.
Attached scalp is excluded from contact, but its 6,224 fixed roots are checked.

Finite samples and volume closures cannot certify that every triangle remains
clear between frames. Review nape, side and rear views of both full clips before
accepting an entry. A numeric PASS is necessary, not a visual production verdict.

## Data contract and submission

`assets/fixture.npz` contains the rest mesh, triangle indices, vertex colours,
15,108 `hair_ids`, fixed root/contact/tip indices, eight `strip_weights` columns,
stature and frame rate. `assets/{walk,jump}.npz` contains full body `vertices`
and 4×4 head transforms. Coordinates are metres, Z up, world space. The two
drives have identical rest topology. Body volume loop closures and membership
are fixed in the fixture and reconstructed by the scorer.

Submit a directory containing exactly the required output data:

```text
my-entry/
  walk.npz       # hair: (119, 15108, 3), finite numeric coordinates
  jump.npz       # hair: (113, 15108, 3), finite numeric coordinates
  solution.py    # or a patch; reviewed separately, never run by score.py
  README.md      # method, dependencies/licences and reproduction command
```

Each NPZ must contain only the `hair` array, in `hair_ids` order. Keep every
vertex and frame; no retiming, topology substitution, body edits, deleted
strips or fixture changes. New solver topology is allowed internally, but bake
back to this output topology. The baseline files are a valid format example.
See [submit instructions](submit/README.md) and repository contribution rules.

For CPU witness renders from the exact submitted data:

```powershell
& $env:BLENDER -b --factory-startup -Y -t 2 --python render.py -- submit/my-entry --all --out $env:CHARFORGE_SCRATCH
```

The scorer never executes entrant code. Review source first, then reproduce it
only in an isolated scratch environment without credentials. No account,
network call, model download or paid service is needed to score an entry.

The baseline includes full-length half-speed walk/jump movies and three-quarter
and rear witness sheets. Renders use sampled vertex colours, so they are physics
evidence rather than a texture or face-quality evaluation. Movie/PNG metadata is
stripped before packaging. No new `.blend` is included in this challenge.
