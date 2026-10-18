# CharForge Challenges

Reproducible, scoreable problems from an open effort to turn a few cartoon images of a character into a
rigged, animated, physics-ready 3D character, for any body type. Each challenge has a shareable input,
an unchanged failing baseline and an offline scorer. Agents and humans are welcome: run the scorer,
beat the baseline, and send a PR.

| Challenge | Input | Task | Scorer |
|---|---|---|---|
| [Hand plant](hand-plant/README.md) | pixiv native skinned rig + CMU 49_06 | Flat, stable anatomical palm support | Native Blender skin measurements |
| [One brow per side](one-brow/README.md) | Pippa test head + layered face | Remove static duplicate ink, preserve one readable feature | C8/6 pixels plus independent static/expression renders |
| [Garment separation](garment-separation/README.md) | Self-made fused clothed mannequin | Closed body, garment and hair with hidden-body completion | Topology, known-body distance, silhouette IoU |
| [Stylised hair physics](stylised-hair/README.md) | Pippa bob + CMU-derived walk/jump | Visible sway without body clipping or clumping | Toolkit C2 surface depth, response and strip spacing |

## Setup

Python 3.12 and Blender 5.x. Tested with Blender 5.2.1. No account, network
connection or AI service is needed after installing dependencies.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
$env:BLENDER = (Get-Command blender).Source
$env:CHARFORGE_SCRATCH = Join-Path $env:TEMP 'charforge-scoring'
Set-Location hand-plant
../.venv/Scripts/python score.py baseline/motion.npz
```

From a challenge directory, `python score.py <submission>` prints PASS/FAIL and
JSON metrics. Exit 0 is PASS, 1 is FAIL (including malformed/unavailable inputs).
`--json report.json` additionally saves the report. A PASS applies only to the
specified benchmark; reviewer approval and reproducibility are separate.
See [SCOREBOARD](SCOREBOARD.md), [CONTRIBUTING](CONTRIBUTING.md),
[SAFETY](SAFETY.md), [asset licences](LICENSES.md), and [test report](VERIFICATION.json).

All thresholds, inputs, code and output hashes are frozen in `MANIFEST.json`.
Never edit a scorer or its reference assets to make a submission pass. Propose
gate corrections separately with a positive control and a failing counterexample.
This release covers one challenge subject per task and separate controls, not
validation on four body types or animals. No cash bounty is implied.

## Where the project is (what works, what doesn't)

**Working, measured:** multiview images to a textured mesh; a procedural humanoid rig that fits any body
from measurements (tested on a 105 cm child, a 185 cm heavy adult, a 90 cm chibi and a quadruped fox); walk
retargeting whose pelvis motion and step width match a motion-capture reference; foot-contact correction;
outfit and hair spring presets scaled by body size; a layered 2D anime face on the 3D head (expressions,
blinks, 81 mouth in-betweens, one active feature per frame); and automated QA that catches hands through
clothing, hair through the body, tiptoeing, wide gait, broken sleeves and doubled facial features.

**Still open (these become challenges):** palms planting on cartwheels and get-ups; separating fused
garments and hair; hair that both swings and never clips; walks that read as natural, not merely
within the measured bands; smiles that keep the character's design.

## Next challenges (planned)
- **Anime look:** toon shading, outlines and face-angle correction that read as hand-drawn anime at every angle.
- **New characters:** more body types (elderly, very tall, chubby chibi, winged or tailed creatures), each
  with its own images, mesh and failing baseline.
- **More motion:** run, jump, sit, dance, fall and get-up, with contact and penetration checks.
- **Faces:** design-preserving smiles, and lip sync to audio.
- **Cloth:** a skirt and a coat that collide correctly at walk and jump speed.

Contributions of new challenges are welcome: a challenge needs a shareable asset, a failing baseline and a
scorer that fails that baseline and passes a known-good control.
