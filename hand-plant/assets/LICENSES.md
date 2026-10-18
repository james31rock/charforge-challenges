# Asset notices — checked 2026-10-17

* `sample.vrm`: (c) 2022 pixiv Inc., VRM Public License 1.0.
  [Publisher](https://github.com/vrm-c/vrm-specification/tree/master/samples/VRM1_Constraint_Twist_Sample),
  [licence](https://vrm.dev/licenses/1.0/), [English text](https://vrm.dev/licenses/1.0/pdf/en.pdf).
  Original SHA-256: 12c2b97e95e700783a6a550dc0eee2d7880aeedccef9ae67bc4c5a2f0f2631a2.
  Embedded metadata permits corporate commercial use, redistribution and modified
  redistribution. Credit is optional; this package gives it. Prohibited-use fields
  still apply. Full embedded conditions are retained in `vrm-meta.json`.
* `rig.blend`: a modified import of that exact asset, with packed images and
  native rig/weights preserved. Saved UI paths and unused source scripts removed.
  The same VRM licence/metadata conditions apply; it is not MIT.
* `49_06.bvh`, `cartwheel.npz`, baseline retarget and their derivatives:
  CMU Graphics Lab Motion Capture Database, funded by NSF EIA-0196217.
  [CMU terms](http://mocap.cs.cmu.edu/) permit use in research and inclusion in
  commercial products; resale of the motion data itself, including conversions,
  is prohibited. This is a free benchmark bundle; do not sell its motion files.
  Courtesy credit: data obtained from mocap.cs.cmu.edu, NSF EIA-0196217 funding.
  Bruce Hahne's [BVH conversion notice](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database/readme-file-for-the-bvh-conversion-release)
  adds no restrictions. The inserted T-pose is excluded. Fingers are synthetic,
  hand/toe channels noisy. The BVH transport is pinned to mirror commit
  `09a07f54f3bbb58797325f009282d0b2048a2871`, file `data/049/49_06.bvh`.
  BVH SHA-256: 96c742ccc9f412d0d2b3f9d7bb11e6672d99db6c35b093f875fb4b82efa476ce.
* `contact-plan.json`: original reviewed annotations derived from that motion;
  motion terms remain attached. No inferred contact is claimed to be measured.
* `tests/control/*`: original geometric hand fixture and generated motion;
  CC0 asset dedication.

The original and modified model metadata must accompany any redistribution.
The source capture and retarget are not placed under the code's MIT licence.
