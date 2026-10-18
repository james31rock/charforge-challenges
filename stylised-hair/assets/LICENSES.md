# Asset notices

The licence boundaries from the existing public head and hand-plant challenges
also apply to these derivatives. Code is MIT; motion data is not MIT.

* **Pippa geometry and vertex colours** (`fixture.npz`, body/hair geometry in
  the drives and baseline): derived from the already public original test-cast
  head/body. The original design and commercial AI-generated mesh are offered
  under the existing [test-cast asset notice](../../one-brow/assets/LICENSES.md).
  Retain that notice, including its restriction against training or building a
  competing 3D-generation model or service. The benchmark does not expand the
  rights granted by that notice. No new generation service was used.
* **Walk and jump motion** (`walk.npz`, `jump.npz`, baseline animation and their
  visual derivatives): CMU Graphics Lab Motion Capture Database, clips 08_03
  and 16_01; funded by NSF EIA-0196217. Courtesy credit: motion data obtained
  from mocap.cs.cmu.edu. [CMU terms](http://mocap.cs.cmu.edu/) allow research
  use and inclusion in commercial products, but prohibit resale of the motion
  data itself, including converted forms. This is a free benchmark bundle.
  Bruce Hahne's [BVH conversion notice](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database/readme-file-for-the-bvh-conversion-release)
  adds no restrictions. Source URLs, hashes and retiming are in `provenance.json`.
  Terms were checked for the existing hand-plant package on 2026-10-17 and
  reused here for the same database and conversion release.
* **Synthetic control** (`../tests/control/*`): original geometric strips,
  sphere and analytic drives authored for scorer validation, dedicated to CC0
  1.0. These have no CMU or provider-generated components.

Meshes are numeric arrays, with no embedded scripts, object names, file paths,
account data or Blender UI state. The full texture/layered face remains available
in the public one-brow challenge; vertex colours here suffice for physics review.
