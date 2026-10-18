# Contributions

Submit a folder containing `submission.json`, an original script or patch,
its declared dependencies/licences, and the output files required below. Include
the exact command, environment versions, seed, runtime and scorer JSON. Provide
both normal-speed and slow review evidence when motion changes. Do not include
credentials, absolute workstation paths, unrelated projects or personal data.

| Task | Required measured output |
|---|---|
| Hand plant | `motion.npz`, preserving supplied names, shapes and 24 Hz sample times |
| One brow | `face.blend`, `feature-free.png`, `face-demo.mp4`, `binding.json`, `atlases/` with the supplied timeline/layout |
| Garment separation | `parts.npz`: body/garment/hair vertices and triangle indices in metres, Z up |

The `submit/` folders contain JSON Schema contracts and examples. Score output
files locally before submitting. Provide permissively licensed code (MIT,
Apache-2.0 or BSD); disclose separate GPL tools and every asset's own terms.
Do not submit copied noncommercial/restricted model implementations or weights.

Maintainers first inspect source/dependencies and output files as data. Outside
code is never executed before review. After review it may be run only in an
isolated scratch virtual environment inside an OS sandbox/container/VM with
network disabled and no credentials or host workspace mounts. A venv alone is
not a security sandbox. Run the trusted scorer on the independently reproduced
output, compare digests and review visible results. Entrant scores are claims
until maintainers reproduce them. No remote contribution endpoint exists yet.
