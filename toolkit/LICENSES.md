# Code and dependency licences

Checked 2026-10-18. All implementation files in this folder are original MIT code or adaptations of that original code, released under the repository MIT licence. No third-party algorithm implementation, model weights, binary application, add-on or character asset is copied into this folder. Adapted source digests are in `SOURCE-MAP.json`; new integration code uses the same MIT terms.

| Import / requirement | Version | Licence and primary source | Role |
|---|---|---|---|
| Python standard library | Python 3.12 | [PSF](https://docs.python.org/3/license.html) | JSON, paths, hashing, tests, processes |
| `numpy` | 2.5.1 | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Numerical arrays and linear algebra |
| `PIL` / Pillow | 12.1.1 | [MIT-CMU / HPND historical notices](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | Image IO and review sheets |
| `cv2` / opencv-python | 5.0.0.93 | [Apache-2.0 core](https://github.com/opencv/opencv/blob/4.x/LICENSE), [wheel notices](https://github.com/opencv/opencv-python/blob/master/LICENSE-3RD-PARTY.txt) | Raster operations and movie decoding |
| `scipy` | 1.18.1 | [BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Existing public benchmark dependencies |
| `trimesh` | 4.12.2 | [MIT](https://github.com/mikedh/trimesh/blob/main/LICENSE.md) | Existing garment benchmark geometry |
| `rtree` | 1.4.1 | [MIT](https://github.com/Toblerity/rtree/blob/main/LICENSE.txt) | Existing garment benchmark indexing |
| libspatialindex | distributed with the Rtree wheel | [MIT](https://github.com/libspatialindex/libspatialindex/blob/main/COPYING) | Native spatial index dependency |
| `bpy`, `bmesh`, `mathutils` | Blender 5.2.1 | [GPL external application](https://www.blender.org/about/license/) | Optional headless adapter host; never copied or redistributed here |
| FFmpeg codec runtime | supplied by the installed OpenCV wheel | [LGPL-2.1 external binary notices](https://github.com/opencv/opencv-python/blob/master/LICENSE-3RD-PARTY.txt) | Optional movie decoding; no codec source or binary is included here |

`IMPORTS.json` checks the complete static import inventory against standard-library, package-local and declared external names. Blender API calls execute in the separately installed application. No GPL implementation is incorporated into these files. Imported numerical/image package source licences are permissive; binary wheel payloads are not uniformly permissive. The optional `requirements-face.txt` references an externally installed OpenCV wheel that includes the LGPL FFmpeg runtime for video decoding. It is separated from the base requirements; do not copy that runtime into an MIT-only distribution. Numerical and still-image APIs do not require movie decoding. Any installed binary wheels keep their own bundled-library notices; this repository does not relicense or redistribute those wheels.

Phonetic cue JSON can be supplied by an external tool; the timing code does not import or bundle it. An optional base-OBJ adapter expects separately licensed CC0 data and does not include a base mesh or add-on. Tests read existing challenge assets in place: their per-asset notices and grants remain controlling. The code's MIT licence does not relicense those assets or permit redistribution of arbitrary user input.
