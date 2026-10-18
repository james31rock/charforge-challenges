# Licence boundaries (checked 2026-10-17)

The MIT licence covers original code only, . In-house
gate subsets are recorded in `VENDORED.json`; no external algorithm source or
model weights were copied. Each challenge's `assets/LICENSES.md` enumerates its
inputs and derivative outputs. Their terms also apply to baselines and controls.

Blender 5.2.1 is a separate GPL application, not redistributed. NumPy/SciPy use
BSD licences; Pillow uses HPND; OpenCV uses Apache-2.0; trimesh uses MIT; Rtree
uses MIT and its libspatialindex dependency uses MIT. These are external runtime
dependencies, not copied into the code licence. See official repositories:
[Blender](https://www.blender.org/about/license/),
[NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt),
[SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt),
[Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE),
[OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE),
[trimesh](https://github.com/mikedh/trimesh/blob/main/LICENSE.md),
[Rtree](https://github.com/Toblerity/rtree/blob/main/LICENSE.txt).

No neural model is included. Provider-created head assets are AI-generated;
they are not blanket MIT/CC0 assets and are not offered as training data.
