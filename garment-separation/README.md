# Separate fused clothing without losing the body

Split `assets/fused.obj` / `fused.npz` into a complete closed body, closed garment
material and closed hair. Preserve the front and side silhouette. The closed
garment is a thick shell: its inner and outer boundaries are separate components.

```powershell
python score.py baseline/parts.npz
python score.py tests/good.npz
```

## Fixture choice and limits

This version deliberately uses a self-made clothed mannequin, not a provider
mesh. Commercial image-to-3D outputs carry provider terms, competitive use/model training. Broad public segmentation reuse was not clearly
cleared, so this is the authorised fallback. See `assets/LICENSES.md`.
The original parts are voxel-unioned to destroy hidden geometry and semantic
boundaries. `make_fixture.py` reproduces them without a service or trained model.
It is an evaluation fixture, not a proposed production character generator.

This gives an actual known hidden-body oracle. Real image-to-3D output has no
such ground truth: a visually plausible inner proxy is an estimate. The public
oracle and seeded solution make this an open diagnostic benchmark, not a secret
competition. Hardcoding the answer is not an accepted reusable solution. A future
release must add held-out bodies and outfits before claiming generalisation.

## What failed before

On four earlier stylized inputs, palette masks plus part completion preserved
silhouette (~0.974–0.991 IoU) but did not return complete anatomical bodies and
semantic garments. A compositional generator returned fused chunks and fragments,
with IoU ~0.727–0.900; neither pipeline passed. Surface partition plus a contracted
inner-body proxy preserves the visible shell but leaves open garments and uncertain
covered anatomy. A watertight wrong part or full vertical occupancy is insufficient.

## Exact pass criteria

Provide finite triangular arrays `body_vertices/body_faces`,
`garment_vertices/garment_faces`, `hair_vertices/hair_faces` in metres, X right,
Y depth, Z up. Vertex indices are zero-based integers. No scale/alignment fit is
applied by the scorer. Same code and thresholds are used for all inputs.

| Check | Required |
|---|---|
| Presence | All three semantic parts |
| Closed geometry | Watertight, consistent winding, positive signed volume, zero nonmanifold edges |
| Semantic surface vs construction oracle | Bidirectional sampled p95 distance <=0.012 h; max <=0.035 h |
| Hidden-body completeness | >=98% of oracle samples within 0.015 h, and >=90% in each occupied height bin |
| Front AND side silhouette IoU | >=0.97 at 768px |

4096 area-weighted surface samples use fixed seed 314159. The 16 height bins
prevent a good torso score from hiding missing feet/head. Vertical occupancy is
reported as a diagnostic only. Oracle distances extend the original L5 topology
and silhouette metrics; they are new benchmark tolerances, not previously
validated biological anatomy limits. Clearance, self-intersection, textures,
cloth suitability and unseen views still need review. Submit a reusable method
and its output arrays, not copied oracle data.
