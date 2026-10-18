# Rig failure patterns

These are diagnostic hypotheses and provisional bands. Pair every proposed rule change with an unchanged professional control and a seeded failure.

## FP-01: Coronal knee splay

A garment-width hip estimate can place hips outside the measured leg columns.

Math: hip_knee_ratio = abs(hip_x)/abs(knee_x); coronal_offset = knee_x - lerp(hip_x, ankle_x, height_fraction).

Predictor: Ratio > 1.18, or inward offset > 8 mm and > 0.22 knee radii.

Repair: Fit hips from observed leg columns; attenuate lateral IK poles near extension.

Limit: Hip-normalized gait width can hide an inflated hip estimate; also normalize by leg length.

## FP-02: Off-centre knee pivot

A pivot displaced from the limb section causes asymmetric travel and collapse.

Math: Intersect 24 rays in the bisecting joint plane; divide centroid-to-joint offset by mean radius.

Predictor: Pivot offset > 0.30 radii or flexed section radius < 0.55 rest radius.

Repair: Recenter the joint from the local surface; verify child compensation and deformed volume.

Limit: Open/concave surfaces and invalid fans need unavailable results, not guessed radii.

## FP-03: Tiptoe bind and misplaced ball joint

Aligning a downward ankle-to-toe bind bone to a horizontal vector rotates the sole; a toe-tip pivot cannot roll the forefoot.

Math: ball_station = (ball_forward - heel_forward)/(tip_forward - heel_forward); heel lift follows lever*sin(pitch).

Predictor: Missing toe joint or ball station outside 0.45–0.85 shoe length.

Repair: Preserve the bind sole frame and measure an anatomical ball pivot.

Limit: Oversized or stylized footwear requires measured support, not a universal shoe percentage.

## FP-04: Sleeve weight contamination

Sleeve vertices weighted to pelvis or legs stretch into a web during arm swing.

Math: Count lower-arm vertices with >=0.10 arm share and >=0.05 lower-body share; inspect FK edge stretch.

Predictor: Any such bleed, max bleed >=0.12, p99 edge stretch >2.2, or invalid elbow fan.

Repair: Constrain diffusion by anatomical branches; inspect seams and shoulder transition.

Limit: Edge stretch alone is not visual silhouette approval; fused surfaces can make partitions uncertain.

## FP-05: Hand clearance starvation

A wide garment consumes the rest arm corridor and the hand intersects it in a sagittal swing.

Math: Coronal margin and static hand-surface distance versus declared garment geometry.

Predictor: Rest margin <5 mm, minimum swing clearance <6 mm, or penetration hits.

Repair: Adjust measured arm corridor while preserving reach and retest sleeve deformation.

Limit: Open shells cannot certify volume penetration; small bodies need scale-aware calibration.

## FP-06: Missing hair response

Hair without dynamic chains or sufficient dynamic skin influence cannot respond independently.

Math: Count spring-like hair bones and dynamic hair weight share.

Predictor: No spring bones or dynamic share <0.05.

Repair: Declare hair topology and attachment roots; solve against body colliders.

Limit: Bone names alone do not prove simulation or collision; rigid hairstyles may intentionally have no secondary motion.

## FP-07: Fused hip-fold intersection

A single fused garment/body mesh folds into itself during deep hip flexion.

Math: Count nonadjacent triangle overlaps near the hip in rest and 90-degree flexion.

Predictor: Single fused mesh and intersection-count increase >350.

Repair: Separate meaningful surfaces and complete hidden body geometry before binding.

Limit: Counts depend on tessellation and cannot be universal physical thresholds.

## FP-08: Pelvis reach and oversized sole lever

Straight leg reach clamping and large shoe pitch corrections can pump the pelvis.

Math: vertical_slack = 1 - vertical_distance/leg_length; sole_lever_ratio = sole_lever/leg_length.

Predictor: Slack <0.0085 with lever >0.48, or slack <0.005 with lever >0.35.

Repair: Plan a smooth feasible root curve with explicit reach and contact constraints.

Limit: Static geometry predicts risk only; inspect actual native soles and pelvis trajectories.
