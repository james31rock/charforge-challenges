# Secondary physics

`physics.resolve_preset(name, measurements, length_m)` resolves `presets.json` from explicit height, mass proxy and measured feature length. Presets cover skirt, coat tail, hoodie hem, drawstrings, sleeves, short/long/twin-tail hair, ears and tail. `hair_profiles.json` contains optional expressive-hair parameter experiments; they are not silently merged into the base presets.

`topology.from_roots_tips` creates ordered chains with pinned roots and optional lateral links. `physics.Lattice.step` advances a fixed step no larger than 1/60 second. Use 1/120 second and a stable frame/substep order for repeatability. `Capsule` supplies explicit body collision proxies. `assess` reports length error, penetration samples, response, settling and explosion checks.

Natural frequency scales as `omega = 2*pi*f*sqrt(reference_length/length)`. Particle stiffness is `m*omega^2`, damping `2*zeta*m*omega`. Semi-implicit integration damps velocity relative to the driven frame, then applies ordered length, angle and collision projections. Virtual segment samples reduce collisions missed between particles.

```python
from toolkit.secondary.physics import resolve_preset, Lattice
from toolkit.secondary.topology import from_roots_tips
preset = resolve_preset('hair_long', {'height_cm': 170, 'mass_proxy_kg': 65}, .35)
topology = from_roots_tips([[0,0,1.6]], [[0,0,1.25]], 5, 'fan', False)
solver = Lattice(**topology, preset=preset)
positions = solver.step(topology['rest'], 1/120)
```

Drive targets from the actual body animation; bake resulting channels only after evaluation. Do not run a second spring solver over baked channels. The public hair geometry sets dimensions in the integration test, and repeated numerical trajectories must match exactly.

Limits: particle/segment samples are not complete surface collision, capsule fitting is external, and pinned roots cannot escape colliders. `garment_surface_gate` remains not tested by proxies. These presets do not certify cloth topology, subframe collisions, engine physics parity or natural-looking sway.
