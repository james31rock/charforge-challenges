# Safety and reviewer procedure

The scorers never import, evaluate or execute an entrant's Python or patches.
NPZ uses `allow_pickle=False`, bounded expanded size and finite numeric arrays.
The C7 rig is fixed; submitted files contain pose matrices only. C8 opens a
submitted blend as data using Blender factory startup and `-Y`; only our trusted
measurement script runs. Blender simple drivers and native parsers still operate.
`-Y` is not a sandbox. Treat blend/media/parser inputs as untrusted and use an
OS-isolated process with no secrets for outside submissions. Reject external
libraries, external texture dependencies and non-simple driver code on review.

No scorer accesses the network or installs dependencies. Render intermediates
go into a temporary directory and are removed even on failure. Set
`CHARFORGE_SCRATCH` to a disposable location outside any source checkout. Each
Blender call has a 14-minute timeout and uses CPU rendering with two threads.

Rendered movie files can be fabricated or disconnected from the submitted head.
C8 independently renders 22 static/expression views, but the full 240-frame
movie must still be reproduced from the reviewed recipe before scoreboard
acceptance. Pixel gates do not establish global likeness, anatomical correctness
or temporal continuity. Review these explicitly; do not weaken failed gates.
