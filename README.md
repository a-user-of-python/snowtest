# SnowTest

A standalone proving-ground app for falling snow on iPad and iPhone.
It exists to judge one thing in isolation: whether GPU-simulated
snowfall looks right and stays fast on real hardware, before any of
it is considered for a bigger app.

## The approach

The simulation is 100% GPU, built around a single `MTLBuffer` of
flakes plus a coarse heightfield for the blanket:

- One buffer holds every flake as a `SnowParticle`
  (`packed_float3 position`, `packed_float3 velocity`,
  `float randomOffset`, `float size`, `float state` — 36 bytes
  per flake). `velocity` is the flake's actual signed velocity;
  `state` is 0 while falling and counts the seconds since it
  settled once it lands.
- A Metal **compute kernel** updates the flake buffer **in place**
  every frame: gravity up to each flake's terminal velocity,
  sine-wave wind drift with a steady wind push, horizontal wrap.
  A flake that reaches the blanket **settles and locks at its
  landing height**, so the blanket fills with particles in layers
  through its whole thickness. Settled flakes recycle to the sky
  after a while (or when buried deep), so the snowfall never dies.
- A second compute pass advances a 256×256 **heightfield** — the
  blanket itself: it grows with the snowfall (rate follows the
  flake count), compacts under its own weight, relaxes slopes
  toward a ~33° angle of repose, creeps downwind and scours in
  strong wind, refills depressions over minutes (trails heal),
  and is pressed down where the walk-mode body stands. At
  ~0.11 m per cell the relief is fine enough for accumulation
  to pile into visible stacks and mounds: a settling flake
  rests on the local surface **plus a per-flake stacking
  offset** (most within millimetres, a few proud by up to
  ~4 cm), so busy landing spots grow knobbly piles on top of
  the settled snow instead of a smooth sheet, while the field
  underneath stays repose-limited and the stacks stay soft.
  The field also **thins toward the edge of the falling
  volume** — growth tapers off and border heights drain gently
  to zero over the outer ~1.5 m — so the blanket ends the way
  snow thins out, not as a full-depth rectangle.
- Ground physics for settled flakes lives in the flake kernel:
  compaction sink, surface-only wind scour (flakes buried more
  than ~5 cm are locked), support loss (a flake whose ground is
  carved away drops), and walk-body displacement.
- A three-pass **visibility culling** stage (v1.4) compacts the
  flakes that can actually be seen into an indirect draw:
  settled flakes buried past the depth where their rendered
  alpha reaches zero (~32 cm), and any flake whose bounding
  sphere misses the camera frustum, are dropped before the
  vertex shader ever sees them. The compaction is order-stable
  (per-group counts, a tiny scan, an ordered write), so
  survivors keep buffer order and nothing shimmers; the visible
  count lands in the indirect-draw arguments on the GPU — the
  CPU never learns it. Burial culling is the constant win once
  a blanket exists; frustum culling varies with the camera and
  bites hardest in walk mode and close zooms.
- The **blanket renders as real geometry** (v1.4): a procedural
  mesh at heightfield resolution whose vertices fetch the field
  and displace to its height, drawn opaque with a depth buffer.
  It has a true top surface, edges that slope down to the
  floor, and walls where a walk trail cuts through it. Its
  shading steps tone in ~5.5 cm horizontal **strata bands** —
  on any sloped face the surface crosses band after band, so
  the build-up reads as stacked layers — with slope shading
  (steep faces cooler and darker) and grain. Where the depth
  is ~0 the same mesh reproduces the bare dark grid floor
  exactly, and four flat apron quads continue that floor
  outside the falling volume. (v1.3 instead painted a snow
  tint on the flat ground plane under sprites floating at
  blanket height with no depth buffer — two parallel sheets
  that read as one hovering slab. The geometry replaced it.)
- The **flake pass reads the same particle buffer** — through
  the culling stage's visible-index list — and draws each
  visible flake as a point primitive (`MTLPrimitiveTypePoint`)
  in a single indirect draw call, masked in the fragment
  shader into a soft round flake. Flakes never read as
  squares. Settled flakes draw at twice the falling diameter
  with a fuller opaque core, and stay fully opaque until
  buried more than ~10 cm; only deeper flakes fade with depth.
  Flakes depth-test against the blanket mesh, so snow inside
  and behind the blanket is occluded by the geometry itself.
- The CPU never touches particle or heightfield data after
  seeding it. Per frame it only passes small uniforms and issues
  two simulation dispatches, the three culling dispatches, and
  the render pass, in the same command buffer.

## On screen

- A minimal scene: dark sky, one gridded ground plane at y = 0,
  and the snow blanket as a displaced surface on top of it.
- Two cameras: **Orbit** (drag to orbit, pinch to zoom) and
  **Walk** — a ground-level body you drag through the snow
  (up/down to move, left/right to turn). Walking presses the
  blanket down, pushes settled flakes aside, kicks some into the
  air, and carves a trail that slowly heals over minutes.
- Sliders for flake count (1,000–5,000,000 on a log scale),
  wind strength, fall speed, and flake size. Flakes are small
  by design (~1–3.6 cm) so the snow reads fine-grained.
  Defaults: 100,000 flakes, light wind. A **Clear snow**
  button restarts the accumulation from bare ground. At the
  5,000,000 ceiling the flake buffer is ~180 MB (plus ~40 MB of
  culling lists), and even after culling the per-flake compute
  update plus heightfield sampling is the dominant frame cost
  — the readout shows the true number, nothing is capped or
  degraded to hide it.
- A live readout of FPS, flake count, and GPU frame time, always
  visible — the numbers this app exists to produce.

## Build

Unsigned IPA via GitHub Actions (`build-ipa.yml`, standard macOS
runner): `xcodebuild` for `iphoneos`, packaged as
`SnowTest-unsigned.ipa`. Bundle id `com.user.snowtest`, universal
(iPhone + iPad), iOS 17+.
