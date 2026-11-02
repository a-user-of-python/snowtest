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
- Ground physics for settled flakes lives in the flake kernel:
  compaction sink, surface-only wind scour (flakes buried more
  than ~5 cm are locked), support loss (a flake whose ground is
  carved away drops), and walk-body displacement.
- The **render pass reads the same buffers directly** — flakes are
  drawn as point primitives (`MTLPrimitiveTypePoint`) in a single
  draw call, each point masked in the fragment shader into a soft
  round flake. Flakes never read as squares. Settled flakes draw
  slightly larger and brighter; buried flakes fade with depth so
  the layers read as strata.
- The CPU never touches particle or heightfield data after
  seeding it. Per frame it only passes small uniforms and issues
  two compute dispatches followed by the render pass, in the same
  command buffer.

## On screen

- A minimal scene: dark sky, one gridded ground plane at y = 0.
- Two cameras: **Orbit** (drag to orbit, pinch to zoom) and
  **Walk** — a ground-level body you drag through the snow
  (up/down to move, left/right to turn). Walking presses the
  blanket down, pushes settled flakes aside, kicks some into the
  air, and carves a trail that slowly heals over minutes.
- Sliders for flake count (1,000–1,000,000 on a log scale),
  wind strength, fall speed, and flake size. Flakes are small
  by design (~1–3.6 cm) so the snow reads fine-grained.
  Defaults: 100,000 flakes, light wind. A **Clear snow**
  button restarts the accumulation from bare ground.
- A live readout of FPS, flake count, and GPU frame time, always
  visible — the numbers this app exists to produce.

## Build

Unsigned IPA via GitHub Actions (`build-ipa.yml`, standard macOS
runner): `xcodebuild` for `iphoneos`, packaged as
`SnowTest-unsigned.ipa`. Bundle id `com.user.snowtest`, universal
(iPhone + iPad), iOS 17+.
