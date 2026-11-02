# SnowTest

A standalone proving-ground app for falling snow on iPad and iPhone.
It exists to judge one thing in isolation: whether GPU-simulated
snowfall looks right and stays fast on real hardware, before any of
it is considered for a bigger app.

## The approach

The simulation is 100% GPU, built around a single `MTLBuffer`:

- One buffer holds every flake as a `SnowParticle`
  (`packed_float3 position`, `packed_float3 velocity`,
  `float randomOffset`, `float size` — 32 bytes per flake).
- A Metal **compute kernel** updates the buffer **in place** every
  frame: gravity fall at each flake's own terminal velocity,
  sine-wave wind drift with a steady wind push, horizontal wrap,
  and floor collision by respawn (a flake that reaches the ground
  recycles to the top of the volume at a fresh random spot).
- The **render pass reads the same buffer directly** — flakes are
  drawn as point primitives (`MTLPrimitiveTypePoint`) in a single
  draw call, each point masked in the fragment shader into a soft
  round flake. Flakes never read as squares.
- The CPU never touches particle data after seeding it. Per frame
  it only passes small uniforms (dt, time, wind, fall speed) and
  issues one compute dispatch (threadgroup width 64) followed by
  the render pass, in the same command buffer.

## On screen

- A minimal scene: dark sky, one gridded ground plane at y = 0,
  orbit camera (drag to orbit, pinch to zoom).
- Sliders for flake count (1,000–100,000), wind strength, and fall
  speed. Defaults: 30,000 flakes, light wind.
- A live readout of FPS, flake count, and GPU frame time, always
  visible — the numbers this app exists to produce.

## Build

Unsigned IPA via GitHub Actions (`build-ipa.yml`, standard macOS
runner): `xcodebuild` for `iphoneos`, packaged as
`SnowTest-unsigned.ipa`. Bundle id `com.user.snowtest`, universal
(iPhone + iPad), iOS 17+.
