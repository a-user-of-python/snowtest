// Shaders.metal — SnowTest
//
// GPU snow, following the compute-particle design this test app
// exists to prove out, extended in v1.1 with a living blanket,
// in v1.2 with finer snow, far more of it, and real stacking,
// in v1.3 with a 5M flake ceiling and ground cover that
// actually covers, and in v1.4 with a blanket that is real
// geometry instead of two flat layers, plus GPU visibility
// culling so invisible flakes stop costing vertex work:
//
//   1. `updateHeightfield` (compute) advances a 256x256
//      heightfield over the falling volume — the blanket. Per
//      frame it grows with the snowfall, compacts under its own
//      weight, relaxes slopes toward a ~33 degree angle of
//      repose, drifts and scours in strong wind, fills depressions
//      back in over minutes (trail regeneration: snowfall refills
//      low spots faster, a slow wind-drift infill works even when
//      calm), and is pressed down (with a raised rim) where the
//      walk-mode body stands. Two buffers ping-pong so the pass is
//      race-free. Since v1.4 the field also THINS toward the edge
//      of the falling volume: growth tapers off over the outer
//      ~1.5 m and a gentle drain pulls border heights toward
//      zero, so the blanket's depth reaches 0 at its boundary
//      instead of ending at full depth on a hard rectangle.
//   2. `updateSnow` (compute) advances every flake IN PLACE in one
//      shared MTLBuffer. Falling flakes integrate a real velocity
//      (gravity up to their terminal speed, wind drift, body push).
//      A flake that reaches the blanket surface SETTLES: it locks
//      at its landing height — the local surface plus a per-flake
//      stacking offset, so it rests on the micro-relief of the
//      flakes already down and busy landing spots grow knobbly
//      stacks instead of a smooth sheet. The blanket fills with
//      particles in layers through its whole thickness. Settled
//      flakes sink slowly with compaction, drop if their support
//      is carved away, can be scoured back into the air by strong
//      they lie at the surface, are displaced by the walk body
//      (slid aside or kicked airborne), and eventually recycle to
//      the sky so the snowfall never dies.
//   3. `cullFlakesCount` / `cullFlakesScan` / `cullFlakesWrite`
//      (compute, v1.4) build the list of flakes that can actually
//      be seen this frame: settled flakes buried past the depth
//      where their rendered alpha reaches zero, and any flake
//      whose bounding sphere misses the camera frustum, are
//      dropped. It is a stable stream compaction (per-threadgroup
//      counts, one tiny scan, then an ordered write), so the
//      surviving flakes keep buffer order — no per-frame
//      reshuffle shimmer — and it ends by writing the visible
//      count into an indirect-draw arguments buffer. The CPU
//      never learns how many flakes were visible.
//   4. `surfaceVertex` / `surfaceFragment` (render, v1.4) draw
//      the blanket itself as real geometry: a procedural grid at
//      heightfield resolution whose vertices fetch the field and
//      displace to its height, with normals from the field. The
//      blanket now has a true top surface, edges that slope down
//      to the floor, and walls where a walk trail cuts through
//      it. Shading is height-banded — horizontal strata tones
//      step every few centimetres of altitude, so a wall crossing
//      several bands shows the layers the snow built up in —
//      plus slope shading (steep faces cooler and darker) and
//      grain. Where the depth is ~0 the same shading reproduces
//      the bare dark grid floor exactly, so fresh ground, the
//      post-Clear floor and thin spots still read as ground.
//      This replaced the v1.3 approach (a snow tint painted on
//      the flat ground plane UNDER sprites floating at blanket
//      height, with no depth buffer): two parallel sheets at two
//      heights, which read as one hovering slab whose border
//      stuck out past the snow. The surface is opaque and writes
//      depth, so flakes below it are occluded by geometry.
//   5. `apronVertex` / `apronFragment` (render, v1.4) draw the
//      bare floor OUTSIDE the falling volume: four flat quads
//      framing the surface mesh out to +/-40 m, same grid look.
//   6. `snowVertex` / `snowFragment` (render) read the SAME
//      particle buffer — through the culling pass's visible-index
//      list — and draw each visible flake as a single point
//      primitive, masked into a soft round flake — precipitation
//      must never read as squares. Settled flakes draw larger
//      (2x diameter) with a fuller opaque core so the blanket
//      surface tiles over instead of peppering dark ground
//      through the gaps between points; flakes within ~10 cm
//      of the surface stay fully opaque, and only deeper
//      buried flakes fade with depth, so the layers still read
//      as strata instead of showing through.
//
// No particle or heightfield data ever comes back to the CPU; the
// CPU only passes uniforms and seeds/resets buffers.
//
// SnowParticle layout is shared byte-for-byte with SnowRenderer.swift:
//   position      bytes  0..<12  (packed_float3)
//   velocity      bytes 12..<24  (packed_float3; the flake's actual
//                                 velocity, signed; (0,0,0) at rest)
//   randomOffset  bytes 24..<28  (float; per-flake identity hash seed)
//   size          bytes 28..<32  (float; world-space flake diameter)
//   state         bytes 32..<36  (float; 0 = falling, > 0 = settled
//                                 for that many seconds)
// Total stride: 36 bytes.

#include <metal_stdlib>
using namespace metal;

// Heightfield resolution (must match SnowRenderer.heightfieldGrid).
// 256x256 over +/-14 m: ~0.11 m cells, fine enough that local
// relief — stacks, rims, trail walls — survives at hand scale.
#define GRID_N 256

struct SnowParticle {
    packed_float3 position;
    packed_float3 velocity;
    float randomOffset;
    float size;
    float state;
};

// Everything the particle simulation needs for one step. Plain
// floats only, so the Swift mirror struct has an identical layout.
struct SimUniforms {
    float dt;           // seconds since last frame (clamped on CPU)
    float time;         // seconds since the app started
    float windStrength; // sideways push, metres per second
    float fallSpeed;    // multiplier on each flake's fall velocity
    float halfExtent;   // falling volume spans +/- halfExtent in x/z
    float height;       // respawn height; the floor is y = 0
    float capHeight;    // blanket depth cap, metres
    float bodyX;        // walk-mode body position (x)
    float bodyZ;        // walk-mode body position (z)
    float bodyRadius;   // walk-mode body radius, metres
    float bodyActive;   // 1 while walk mode is on, else 0
};

// Everything the blanket pass needs for one step.
struct HFUniforms {
    float dt;
    float growthRate;   // blanket growth, m/s at noise = 1 (count-scaled on CPU)
    float capHeight;    // blanket depth cap, metres
    float windStrength; // for surface drift + scour
    float bodyX;
    float bodyZ;
    float bodyRadius;
    float bodyActive;   // 1 while walk mode is on, else 0
    float bodyMoving;   // 1 while the body is actually moving
    float pressDepth;   // depth the body compacts snow down to
    float halfExtent;
    float time;
};

// Everything the render passes need for one frame.
struct FrameUniforms {
    float4x4 viewProjection;
    float pointScale;   // pixels per world unit at clip.w = 1
    float halfExtent;   // for heightfield sampling in the vertex shader
    float sizeScale;    // user flake-size multiplier (1 = as seeded)
};

// Everything the visibility-culling passes need (v1.4). The
// view-projection matrix is the same one the render passes use;
// the frustum planes are derived from its rows in the kernel.
struct CullUniforms {
    float4x4 viewProjection;
    float halfExtent;   // for heightfield sampling (burial test)
    float sizeScale;    // user flake-size multiplier
    uint flakeCount;
    uint groupCount;    // ceil(flakeCount / 256)
};

// Indirect-draw arguments, written by cullFlakesScan and consumed
// by drawPrimitives(indirectBuffer:) for the flake points. Layout
// must match MTLDrawPrimitivesIndirectArguments exactly.
struct DrawArgs {
    uint vertexCount;
    uint instanceCount;
    int vertexStart;
    uint baseInstance;
};

// Burial fade (shared by the vertex shader and the culling pass,
// so they can never disagree): a settled flake's rendered alpha
// starts fading once it is BURY_FADE_START metres below the local
// blanket surface and reaches exactly zero at BURY_CULL_DEPTH.
// Past that depth the flake contributes nothing on screen and the
// culling pass drops it from the draw entirely.
#define BURY_FADE_START 0.10
#define BURY_FADE_SPAN 0.22
#define BURY_CULL_DEPTH (BURY_FADE_START + BURY_FADE_SPAN)

// Cheap deterministic hash -> [0, 1). Used so a recycled flake gets a
// fresh, time-varying respawn spot instead of the same place forever,
// and to derive stable per-flake values from its randomOffset.
static float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

// Bilinear sample of the blanket heightfield at world (x, z).
// The grid spans +/- halfExtent; cell (ix, iz) centres sit at
// ((ix/(N-1))*2-1)*halfExtent, matching updateHeightfield.
static float sampleHeight(const device float *H, float halfExtent, float x, float z) {
    float fx = clamp((x / halfExtent) * 0.5 + 0.5, 0.0, 1.0) * float(GRID_N - 1);
    float fz = clamp((z / halfExtent) * 0.5 + 0.5, 0.0, 1.0) * float(GRID_N - 1);
    int ix = int(floor(fx));
    int iz = int(floor(fz));
    int ix1 = min(ix + 1, GRID_N - 1);
    int iz1 = min(iz + 1, GRID_N - 1);
    float tx = fx - float(ix);
    float tz = fz - float(iz);
    float h00 = H[iz * GRID_N + ix];
    float h10 = H[iz * GRID_N + ix1];
    float h01 = H[iz1 * GRID_N + ix];
    float h11 = H[iz1 * GRID_N + ix1];
    return mix(mix(h00, h10, tx), mix(h01, h11, tx), tz);
}

// ----------------------------------------------------------------
// Blanket pass: growth, compaction, slope relaxation, wind drift
// and scour, and the walk body's footprint press. Jacobi-style:
// reads `src`, writes `dst`, so there are no races.
// ----------------------------------------------------------------
kernel void updateHeightfield(const device float *src [[buffer(0)]],
                              device float *dst [[buffer(1)]],
                              const device float *noise [[buffer(2)]],
                              constant HFUniforms &u [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    int ix = int(gid.x);
    int iz = int(gid.y);
    int idx = iz * GRID_N + ix;
    float cell = 2.0 * u.halfExtent / float(GRID_N);
    float hC = src[idx];
    float h = hC;

    // This cell's world position, and its edge taper (v1.4): the
    // blanket thins out toward the boundary of the falling volume
    // instead of ending at full depth. taper is 0 at the outermost
    // cells and 1 once ~1.5 m inside; the growth term below scales
    // by it, and a gentle drain at the end of the pass pulls
    // border heights toward zero, so the field's depth reaches 0
    // exactly where the snow area ends — the displaced surface
    // then slopes down and lands on the bare floor.
    float cx = (float(ix) / float(GRID_N - 1) * 2.0 - 1.0) * u.halfExtent;
    float cz = (float(iz) / float(GRID_N - 1) * 2.0 - 1.0) * u.halfExtent;
    float edgeDist = min(u.halfExtent - abs(cx), u.halfExtent - abs(cz));
    float taper = smoothstep(0.0, 1.5, edgeDist);

    // 1. Growth: fresh snowfall settles onto the blanket. The
    //    per-cell noise keeps the surface from rising as one flat
    //    sheet; the cap keeps it sane for the scene; the edge
    //    taper (v1.4) feathers the growth out at the boundary.
    h = min(h + u.growthRate * noise[idx] * taper * u.dt, u.capHeight);

    // 2. Compaction: snow sinks under its own weight, deeper snow
    //    a little faster. This is what lets the blanket very
    //    slowly settle after the snowfall eases off.
    float compactRate = 0.0016 + 0.0022 * clamp(h / u.capHeight, 0.0, 1.0);
    h -= h * compactRate * u.dt;

    // Neighbours; border cells mirror themselves, so no snow
    // flows in or out across the edge of the volume.
    float hL = src[iz * GRID_N + max(ix - 1, 0)];
    float hR = src[iz * GRID_N + min(ix + 1, GRID_N - 1)];
    float hD = src[max(iz - 1, 0) * GRID_N + ix];
    float hU = src[min(iz + 1, GRID_N - 1) * GRID_N + ix];

    // 3. Wind on the blanket (only in real wind): mass creeps
    //    downwind (+x), and faces exposed to the wind erode a
    //    little faster — drifts migrate and sharp windward faces
    //    soften while the bulk stays put.
    if (u.windStrength > 4.0) {
        float excessWind = u.windStrength - 4.0;
        float driftK = excessWind * 0.008;
        h += (hL - hC) * driftK * u.dt;
        float exposure = clamp((hC - hL) / (2.0 * cell), 0.0, 1.0);
        h -= hC * excessWind * 0.002 * (0.3 + exposure) * u.dt;
    }

    // 4. Slope relaxation toward the angle of repose (~33
    //    degrees): wherever a face is steeper than repose, some
    //    snow slumps downhill. This is what turns piles and trail
    //    walls into soft, natural profiles instead of spikes.
    float maxDiff = 0.6494 * cell; // tan(33 deg) * cell size
    float relaxF = clamp(u.dt * 10.0, 0.0, 0.15);
    float net = 0.0;
    float neighbours[4] = {hL, hR, hD, hU};
    for (int i = 0; i < 4; i++) {
        float n = neighbours[i];
        float outExcess = (hC - n) - maxDiff;
        if (outExcess > 0.0) net -= outExcess * relaxF;
        float inExcess = (n - hC) - maxDiff;
        if (inExcess > 0.0) net += inExcess * relaxF;
    }
    h += net;

    // 5. Depression refill — trail regeneration. Real trails fill
    //    back in: fresh snow settles preferentially into low
    //    spots, and wind drift keeps levelling them even when the
    //    snowfall is light. A cell sitting below the mean of its
    //    neighbours gains a fraction of that deficit: fast while
    //    snow falls (timescale ~3 minutes), slow when calm
    //    (~12 minutes, drift only). It composes with the slope
    //    relaxation above — relaxation slumps steep faces, this
    //    levels gentle bowls — and it can never overshoot the
    //    neighbours' mean, so the trail melts away asymptotically
    //    instead of bouncing back. Settled flakes buried by the
    //    rising floor fade and recycle; fresh arrivals form the
    //    healed surface.
    float meanN = (hL + hR + hD + hU) * 0.25;
    float deficit = meanN - hC;
    if (deficit > 0.0) {
        float refillF = clamp(u.dt * (u.growthRate * 2.5 + 0.0012), 0.0, 0.02);
        h += deficit * refillF;
    }

    // 6. The walk body presses the blanket down underfoot and
    //    piles a small rim around the pressed disc while moving.
    if (u.bodyActive > 0.5) {
        float d = distance(float2(cx, cz), float2(u.bodyX, u.bodyZ));
        float pressR = u.bodyRadius + 0.10;
        if (d < pressR && h > u.pressDepth) {
            float w = 1.0 - smoothstep(pressR - 0.12, pressR, d);
            h = mix(h, u.pressDepth, w);
        } else if (u.bodyMoving > 0.5 && d >= pressR && d < pressR + 0.30) {
            float w = 1.0 - (d - pressR) / 0.30;
            h += 0.030 * w * u.dt;
        }
    }

    // 7. Edge drain (v1.4): in the tapered border ring, heights
    //    decay toward zero, so any snow the relaxation pushes
    //    outward (or that built up before the taper existed)
    //    melts away at the boundary and the blanket's edge is a
    //    thinning skirt, not a wall. Settled flakes standing on
    //    drained ground lose their support and drop/recycle via
    //    the existing particle rules.
    h *= 1.0 - (1.0 - taper) * min(u.dt * 0.8, 0.05);

    dst[idx] = clamp(h, 0.0, u.capHeight);
}

// ----------------------------------------------------------------
// Particle pass: falling physics, settling into layers, ground
// physics for settled flakes, and walk-body displacement.
// ----------------------------------------------------------------
kernel void updateSnow(device SnowParticle *particles [[buffer(0)]],
                       constant SimUniforms &u [[buffer(1)]],
                       const device float *heightfield [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    SnowParticle p = particles[id];
    float ro = p.randomOffset;

    // Stable per-flake values derived from its random offset.
    float baseFall = 0.6 + fract(ro * 1.61803) * 1.0;   // terminal speed at 1x, m/s
    // Stacking rest height: a settling flake comes to rest ON the
    // micro-relief of the flakes already down, not on the smooth
    // field itself, so its rest height is the local surface plus
    // this per-flake offset. The distribution is one-sided and
    // bottom-heavy (stackH^2): most flakes sit within millimetres
    // of the surface, a few ride proud by up to ~4 cm — the
    // granular profile of the top of a real pile. The field
    // underneath stays repose-limited, so busy landing spots
    // (rims, drift crests, resettle clusters) read as soft
    // knobbly stacks at hand scale, never needles.
    float stackH = hash1(ro * 3.71);
    float restOffset = -0.006 + stackH * stackH * 0.048; // -6 mm .. +4.2 cm
    float dwellLimit = 30.0 + fract(ro * 0.9173) * 40.0; // settled lifetime, 30..70 s
    bool kickable = fract(ro * 0.613) < 0.30;           // 30% of flakes kick when displaced

    if (p.state <= 0.0) {
        // ---------------- FALLING ----------------
        // Gravity pulls the fall speed toward (never past) the
        // flake's terminal velocity. A flake kicked into the air
        // starts with upward velocity and arcs back down.
        float terminal = baseFall * u.fallSpeed;
        p.velocity.y = max(p.velocity.y - 4.0 * u.dt, -terminal);

        // Horizontal velocity relaxes into the wind: a steady +x
        // push (this is what visibly bends the fall) plus a gentle
        // per-flake sway. Kicked flakes bleed their sideways
        // velocity off into this drift within about half a second.
        float swayX = sin(u.time * 0.9 + ro * 17.0);
        float swayZ = cos(u.time * 0.7 + ro * 29.0);
        float driftX = u.windStrength + swayX * 0.35;
        float driftZ = swayZ * 0.35;
        float blend = min(1.0, u.dt * 2.5);
        p.velocity.x += (driftX - p.velocity.x) * blend;
        p.velocity.z += (driftZ - p.velocity.z) * blend;

        p.position.x += p.velocity.x * u.dt;
        p.position.y += p.velocity.y * u.dt;
        p.position.z += p.velocity.z * u.dt;

        // Wrap horizontally so wind and sway never empty the volume.
        float span = 2.0 * u.halfExtent;
        float wrappedX = fmod(p.position.x + u.halfExtent, span);
        if (wrappedX < 0.0) wrappedX += span;
        p.position.x = wrappedX - u.halfExtent;
        float wrappedZ = fmod(p.position.z + u.halfExtent, span);
        if (wrappedZ < 0.0) wrappedZ += span;
        p.position.z = wrappedZ - u.halfExtent;

        // The walk body shoulders falling flakes aside too (nearly
        // free: the same distance test as the settled branch).
        if (u.bodyActive > 0.5 && p.position.y < 2.0) {
            float bdx = p.position.x - u.bodyX;
            float bdz = p.position.z - u.bodyZ;
            float bDist = length(float2(bdx, bdz));
            float bodyR = u.bodyRadius + 0.10;
            if (bDist < bodyR && bDist > 1e-4) {
                float s = bodyR / bDist;
                p.position.x = u.bodyX + bdx * s;
                p.position.z = u.bodyZ + bdz * s;
            }
        }

        // Landing on the blanket surface: the flake settles and
        // LOCKS at this height — this is how the blanket fills
        // with particles in layers as the surface rises past
        // earlier arrivals.
        float landY = sampleHeight(heightfield, u.halfExtent,
                                   p.position.x, p.position.z) + restOffset;
        if (p.position.y <= landY) {
            p.position.y = landY;
            p.velocity.x = 0.0;
            p.velocity.y = 0.0;
            p.velocity.z = 0.0;
            p.state = 0.001; // settled, age ~0
        }
    } else {
        // ---------------- SETTLED ----------------
        float age = p.state + u.dt;
        p.state = age;
        float surfY = sampleHeight(heightfield, u.halfExtent,
                                   p.position.x, p.position.z) + restOffset;
        float burial = surfY - p.position.y; // > 0: surface has risen above this flake

        float bdx = p.position.x - u.bodyX;
        float bdz = p.position.z - u.bodyZ;
        float bDist = length(float2(bdx, bdz));
        float bodyR = u.bodyRadius + 0.10;

        if (u.bodyActive > 0.5 && bDist < bodyR && burial < 0.15) {
            // The body is standing in this flake: push it out of
            // the body's volume. Only near-surface flakes are
            // displaced — deep layers stay buried (they lose their
            // support to the press instead and drop, below). Some
            // flakes kick into the air and flutter back down; the
            // rest slide to the rim and resettle there — together
            // they carve the trail.
            float dirx = bDist > 1e-4 ? bdx / bDist : cos(ro * 39.0);
            float dirz = bDist > 1e-4 ? bdz / bDist : sin(ro * 39.0);
            if (kickable) {
                float h1 = fract(ro * 0.731);
                float h2 = fract(ro * 0.357);
                p.position.x = u.bodyX + dirx * (bodyR + 0.03);
                p.position.z = u.bodyZ + dirz * (bodyR + 0.03);
                p.velocity.x = dirx * (0.6 + 1.1 * h1);
                p.velocity.y = 1.0 + 1.4 * h2;
                p.velocity.z = dirz * (0.6 + 1.1 * h1);
                p.state = 0.0; // airborne again
            } else {
                float rim = bodyR + 0.02 + fract(ro * 0.377) * 0.06;
                p.position.x = u.bodyX + dirx * rim;
                p.position.z = u.bodyZ + dirz * rim;
                p.position.y = sampleHeight(heightfield, u.halfExtent,
                                            p.position.x, p.position.z) + restOffset;
            }
        } else if (u.windStrength > 5.0 && burial < 0.05 &&
                   hash1(ro * 53.1 + u.time * 13.7) < (u.windStrength - 5.0) * 0.06 * u.dt) {
            // Wind scour: only flakes lying AT the surface (buried
            // less than 5 cm) can be lifted; deeper layers are
            // locked. A lifted flake blows downwind and resettles.
            float h1 = hash1(ro * 11.3 + u.time * 0.9);
            p.velocity.x = u.windStrength * (0.5 + 0.4 * h1);
            p.velocity.y = 0.7 + 0.9 * h1;
            p.velocity.z = (h1 - 0.5) * 0.6;
            p.state = 0.0; // airborne again
        } else if (age > dwellLimit || burial > 0.30) {
            // Recycle to the sky: this flake's turn in the blanket
            // is over (lived its settled lifetime, or buried deep
            // enough that it would never be seen again). Recycling
            // is what keeps the snowfall alive forever.
            float r1 = hash1(ro * 91.7 + u.time * 0.618);
            float r2 = hash1(ro * 47.3 + u.time * 1.313);
            float r3 = hash1(ro * 13.1 + u.time * 0.917);
            p.position.x = (r1 * 2.0 - 1.0) * u.halfExtent;
            p.position.z = (r2 * 2.0 - 1.0) * u.halfExtent;
            p.position.y = u.height + r3 * 1.5;
            p.velocity.x = 0.0;
            p.velocity.y = -baseFall;
            p.velocity.z = 0.0;
            p.state = 0.0;
        } else if (surfY < p.position.y - 0.06) {
            // Support fell away under this flake — the body carved
            // through, or the slope relaxed out from under it.
            // Drop as a falling flake from here; gravity takes it
            // to the new surface within a few frames.
            p.velocity.x = 0.0;
            p.velocity.y = 0.0;
            p.velocity.z = 0.0;
            p.state = 0.0;
        } else {
            // Resting in its layer: sink slowly with compaction,
            // at the same depth-dependent rate the blanket itself
            // uses, so a flake tracks its own stratum as it densifies.
            float compactRate = 0.0016 + 0.0022 * clamp(p.position.y / u.capHeight, 0.0, 1.0);
            p.position.y -= p.position.y * compactRate * u.dt;
        }
    }

    particles[id] = p;
}

// ----------------------------------------------------------------
// Visibility culling (v1.4): a stable stream compaction in three
// tiny passes. Pass 1 evaluates every flake and prefix-sums the
// visibility flags inside each 256-thread group; pass 2 (a single
// thread) prefix-sums the group totals and writes the visible
// count into the indirect-draw arguments; pass 3 scatters each
// visible flake's index into the draw list at its summed rank.
// Survivors keep their buffer order, so overlapping flakes blend
// in the same order every frame — no reshuffle shimmer. A flake
// is visible iff:
//   - it is not settled so deep that its rendered alpha is zero
//     (burial >= BURY_CULL_DEPTH — exactly the depth where the
//     vertex shader's burial fade reaches 0), and
//   - its bounding sphere (radius = its drawn radius, so the
//     settled 2x size counts) intersects the camera frustum,
//     tested exactly against the six planes of the view-
//     projection matrix (Metal clip space: x,y in [-w, w],
//     z in [0, w]).
// Buried settled flakes are the constant win once a blanket
// exists — at the 5M ceiling with the blanket at cap, a large
// share of settled flakes sit below the fade range and used to
// pay full vertex + fragment cost for zero pixels. Frustum
// culling varies with the camera: it removes everything behind
// or beside the view, which in walk mode and close zooms is
// most of the volume.
// ----------------------------------------------------------------

static bool flakeVisible(SnowParticle p, const device float *heightfield,
                         constant CullUniforms &u) {
    bool settled = p.state > 0.0;
    if (settled) {
        float surfY = sampleHeight(heightfield, u.halfExtent,
                                   p.position.x, p.position.z);
        if (surfY - p.position.y >= BURY_CULL_DEPTH) return false;
    }
    float4x4 vp = u.viewProjection;
    // Rows of the VP matrix: rowK picks clip component K.
    float4 row0 = float4(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
    float4 row1 = float4(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
    float4 row2 = float4(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    float4 row3 = float4(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);
    float4 planes[6];
    planes[0] = row3 + row0; // left:   x >= -w
    planes[1] = row3 - row0; // right:  x <=  w
    planes[2] = row3 + row1; // bottom: y >= -w
    planes[3] = row3 - row1; // top:    y <=  w
    planes[4] = row2;        // near:   z >= 0
    planes[5] = row3 - row2; // far:    z <=  w
    float r = 0.5 * p.size * u.sizeScale * (settled ? 2.0 : 1.0);
    for (int i = 0; i < 6; i++) {
        float3 n = planes[i].xyz;
        float dist = (dot(n, p.position) + planes[i].w)
                   / max(length(n), 1e-6);
        if (dist < -r) return false;
    }
    return true;
}

kernel void cullFlakesCount(const device SnowParticle *particles [[buffer(0)]],
                            const device float *heightfield [[buffer(1)]],
                            constant CullUniforms &u [[buffer(2)]],
                            device uint *rankScratch [[buffer(3)]],
                            device uint *groupTotals [[buffer(4)]],
                            uint id [[thread_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint gid [[threadgroup_position_in_grid]]) {
    uint flag = 0;
    if (id < u.flakeCount &&
        flakeVisible(particles[id], heightfield, u)) {
        flag = 1;
    }
    // Hillis-Steele inclusive scan of the flags in threadgroup
    // memory. Every dispatched thread takes part (the caller
    // dispatches whole groups), so the barriers are safe.
    threadgroup uint shared[256];
    shared[tid] = flag;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < 256; offset <<= 1) {
        uint add = (tid >= offset) ? shared[tid - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (id < u.flakeCount) {
        // Inclusive rank (1-based) for visible flakes, 0 for
        // culled ones — pass 3 tells them apart by the zero.
        rankScratch[id] = flag ? shared[tid] : 0;
    }
    if (tid == 255) {
        groupTotals[gid] = shared[255];
    }
}

kernel void cullFlakesScan(const device uint *groupTotals [[buffer(0)]],
                           device uint *groupOffsets [[buffer(1)]],
                           device DrawArgs *args [[buffer(2)]],
                           constant CullUniforms &u [[buffer(3)]]) {
    // One thread: at most ceil(5M / 256) = 19,532 additions.
    uint sum = 0;
    for (uint g = 0; g < u.groupCount; g++) {
        groupOffsets[g] = sum;
        sum += groupTotals[g];
    }
    args->vertexCount = sum;
    args->instanceCount = 1;
    args->vertexStart = 0;
    args->baseInstance = 0;
}

kernel void cullFlakesWrite(const device uint *rankScratch [[buffer(0)]],
                            const device uint *groupOffsets [[buffer(1)]],
                            device uint *visibleIndices [[buffer(2)]],
                            constant CullUniforms &u [[buffer(3)]],
                            uint id [[thread_position_in_grid]],
                            uint gid [[threadgroup_position_in_grid]]) {
    if (id >= u.flakeCount) return;
    uint rank = rankScratch[id];
    if (rank != 0) {
        visibleIndices[groupOffsets[gid] + rank - 1] = id;
    }
}

struct FlakeOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float shade;
    float alpha;
    float settledMix;   // 1 for settled flakes, 0 for falling
};

vertex FlakeOut snowVertex(const device SnowParticle *particles [[buffer(0)]],
                           constant FrameUniforms &f [[buffer(1)]],
                           const device float *heightfield [[buffer(2)]],
                           const device uint *visibleIndices [[buffer(3)]],
                           uint vid [[vertex_id]]) {
    // Since v1.4 the draw is indirect over the culling pass's
    // visible list: vertex_id indexes the survivors, not the
    // particle buffer directly.
    SnowParticle p = particles[visibleIndices[vid]];
    bool settled = p.state > 0.0;
    float4 clip = f.viewProjection * float4(p.position, 1.0);
    FlakeOut out;
    out.position = clip;
    // Perspective point size: world diameter scaled by the frame's
    // point scale and divided by depth (clip.w = -viewZ). Settled
    // flakes draw at twice the diameter so the blanket surface
    // reads as packed snow cover rather than sparse points (1.5x
    // in v1.2 — still sparse enough at grazing angles for the
    // dark ground to pepper through). Clamped so a flake next to
    // the camera cannot balloon and distant flakes never vanish
    // below one pixel.
    float depth = max(clip.w, 0.1);
    float sizeScale = settled ? 2.0 : 1.0;
    out.pointSize = clamp(p.size * sizeScale * f.sizeScale * f.pointScale / depth, 1.0, 64.0);
    // Slight per-flake brightness variation so the field does not
    // read as one flat texture; settled snow a touch brighter.
    out.shade = 0.78 + 0.22 * fract(p.randomOffset * 7.31);
    if (settled) out.shade = min(out.shade + 0.10, 1.05);
    // Burial fade: a flake well below the local blanket surface
    // fades out over BURY_FADE_SPAN metres of burial, so deep
    // layers read as dimmer strata at grazing angles instead of
    // showing through the surface. The fade only starts past
    // BURY_FADE_START of burial: the whole near-surface layer —
    // the flakes doing the covering — stays fully opaque, and
    // only genuinely deep snow dims. At BURY_CULL_DEPTH the alpha
    // is exactly zero and the culling pass no longer draws the
    // flake at all (same constants, so the two never disagree).
    float surfY = sampleHeight(heightfield, f.halfExtent, p.position.x, p.position.z);
    float burial = surfY - p.position.y;
    float fade = clamp(1.0 - max(burial - BURY_FADE_START, 0.0) / BURY_FADE_SPAN, 0.0, 1.0);
    out.alpha = (settled ? 1.0 : 0.9) * fade;
    out.settledMix = settled ? 1.0 : 0.0;
    return out;
}

fragment float4 snowFragment(FlakeOut in [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
    // Mask the square point into a soft round flake: fully opaque
    // near the centre, fading smoothly to nothing at the rim.
    // Settled flakes use a much fuller disc — opaque out to 60%
    // of the radius instead of 20% — so neighbouring flakes on
    // the blanket overlap into continuous cover; falling flakes
    // keep the airier v1.2 profile so the snowfall stays light.
    // Either way the alpha reaches zero at the rim: soft round,
    // never square.
    float d = length(pointCoord - float2(0.5));
    float core = mix(0.10, 0.30, in.settledMix);
    float alpha = (1.0 - smoothstep(core, 0.5, d)) * in.alpha;
    float3 colour = float3(0.93, 0.96, 1.0) * in.shade;
    return float4(colour, alpha);
}

// The bare-floor look, shared by the surface mesh (wherever the
// blanket depth is ~0) and the apron (everywhere), so the two are
// indistinguishable where they meet: one-metre grid over a dark
// blue-grey base. The distance fade to the clear colour is
// applied by the caller, identically in both.
static float3 floorBase(float2 xz) {
    float3 base = float3(0.055, 0.075, 0.115);
    float2 grid = abs(fract(xz - 0.5) - 0.5) / fwidth(xz);
    float line = 1.0 - min(min(grid.x, grid.y), 1.0);
    return base + float3(0.10, 0.12, 0.15) * line * 0.6;
}

static float3 applyDistanceFade(float3 colour, float3 world) {
    float3 clearColour = float3(0.015, 0.025, 0.05);
    float fade = 1.0 - smoothstep(14.0, 36.0, length(world.xz));
    return mix(clearColour, colour, fade);
}

struct SurfaceOut {
    float4 position [[position]];
    float3 world;
    float3 normal;
    float height;
};

vertex SurfaceOut surfaceVertex(const device float *heightfield [[buffer(0)]],
                                constant FrameUniforms &f [[buffer(1)]],
                                uint vid [[vertex_id]]) {
    // Procedural grid at heightfield resolution: (GRID_N-1)^2
    // cells, two triangles per cell, vertices exactly at the
    // field's sample points — the same lattice sampleHeight
    // interpolates, so the mesh top IS the surface the flakes
    // land on. Each vertex fetches its own cell height plus its
    // neighbours' for the normal.
    uint cells = GRID_N - 1;
    uint cellIndex = vid / 6;
    uint corner = vid % 6;
    uint cx = cellIndex % cells;
    uint cz = cellIndex / cells;
    // Corner offsets in the old ground quad's pattern:
    // (0,0) (1,0) (1,1) / (0,0) (1,1) (0,1).
    uint ix = cx + ((corner == 1 || corner == 2 || corner == 4) ? 1 : 0);
    uint iz = cz + ((corner == 2 || corner == 4 || corner == 5) ? 1 : 0);
    float h = heightfield[iz * GRID_N + ix];
    float x = (float(ix) / float(cells) * 2.0 - 1.0) * f.halfExtent;
    float z = (float(iz) / float(cells) * 2.0 - 1.0) * f.halfExtent;
    uint ixL = ix > 0 ? ix - 1 : 0;
    uint ixR = min(ix + 1, cells);
    uint izD = iz > 0 ? iz - 1 : 0;
    uint izU = min(iz + 1, cells);
    float hL = heightfield[iz * GRID_N + ixL];
    float hR = heightfield[iz * GRID_N + ixR];
    float hD = heightfield[izD * GRID_N + ix];
    float hU = heightfield[izU * GRID_N + ix];
    float step = 2.0 * f.halfExtent / float(cells);
    float3 nrm = normalize(float3(-(hR - hL) / (2.0 * step), 1.0,
                                  -(hU - hD) / (2.0 * step)));
    SurfaceOut out;
    out.position = f.viewProjection * float4(x, h, z, 1.0);
    out.world = float3(x, h, z);
    out.normal = nrm;
    out.height = h;
    return out;
}

fragment float4 surfaceFragment(SurfaceOut in [[stage_in]]) {
    // The blanket's top surface. Where the depth is ~0 this is
    // the bare floor (identical shading to the apron), so fresh
    // ground and the post-Clear floor read as ground; the cover
    // ramp is the same 2 cm -> 10 cm the v1.3 tint used, now on
    // geometry at the right height instead of a painted plane.
    float cover = smoothstep(0.02, 0.10, in.height);
    // Strata: snow was laid down in layers over time, and layers
    // are horizontal — so tone steps by absolute altitude, in
    // ~5.5 cm bands with a darker seam at each band's base. On
    // the flat top only the newest band shows; on any sloped
    // face — a mound flank, the blanket's edge skirt, a walk
    // trail's walls — the surface crosses band after band and
    // the build-up reads as stacked layers.
    float bandCoord = in.world.y / 0.055;
    float bandTone = 0.94 + 0.11 * hash1(floor(bandCoord) * 3.913 + 0.17);
    float seam = mix(0.86, 1.0, smoothstep(0.0, 0.16, fract(bandCoord)));
    // Slope shading: a soft key light, and steep faces (trail
    // walls, skirt edges) a step cooler and darker so the cut
    // reads as a cut.
    float3 N = normalize(in.normal);
    float3 L = normalize(float3(0.35, 0.85, 0.30));
    float diffuse = 0.70 + 0.30 * clamp(dot(N, L), 0.0, 1.0);
    float wallShade = mix(1.0, 0.80, clamp((1.0 - N.y) * 1.3, 0.0, 1.0));
    // A faint world-space grain so the surface doesn't read as
    // flat paint (kept from the v1.3 tint).
    float grain = 0.96 + 0.08 * hash1(floor(in.world.x * 24.0) * 127.1
                                    + floor(in.world.z * 24.0) * 311.7);
    float3 snow = float3(0.78, 0.84, 0.96)
                * bandTone * seam * diffuse * wallShade * grain;
    float3 colour = mix(floorBase(in.world.xz), snow, cover);
    return float4(applyDistanceFade(colour, in.world), 1.0);
}

struct ApronOut {
    float4 position [[position]];
    float3 world;
};

vertex ApronOut apronVertex(constant FrameUniforms &f [[buffer(1)]],
                            uint vid [[vertex_id]]) {
    // Four flat quads framing the surface mesh, from its edge at
    // +/-halfExtent out to +/-40 m, generated from the vertex id
    // (24 vertices, no vertex buffer). The mesh's own edge sits
    // at height 0 (the field thins to zero at its boundary), so
    // apron and mesh meet flush at y = 0.
    float E = f.halfExtent;
    float4 rects[4]; // (x0, z0, x1, z1)
    rects[0] = float4(-40.0, -40.0, 40.0, -E);
    rects[1] = float4(-40.0, E, 40.0, 40.0);
    rects[2] = float4(-40.0, -E, -E, E);
    rects[3] = float4(E, -E, 40.0, E);
    uint quad = vid / 6;
    uint corner = vid % 6;
    float4 r = rects[quad];
    float x = (corner == 1 || corner == 2 || corner == 4) ? r.z : r.x;
    float z = (corner == 2 || corner == 4 || corner == 5) ? r.w : r.y;
    float3 wp = float3(x, 0.0, z);
    ApronOut out;
    out.position = f.viewProjection * float4(wp, 1.0);
    out.world = wp;
    return out;
}

fragment float4 apronFragment(ApronOut in [[stage_in]]) {
    return float4(applyDistanceFade(floorBase(in.world.xz), in.world), 1.0);
}
