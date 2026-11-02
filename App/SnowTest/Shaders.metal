// Shaders.metal — SnowTest
//
// GPU snow, following the compute-particle design this test app
// exists to prove out:
//
//   1. `updateSnow` (compute) advances every flake IN PLACE in one
//      shared MTLBuffer — gravity fall, sine-wave wind drift, and
//      floor collision by respawn. No particle data ever comes
//      back to the CPU.
//   2. `snowVertex` / `snowFragment` (render) read the SAME buffer
//      and draw every flake as a single point primitive, masked in
//      the fragment shader into a soft round flake — precipitation
//      must never read as squares.
//   3. `groundVertex` / `groundFragment` draw the one ground plane
//      procedurally (no vertex buffer) so the floor reads clearly.
//
// SnowParticle layout is shared byte-for-byte with SnowRenderer.swift:
//   position      bytes  0..<12  (packed_float3)
//   velocity      bytes 12..<24  (packed_float3; .y = fall speed)
//   randomOffset  bytes 24..<28  (float)
//   size          bytes 28..<32  (float; world-space flake diameter)
// Total stride: 32 bytes.

#include <metal_stdlib>
using namespace metal;

struct SnowParticle {
    packed_float3 position;
    packed_float3 velocity;
    float randomOffset;
    float size;
};

// Everything the simulation needs for one step. Plain floats only,
// so the Swift mirror struct has an identical layout.
struct SimUniforms {
    float dt;           // seconds since last frame (clamped on CPU)
    float time;         // seconds since the app started
    float windStrength; // sideways push, metres per second
    float fallSpeed;    // multiplier on each flake's fall velocity
    float halfExtent;   // falling volume spans +/- halfExtent in x/z
    float height;       // respawn height; the floor is y = 0
};

// Everything the render passes need for one frame.
struct FrameUniforms {
    float4x4 viewProjection;
    float pointScale;   // pixels per world unit at clip.w = 1
};

// Cheap deterministic hash -> [0, 1). Used so a recycled flake gets a
// fresh, time-varying respawn spot instead of the same place forever.
static float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

kernel void updateSnow(device SnowParticle *particles [[buffer(0)]],
                       constant SimUniforms &u [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    SnowParticle p = particles[id];

    // 1. Fall at the flake's own terminal velocity, scaled by the
    //    fall-speed control.
    p.position.y -= p.velocity.y * u.fallSpeed * u.dt;

    // 2. Wind: a steady +x push (this is what visibly bends the
    //    fall) plus a gentle per-flake sway from sine waves keyed
    //    to the flake's random offset.
    float swayX = sin(u.time * 0.9 + p.randomOffset * 17.0);
    float swayZ = cos(u.time * 0.7 + p.randomOffset * 29.0);
    p.position.x += (u.windStrength + swayX * 0.35) * u.dt;
    p.position.z += swayZ * 0.35 * u.dt;

    // 3. Wrap horizontally so wind and sway never empty the volume:
    //    a flake leaving one side re-enters on the other.
    float span = 2.0 * u.halfExtent;
    float wrappedX = fmod(p.position.x + u.halfExtent, span);
    if (wrappedX < 0.0) wrappedX += span;
    p.position.x = wrappedX - u.halfExtent;
    float wrappedZ = fmod(p.position.z + u.halfExtent, span);
    if (wrappedZ < 0.0) wrappedZ += span;
    p.position.z = wrappedZ - u.halfExtent;

    // 4. Floor collision: recycle to the top of the volume at a
    //    fresh random horizontal spot (the hash mixes in the time,
    //    so the spot differs every cycle).
    if (p.position.y < 0.0) {
        float r1 = hash1(p.randomOffset * 91.7 + u.time * 0.618);
        float r2 = hash1(p.randomOffset * 47.3 + u.time * 1.313);
        float r3 = hash1(p.randomOffset * 13.1 + u.time * 0.917);
        p.position.x = (r1 * 2.0 - 1.0) * u.halfExtent;
        p.position.z = (r2 * 2.0 - 1.0) * u.halfExtent;
        p.position.y = u.height + r3 * 1.5;
    }

    particles[id] = p;
}

struct FlakeOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float shade;
};

vertex FlakeOut snowVertex(const device SnowParticle *particles [[buffer(0)]],
                           constant FrameUniforms &f [[buffer(1)]],
                           uint id [[vertex_id]]) {
    SnowParticle p = particles[id];
    float4 clip = f.viewProjection * float4(p.position, 1.0);
    FlakeOut out;
    out.position = clip;
    // Perspective point size: world diameter scaled by the frame's
    // point scale and divided by depth (clip.w = -viewZ). Clamped
    // so a flake next to the camera cannot balloon and distant
    // flakes never vanish below one pixel.
    float depth = max(clip.w, 0.1);
    out.pointSize = clamp(p.size * f.pointScale / depth, 1.0, 64.0);
    // Slight per-flake brightness variation so the field does not
    // read as one flat texture.
    out.shade = 0.78 + 0.22 * fract(p.randomOffset * 7.31);
    return out;
}

fragment float4 snowFragment(FlakeOut in [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
    // Mask the square point into a soft round flake: fully opaque
    // near the centre, fading smoothly to nothing at the rim.
    float d = length(pointCoord - float2(0.5));
    float alpha = (1.0 - smoothstep(0.1, 0.5, d)) * 0.9;
    float3 colour = float3(0.93, 0.96, 1.0) * in.shade;
    return float4(colour, alpha);
}

struct GroundOut {
    float4 position [[position]];
    float3 world;
};

vertex GroundOut groundVertex(constant FrameUniforms &f [[buffer(1)]],
                              uint id [[vertex_id]]) {
    // Two triangles covering [-40, 40]^2 at y = 0, generated from
    // the vertex id — no vertex buffer needed.
    const float2 corners[6] = {
        float2(-40.0, -40.0), float2(40.0, -40.0), float2(40.0, 40.0),
        float2(-40.0, -40.0), float2(40.0, 40.0), float2(-40.0, 40.0)
    };
    float2 c = corners[id];
    float3 wp = float3(c.x, 0.0, c.y);
    GroundOut out;
    out.position = f.viewProjection * float4(wp, 1.0);
    out.world = wp;
    return out;
}

fragment float4 groundFragment(GroundOut in [[stage_in]]) {
    // One-metre antialiased grid over a dark blue-grey base, fading
    // to the clear colour with distance so the plane reads as a
    // floor under the snowfall rather than a hard-edged slab.
    float3 base = float3(0.055, 0.075, 0.115);
    float3 clearColour = float3(0.015, 0.025, 0.05);
    float2 grid = abs(fract(in.world.xz - 0.5) - 0.5) / fwidth(in.world.xz);
    float line = 1.0 - min(min(grid.x, grid.y), 1.0);
    float3 colour = base + float3(0.10, 0.12, 0.15) * line * 0.6;
    float fade = 1.0 - smoothstep(14.0, 36.0, length(in.world.xz));
    return float4(mix(clearColour, colour, fade), 1.0);
}
