// SnowRenderer.swift — all Metal state for SnowTest.
//
// The simulation is 100% GPU: one MTLBuffer holds every flake, the
// compute kernel in Shaders.metal updates it in place each frame,
// and the render pass reads the same buffer. A second pair of
// buffers holds the blanket heightfield (ping-ponged), advanced by
// its own compute pass: growth, compaction, slope relaxation,
// wind drift, depression refill, and the walk body's footprint
// press. Since v1.4 a three-pass GPU culling stage compacts the
// visible flakes into an indirect draw, and the blanket renders
// as a heightfield-displaced mesh with depth testing. The CPU
// only fills buffers (at start, on count change, and on Clear),
// passes small uniforms, and issues the passes. Particle and
// heightfield data are never copied back.

import Metal
import MetalKit
import QuartzCore
import simd

/// Byte offsets of each field inside one particle, matching the
/// `SnowParticle` layout in Shaders.metal (packed_float3 = 12
/// bytes). The CPU writes initial particles at these offsets; after
/// that only the GPU touches the buffer.
enum SnowParticleLayout {
    static let stride = 36
    static let positionX = 0
    static let positionY = 4
    static let positionZ = 8
    static let velocityX = 12
    static let velocityY = 16
    static let velocityZ = 20
    static let randomOffset = 24
    static let size = 28
    static let state = 32
}

/// Mirror of `SimUniforms` in Shaders.metal — plain floats only, so
/// the layout is identical on both sides (44 bytes).
struct SimUniforms {
    var dt: Float = 0
    var time: Float = 0
    var windStrength: Float = 0
    var fallSpeed: Float = 1
    var halfExtent: Float = 14
    var height: Float = 13
    var capHeight: Float = 0.8
    var bodyX: Float = 0
    var bodyZ: Float = 0
    var bodyRadius: Float = 0.35
    var bodyActive: Float = 0
}

/// Mirror of `HFUniforms` in Shaders.metal — plain floats only
/// (48 bytes).
struct HFUniforms {
    var dt: Float = 0
    var growthRate: Float = 0
    var capHeight: Float = 0.8
    var windStrength: Float = 0
    var bodyX: Float = 0
    var bodyZ: Float = 0
    var bodyRadius: Float = 0.35
    var bodyActive: Float = 0
    var bodyMoving: Float = 0
    var pressDepth: Float = 0.16
    var halfExtent: Float = 14
    var time: Float = 0
}

/// Mirror of `FrameUniforms` in Shaders.metal.
struct FrameUniforms {
    var viewProjection = matrix_identity_float4x4
    var pointScale: Float = 1
    var halfExtent: Float = 14
    var sizeScale: Float = 1
}

/// Mirror of `CullUniforms` in Shaders.metal (v1.4) — matrix plus
/// plain scalars, identical layout on both sides (80 bytes).
struct CullUniforms {
    var viewProjection = matrix_identity_float4x4
    var halfExtent: Float = 14
    var sizeScale: Float = 1
    var flakeCount: UInt32 = 0
    var groupCount: UInt32 = 0
}

/// Small deterministic RNG for seeding the initial particle field
/// and the heightfield noise (reproducible runs, no
/// SystemRandom dependency).
struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    /// Uniform float in [0, 1) from the top 24 bits.
    mutating func nextFloat() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}

enum SnowMath {
    /// Metal clip-space perspective projection (z in [0, w],
    /// w = -viewZ). fovY in radians.
    static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4<Float>(xs, 0, 0, 0)
        m.columns.1 = SIMD4<Float>(0, ys, 0, 0)
        m.columns.2 = SIMD4<Float>(0, 0, zs, -1)
        m.columns.3 = SIMD4<Float>(0, 0, zs * near, 0)
        return m
    }

    /// Right-handed view matrix looking from `eye` toward `target`.
    static func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let z = normalize(eye - target)   // backward
        let x = normalize(cross(up, z))   // right
        let y = cross(z, x)               // true up
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4<Float>(x.x, y.x, z.x, 0)
        m.columns.1 = SIMD4<Float>(x.y, y.y, z.y, 0)
        m.columns.2 = SIMD4<Float>(x.z, y.z, z.z, 0)
        m.columns.3 = SIMD4<Float>(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
        return m
    }
}

final class SnowRenderer: NSObject, ObservableObject, MTKViewDelegate {
    // Controls bound by the SwiftUI overlay. This object is the
    // single source of truth for them; the frame loop reads them.
    @Published var flakeCount: Double = 100_000
    @Published var wind: Double = 1.5
    @Published var fallSpeed: Double = 1.0
    @Published var flakeSize: Double = 1.0
    @Published var walkMode: Bool = false

    // Live stats shown in the overlay, refreshed twice a second.
    @Published var fps: Double = 0
    @Published var gpuMs: Double = 0
    @Published var liveCount: Int = 0

    // Orbit camera. Plain vars: gestures write them, the frame loop
    // reads them every frame, no UI binding needed.
    var cameraYaw: Float = 0.7
    var cameraPitch: Float = 0.38
    var cameraDistance: Float = 16

    // Walk-mode body. Positioned from the orbit camera the first
    // time walk mode is entered, then driven by drag gestures
    // (see applyWalkDrag). The same position is passed to the
    // compute passes as the displacement body.
    private var bodyX: Float = 0
    private var bodyZ: Float = 10
    private var bodyYaw: Float = 0.7
    private var bodyPrevX: Float = 0
    private var bodyPrevZ: Float = 10
    private var wasWalkMode = false
    private let bodyRadius: Float = 0.35

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let computePipeline: MTLComputePipelineState
    private let heightfieldPipeline: MTLComputePipelineState
    private let cullCountPipeline: MTLComputePipelineState
    private let cullScanPipeline: MTLComputePipelineState
    private let cullWritePipeline: MTLComputePipelineState
    private let snowPipeline: MTLRenderPipelineState
    private let surfacePipeline: MTLRenderPipelineState
    private let apronPipeline: MTLRenderPipelineState
    // Surface and apron are opaque and write depth; flakes test
    // against it (so the blanket's geometry occludes the flakes
    // inside and behind it) but never write.
    private let depthWriteState: MTLDepthStencilState
    private let depthTestState: MTLDepthStencilState

    private var particleBuffer: MTLBuffer?
    private var particleCount = 0
    // Visibility-culling state (v1.4), sized to match the live
    // particle buffer and rebuilt with it in rebuildParticles:
    // the visible-index draw list, the per-flake rank scratch,
    // per-threadgroup visible totals and their prefix offsets,
    // and the indirect-draw arguments the scan pass fills in.
    // The CPU never reads any of it back.
    private var visibleIndexBuffer: MTLBuffer?
    private var rankScratchBuffer: MTLBuffer?
    private var groupTotalsBuffer: MTLBuffer?
    private var groupOffsetsBuffer: MTLBuffer?
    private var cullArgsBuffer: MTLBuffer?
    private var cullGroupCount = 0
    // Count-change debounce: at v1.2/v1.3 counts (up to 5M) a
    // rebuild is a real allocation + seed, so while the Flakes
    // slider is being dragged the rebuild waits until the
    // requested count holds still briefly instead of thrashing
    // one rebuild per step. The blanket is never touched either
    // way.
    private var pendingCount = 0
    private var pendingSince: CFTimeInterval = 0
    // A count whose buffer allocation failed (memory pressure at
    // the 5M ceiling): rebuildParticles keeps the old buffer in
    // that case, and draw(in:) must not retry the same failing
    // allocation every debounce window — only a different
    // requested count clears the block.
    private var failedCount = 0

    // Falling volume: +/- 14 m in x/z, respawn at 13 m, floor y = 0.
    private let volumeHalfExtent: Float = 14
    private let volumeHeight: Float = 13
    private let cameraTarget = SIMD3<Float>(0, 3.2, 0)
    private let fovY: Float = .pi / 3

    // Blanket (heightfield) state. Two float buffers ping-pong:
    // each frame the heightfield pass reads one and writes the
    // other, then the particle pass and the vertex shader read the
    // freshly written one. Must match GRID_N in Shaders.metal.
    // 256x256 in v1.2 (was 128): ~0.11 m cells, so accumulation
    // relief survives at hand scale and stacks can form.
    static let heightfieldGrid = 256
    private let blanketCap: Float = 0.8
    private let pressDepth: Float = 0.16
    private var heightBuffers: [MTLBuffer] = []
    private var noiseBuffer: MTLBuffer?
    private var hfReadIndex = 0
    // The CPU's own scalar estimate of the average blanket depth,
    // integrated from the same growth/compaction rates the GPU
    // uses (never read back from the GPU). It only positions the
    // walk-mode eye height, so an approximation is fine.
    private var avgBlanket: Float = 0

    private var startTime = CACurrentMediaTime()
    private var lastDrawTime: CFTimeInterval?
    private var statFrames = 0
    private var statWindowStart: CFTimeInterval = 0
    private let statsLock = NSLock()
    private var gpuTimeTotal: Double = 0
    private var gpuTimeSamples = 0

    override init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("SnowTest requires a Metal-capable device")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create a Metal command queue")
        }
        self.queue = queue
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not load the default Metal library")
        }
        do {
            guard let updateFunction = library.makeFunction(name: "updateSnow") else {
                fatalError("Missing compute function updateSnow")
            }
            computePipeline = try device.makeComputePipelineState(function: updateFunction)
            guard let hfFunction = library.makeFunction(name: "updateHeightfield") else {
                fatalError("Missing compute function updateHeightfield")
            }
            heightfieldPipeline = try device.makeComputePipelineState(function: hfFunction)
            guard let cullCountFunction = library.makeFunction(name: "cullFlakesCount"),
                  let cullScanFunction = library.makeFunction(name: "cullFlakesScan"),
                  let cullWriteFunction = library.makeFunction(name: "cullFlakesWrite") else {
                fatalError("Missing culling compute functions")
            }
            cullCountPipeline = try device.makeComputePipelineState(function: cullCountFunction)
            cullScanPipeline = try device.makeComputePipelineState(function: cullScanFunction)
            cullWritePipeline = try device.makeComputePipelineState(function: cullWriteFunction)

            let snowDescriptor = MTLRenderPipelineDescriptor()
            snowDescriptor.vertexFunction = library.makeFunction(name: "snowVertex")
            snowDescriptor.fragmentFunction = library.makeFunction(name: "snowFragment")
            snowDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            snowDescriptor.depthAttachmentPixelFormat = .depth32Float
            // Soft flakes blend over the ground and each other.
            if let snowAttachment = snowDescriptor.colorAttachments[0] {
                snowAttachment.isBlendingEnabled = true
                snowAttachment.rgbBlendOperation = .add
                snowAttachment.alphaBlendOperation = .add
                snowAttachment.sourceRGBBlendFactor = .sourceAlpha
                snowAttachment.sourceAlphaBlendFactor = .one
                snowAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
                snowAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            snowPipeline = try device.makeRenderPipelineState(descriptor: snowDescriptor)

            let surfaceDescriptor = MTLRenderPipelineDescriptor()
            surfaceDescriptor.vertexFunction = library.makeFunction(name: "surfaceVertex")
            surfaceDescriptor.fragmentFunction = library.makeFunction(name: "surfaceFragment")
            surfaceDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            surfaceDescriptor.depthAttachmentPixelFormat = .depth32Float
            surfacePipeline = try device.makeRenderPipelineState(descriptor: surfaceDescriptor)

            let apronDescriptor = MTLRenderPipelineDescriptor()
            apronDescriptor.vertexFunction = library.makeFunction(name: "apronVertex")
            apronDescriptor.fragmentFunction = library.makeFunction(name: "apronFragment")
            apronDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            apronDescriptor.depthAttachmentPixelFormat = .depth32Float
            apronPipeline = try device.makeRenderPipelineState(descriptor: apronDescriptor)

            let writeDescriptor = MTLDepthStencilDescriptor()
            writeDescriptor.depthCompareFunction = .less
            writeDescriptor.isDepthWriteEnabled = true
            guard let writeState = device.makeDepthStencilState(descriptor: writeDescriptor) else {
                fatalError("Could not create depth-write state")
            }
            depthWriteState = writeState
            let testDescriptor = MTLDepthStencilDescriptor()
            testDescriptor.depthCompareFunction = .less
            testDescriptor.isDepthWriteEnabled = false
            guard let testState = device.makeDepthStencilState(descriptor: testDescriptor) else {
                fatalError("Could not create depth-test state")
            }
            depthTestState = testState
        } catch {
            fatalError("Metal pipeline setup failed: \(error)")
        }
        // Indirect-draw arguments for the culled flake draw: 16
        // bytes matching MTLDrawPrimitivesIndirectArguments,
        // written each frame by cullFlakesScan on the GPU.
        cullArgsBuffer = device.makeBuffer(length: 16, options: .storageModeShared)
        super.init()
        buildHeightfield()
    }

    /// Create the two zeroed heightfield buffers and the static
    /// per-cell growth-noise buffer (smooth value noise in
    /// [0.7, 1.3], so the blanket does not rise as one flat sheet).
    private func buildHeightfield() {
        let n = SnowRenderer.heightfieldGrid
        let byteCount = n * n * MemoryLayout<Float>.stride
        heightBuffers = (0..<2).compactMap { _ in
            device.makeBuffer(length: byteCount, options: .storageModeShared)
        }
        for b in heightBuffers {
            memset(b.contents(), 0, b.length)
        }
        // Coarse 64x64 random grid, bilinearly upsampled to n x n.
        // (32x32 in v1.1; at the finer v1.2 grid the growth noise
        // gets a finer feature scale too, ~0.44 m, so fresh snow
        // rises in soft mounds rather than one flat sheet.)
        let coarse = 64
        var rng = SplitMix64(state: 0x4E4F_4953_4500_0001)
        var grid = [Float](repeating: 0, count: coarse * coarse)
        for i in 0..<(coarse * coarse) { grid[i] = rng.nextFloat() }
        guard let noise = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            noiseBuffer = nil
            return
        }
        let base = noise.contents()
        for iz in 0..<n {
            for ix in 0..<n {
                let fx = Float(ix) / Float(n - 1) * Float(coarse - 1)
                let fz = Float(iz) / Float(n - 1) * Float(coarse - 1)
                let x0 = min(Int(fx), coarse - 2)
                let z0 = min(Int(fz), coarse - 2)
                let tx = fx - Float(x0)
                let tz = fz - Float(z0)
                let v00 = grid[z0 * coarse + x0]
                let v10 = grid[z0 * coarse + x0 + 1]
                let v01 = grid[(z0 + 1) * coarse + x0]
                let v11 = grid[(z0 + 1) * coarse + x0 + 1]
                let v = (v00 * (1 - tx) + v10 * tx) * (1 - tz)
                      + (v01 * (1 - tx) + v11 * tx) * tz
                base.storeBytes(of: Float(0.7) + v * Float(0.6),
                                toByteOffset: (iz * n + ix) * MemoryLayout<Float>.stride,
                                as: Float.self)
            }
        }
        noiseBuffer = noise
    }

    /// Blanket growth rate in m/s for the current flake count:
    /// a visible coating inside a minute, a real layer over
    /// several minutes at the default count. The reference count
    /// is the v1.2 default (100k): v1.1 paced 0.0016 m/s at its
    /// 30k default, and the rate is rescaled so the pacing at
    /// the default is unchanged even though the default moved.
    private func growthRate(count: Int) -> Float {
        0.0016 * Float(count) / 100_000
    }

    /// Restart the accumulation from bare ground: zero the blanket,
    /// reseed the flakes, reset the CPU blanket estimate. The GPU
    /// state is rewritten, never read.
    func clearSnow() {
        for b in heightBuffers {
            memset(b.contents(), 0, b.length)
        }
        avgBlanket = 0
        rebuildParticles(count: Int(flakeCount.rounded()))
        pendingCount = particleCount
    }

    /// Walk-mode drag: horizontal movement turns the body,
    /// vertical movement walks it forward (drag up) or backward
    /// (drag down) along its facing. Called per gesture increment.
    func applyWalkDrag(dx: Float, dy: Float) {
        bodyYaw -= dx * 0.005
        let move = -dy * 0.035
        bodyX += -sin(bodyYaw) * move
        bodyZ += -cos(bodyYaw) * move
        let limit = volumeHalfExtent - 0.8
        bodyX = min(max(bodyX, -limit), limit)
        bodyZ = min(max(bodyZ, -limit), limit)
    }

    /// (Re)build the particle buffer with fresh random flakes. Runs
    /// only when the requested count changes (debounced, see
    /// draw(in:)) and on Clear — never per frame. The blanket is
    /// untouched, so changing the count mid-run does not disturb
    /// the accumulation. If the allocation itself fails — the
    /// real risk at the v1.3 ceiling, where the buffer is
    /// 36 bytes x 5M = ~180 MB — the previous buffer and count
    /// stay live (the snow on screen is untouched) and the failed
    /// count is remembered so the frame loop doesn't retry it.
    private func rebuildParticles(count: Int) {
        guard count > 0 else {
            particleCount = 0
            particleBuffer = nil
            return
        }
        let stride = SnowParticleLayout.stride
        guard let buffer = device.makeBuffer(length: stride * count,
                                             options: .storageModeShared) else {
            failedCount = count
            return
        }
        // Seed through a typed float pointer: at the v1.3 ceiling
        // (5M flakes, 45M floats) the per-field storeBytes loop of
        // v1.1 would stall the frame loop for seconds. The typed
        // loop is O(count) with no per-flake bridging; even so,
        // first-touching ~180 MB of fresh buffer is the one place
        // the app may hitch briefly when the slider first settles
        // at the top end — a one-time cost per count change, never
        // per frame. Field order matches SnowParticleLayout byte
        // offsets / 4. The buffer is not yet submitted to the GPU,
        // so binding it here is safe.
        let floatsPerFlake = stride / MemoryLayout<Float>.stride
        let floats = buffer.contents().bindMemory(to: Float.self,
                                                  capacity: count * floatsPerFlake)
        var rng = SplitMix64(state: 0x5EED_0000_0000_0001)
        for i in 0..<count {
            let o = i * floatsPerFlake
            let fall = 0.6 + rng.nextFloat() * 1.0        // 0.6...1.6 m/s
            // v1.2: flakes are roughly half v1.1's size so the
            // snow reads fine-grained — 1.0...3.6 cm diameter
            // (was 2...6.5 cm).
            let diameter = 0.010 + rng.nextFloat() * 0.026
            floats[o + 0] = (rng.nextFloat() * 2 - 1) * volumeHalfExtent
            floats[o + 1] = rng.nextFloat() * volumeHeight
            floats[o + 2] = (rng.nextFloat() * 2 - 1) * volumeHalfExtent
            floats[o + 3] = 0
            // Velocity is the flake's actual signed velocity: it
            // starts out falling at its terminal speed.
            floats[o + 4] = -fall
            floats[o + 5] = 0
            floats[o + 6] = rng.nextFloat() * 100
            floats[o + 7] = diameter
            floats[o + 8] = 0
        }
        // The v1.4 culling buffers are sized to this flake count
        // and must exist for the draw, so they are allocated as
        // part of the same rebuild: if any of them fails, the
        // whole rebuild counts as failed and the previous buffer
        // set stays live (same rule as the particle allocation
        // above). At the 5M ceiling they add ~40 MB over the
        // ~180 MB particle buffer (two 20 MB uint arrays plus
        // small per-group arrays).
        let groups = (count + 255) / 256
        guard let indices = device.makeBuffer(length: count * 4,
                                              options: .storageModeShared),
              let scratch = device.makeBuffer(length: count * 4,
                                              options: .storageModeShared),
              let totals = device.makeBuffer(length: groups * 4,
                                             options: .storageModeShared),
              let offsets = device.makeBuffer(length: groups * 4,
                                              options: .storageModeShared) else {
            failedCount = count
            return
        }
        visibleIndexBuffer = indices
        rankScratchBuffer = scratch
        groupTotalsBuffer = totals
        groupOffsetsBuffer = offsets
        cullGroupCount = groups
        particleCount = count
        particleBuffer = buffer
        failedCount = 0
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Nothing cached by size — draw(in:) reads the live size.
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let rawDt = lastDrawTime.map { now - $0 } ?? (1.0 / 60.0)
        lastDrawTime = now
        // Clamp the step so a hitch (or first frame) cannot fling
        // flakes across the volume.
        let dt = Float(min(max(rawDt, 0), 0.05))
        let time = Float(now - startTime)

        let wanted = Int(flakeCount.rounded())
        if wanted != particleCount && wanted != failedCount {
            if particleCount == 0 {
                rebuildParticles(count: wanted) // first build: no wait
                pendingCount = particleCount
            } else if wanted != pendingCount {
                pendingCount = wanted
                pendingSince = now
            } else if now - pendingSince > 0.15 {
                rebuildParticles(count: wanted)
                pendingCount = particleCount
            }
        }
        guard particleCount > 0,
              let buffer = particleBuffer,
              heightBuffers.count == 2,
              let noise = noiseBuffer,
              let passDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let commandBuffer = queue.makeCommandBuffer() else {
            return
        }

        // Entering walk mode: put the body where the orbit camera
        // is, facing the way it faces, so the view doesn't jump.
        if walkMode != wasWalkMode {
            if walkMode {
                let eye = cameraTarget + SIMD3<Float>(
                    cos(cameraPitch) * sin(cameraYaw),
                    sin(cameraPitch),
                    cos(cameraPitch) * cos(cameraYaw)) * cameraDistance
                let limit = volumeHalfExtent - 0.8
                bodyX = min(max(eye.x, -limit), limit)
                bodyZ = min(max(eye.z, -limit), limit)
                bodyYaw = cameraYaw
                bodyPrevX = bodyX
                bodyPrevZ = bodyZ
            }
            wasWalkMode = walkMode
        }

        // Body movement this frame (for the footprint rim) and the
        // CPU-side average-blanket estimate (walk eye height only).
        let bodyMoved = hypot(bodyX - bodyPrevX, bodyZ - bodyPrevZ)
        let bodyMoving: Float = (walkMode && bodyMoved > 0.003) ? 1 : 0
        bodyPrevX = bodyX
        bodyPrevZ = bodyZ
        let growth = growthRate(count: particleCount)
        let compactAvg = 0.0016 + 0.0022 * (avgBlanket / blanketCap)
        var blanketDelta = Double(growth) - Double(avgBlanket) * Double(compactAvg)
        if wind > 4 {
            blanketDelta -= Double(avgBlanket) * (wind - 4) * 0.004
        }
        avgBlanket = Float(min(max(Double(avgBlanket) + blanketDelta * rawDt, 0),
                               Double(blanketCap)))

        // --- Camera + frame uniforms, computed first: the
        // culling passes below need the same view-projection
        // matrix the render passes will use. ---
        let drawableSize = view.drawableSize
        let aspect = Float(drawableSize.width / max(drawableSize.height, 1))
        let projection = SnowMath.perspective(fovY: fovY, aspect: aspect, near: 0.1, far: 200)
        let viewMatrix: simd_float4x4
        if walkMode {
            // Eye just above the blanket at the body's position,
            // looking along its facing, pitched very slightly down.
            let eye = SIMD3<Float>(bodyX, avgBlanket + 1.5, bodyZ)
            let forward = SIMD3<Float>(-sin(bodyYaw), -0.055, -cos(bodyYaw))
            viewMatrix = SnowMath.lookAt(eye: eye, target: eye + forward,
                                         up: SIMD3<Float>(0, 1, 0))
        } else {
            let eye = cameraTarget + SIMD3<Float>(
                cos(cameraPitch) * sin(cameraYaw),
                sin(cameraPitch),
                cos(cameraPitch) * cos(cameraYaw)) * cameraDistance
            viewMatrix = SnowMath.lookAt(eye: eye, target: cameraTarget,
                                         up: SIMD3<Float>(0, 1, 0))
        }
        var frame = FrameUniforms()
        frame.viewProjection = projection * viewMatrix
        // Pixels per world unit at unit depth: drawableHeight/2 * proj[1][1].
        frame.pointScale = Float(drawableSize.height) * 0.5 / tan(fovY * 0.5)
        frame.halfExtent = volumeHalfExtent
        frame.sizeScale = Float(flakeSize)

        var cull = CullUniforms()
        cull.viewProjection = frame.viewProjection
        cull.halfExtent = volumeHalfExtent
        cull.sizeScale = Float(flakeSize)
        cull.flakeCount = UInt32(particleCount)
        cull.groupCount = UInt32(cullGroupCount)

        let hfSrc = heightBuffers[hfReadIndex]
        let hfDst = heightBuffers[1 - hfReadIndex]

        // --- Pass 1 (compute): advance the blanket heightfield. ---
        var hf = HFUniforms(dt: dt,
                            growthRate: growth,
                            capHeight: blanketCap,
                            windStrength: Float(wind),
                            bodyX: bodyX,
                            bodyZ: bodyZ,
                            bodyRadius: bodyRadius,
                            bodyActive: walkMode ? 1 : 0,
                            bodyMoving: bodyMoving,
                            pressDepth: pressDepth,
                            halfExtent: volumeHalfExtent,
                            time: time)
        if let hfEncoder = commandBuffer.makeComputeCommandEncoder() {
            hfEncoder.setComputePipelineState(heightfieldPipeline)
            hfEncoder.setBuffer(hfSrc, offset: 0, index: 0)
            hfEncoder.setBuffer(hfDst, offset: 0, index: 1)
            hfEncoder.setBuffer(noise, offset: 0, index: 2)
            hfEncoder.setBytes(&hf, length: MemoryLayout<HFUniforms>.stride, index: 3)
            let grid = SnowRenderer.heightfieldGrid
            hfEncoder.dispatchThreads(MTLSize(width: grid, height: grid, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            hfEncoder.endEncoding()
        }

        // --- Pass 2 (compute): advance every flake in place,
        // against the blanket the pass above just produced. ---
        var sim = SimUniforms(dt: dt,
                              time: time,
                              windStrength: Float(wind),
                              fallSpeed: Float(fallSpeed),
                              halfExtent: volumeHalfExtent,
                              height: volumeHeight,
                              capHeight: blanketCap,
                              bodyX: bodyX,
                              bodyZ: bodyZ,
                              bodyRadius: bodyRadius,
                              bodyActive: walkMode ? 1 : 0)
        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.setComputePipelineState(computePipeline)
            compute.setBuffer(buffer, offset: 0, index: 0)
            compute.setBytes(&sim, length: MemoryLayout<SimUniforms>.stride, index: 1)
            compute.setBuffer(hfDst, offset: 0, index: 2)
            compute.dispatchThreads(MTLSize(width: particleCount, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            compute.endEncoding()
        }

        // --- Pass 3 (compute, v1.4): visibility culling. Three
        // small dispatches compact the flakes that can actually
        // be seen (not buried past the fade range, inside the
        // frustum) into the visible-index list and write the
        // survivor count into the indirect-draw arguments. The
        // culling buffers always exist alongside the particle
        // buffer (allocated together in rebuildParticles). ---
        if let indices = visibleIndexBuffer,
           let scratch = rankScratchBuffer,
           let totals = groupTotalsBuffer,
           let offsets = groupOffsetsBuffer,
           let args = cullArgsBuffer {
            let groupSize = MTLSize(width: 256, height: 1, depth: 1)
            let groupGrid = MTLSize(width: cullGroupCount, height: 1, depth: 1)
            if let countEnc = commandBuffer.makeComputeCommandEncoder() {
                countEnc.setComputePipelineState(cullCountPipeline)
                countEnc.setBuffer(buffer, offset: 0, index: 0)
                countEnc.setBuffer(hfDst, offset: 0, index: 1)
                countEnc.setBytes(&cull, length: MemoryLayout<CullUniforms>.stride, index: 2)
                countEnc.setBuffer(scratch, offset: 0, index: 3)
                countEnc.setBuffer(totals, offset: 0, index: 4)
                countEnc.dispatchThreadgroups(groupGrid, threadsPerThreadgroup: groupSize)
                countEnc.endEncoding()
            }
            if let scanEnc = commandBuffer.makeComputeCommandEncoder() {
                scanEnc.setComputePipelineState(cullScanPipeline)
                scanEnc.setBuffer(totals, offset: 0, index: 0)
                scanEnc.setBuffer(offsets, offset: 0, index: 1)
                scanEnc.setBuffer(args, offset: 0, index: 2)
                scanEnc.setBytes(&cull, length: MemoryLayout<CullUniforms>.stride, index: 3)
                scanEnc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                scanEnc.endEncoding()
            }
            if let writeEnc = commandBuffer.makeComputeCommandEncoder() {
                writeEnc.setComputePipelineState(cullWritePipeline)
                writeEnc.setBuffer(scratch, offset: 0, index: 0)
                writeEnc.setBuffer(offsets, offset: 0, index: 1)
                writeEnc.setBuffer(indices, offset: 0, index: 2)
                writeEnc.setBytes(&cull, length: MemoryLayout<CullUniforms>.stride, index: 3)
                writeEnc.dispatchThreadgroups(groupGrid, threadsPerThreadgroup: groupSize)
                writeEnc.endEncoding()
            }

            // --- Pass 4 (render): the blanket surface and the
            // floor apron (opaque, depth-writing), then every
            // visible flake in ONE indirect point draw call
            // (depth-tested against the surface, never writing). ---
            if let render = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) {
                let surfaceVertexCount = (SnowRenderer.heightfieldGrid - 1)
                                       * (SnowRenderer.heightfieldGrid - 1) * 6
                render.setDepthStencilState(depthWriteState)
                render.setRenderPipelineState(surfacePipeline)
                render.setVertexBuffer(hfDst, offset: 0, index: 0)
                render.setVertexBytes(&frame, length: MemoryLayout<FrameUniforms>.stride, index: 1)
                render.drawPrimitives(type: .triangle, vertexStart: 0,
                                      vertexCount: surfaceVertexCount)

                render.setRenderPipelineState(apronPipeline)
                render.setVertexBytes(&frame, length: MemoryLayout<FrameUniforms>.stride, index: 1)
                render.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 24)

                render.setDepthStencilState(depthTestState)
                render.setRenderPipelineState(snowPipeline)
                render.setVertexBuffer(buffer, offset: 0, index: 0)
                render.setVertexBytes(&frame, length: MemoryLayout<FrameUniforms>.stride, index: 1)
                render.setVertexBuffer(hfDst, offset: 0, index: 2)
                render.setVertexBuffer(indices, offset: 0, index: 3)
                render.drawPrimitives(type: .point, indirectBuffer: args,
                                      indirectBufferOffset: 0)
                render.endEncoding()
            }
        }

        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { [weak self] completed in
            guard let self = self else { return }
            let ms = (completed.gpuEndTime - completed.gpuStartTime) * 1000
            self.statsLock.lock()
            self.gpuTimeTotal += ms
            self.gpuTimeSamples += 1
            self.statsLock.unlock()
        }
        commandBuffer.commit()
        hfReadIndex = 1 - hfReadIndex

        // --- Stats window: refresh the overlay twice a second. ---
        statFrames += 1
        if statWindowStart == 0 { statWindowStart = now }
        let elapsed = now - statWindowStart
        if elapsed >= 0.5 {
            fps = Double(statFrames) / elapsed
            statsLock.lock()
            if gpuTimeSamples > 0 {
                gpuMs = gpuTimeTotal / Double(gpuTimeSamples)
            }
            gpuTimeTotal = 0
            gpuTimeSamples = 0
            statsLock.unlock()
            liveCount = particleCount
            statFrames = 0
            statWindowStart = now
        }
    }
}
