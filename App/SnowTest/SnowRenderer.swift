// SnowRenderer.swift — all Metal state for SnowTest.
//
// The simulation is 100% GPU: one MTLBuffer holds every flake, the
// compute kernel in Shaders.metal updates it in place each frame,
// and the render pass reads the same buffer. The CPU only fills the
// buffer once (at start and when the flake count changes), passes
// small uniforms, and issues one compute dispatch + one point draw
// per frame. Particle data is never copied back.

import Metal
import MetalKit
import QuartzCore
import simd

/// Byte offsets of each field inside one particle, matching the
/// `SnowParticle` layout in Shaders.metal (packed_float3 = 12
/// bytes). The CPU writes initial particles at these offsets; after
/// that only the GPU touches the buffer.
enum SnowParticleLayout {
    static let stride = 32
    static let positionX = 0
    static let positionY = 4
    static let positionZ = 8
    static let velocityX = 12
    static let velocityY = 16
    static let velocityZ = 20
    static let randomOffset = 24
    static let size = 28
}

/// Mirror of `SimUniforms` in Shaders.metal — plain floats only, so
/// the layout is identical on both sides (24 bytes).
struct SimUniforms {
    var dt: Float = 0
    var time: Float = 0
    var windStrength: Float = 0
    var fallSpeed: Float = 1
    var halfExtent: Float = 14
    var height: Float = 13
}

/// Mirror of `FrameUniforms` in Shaders.metal.
struct FrameUniforms {
    var viewProjection = matrix_identity_float4x4
    var pointScale: Float = 1
}

/// Small deterministic RNG for seeding the initial particle field
/// (reproducible runs, no SystemRandom dependency).
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
    @Published var flakeCount: Double = 30_000
    @Published var wind: Double = 1.5
    @Published var fallSpeed: Double = 1.0

    // Live stats shown in the overlay, refreshed twice a second.
    @Published var fps: Double = 0
    @Published var gpuMs: Double = 0
    @Published var liveCount: Int = 0

    // Orbit camera. Plain vars: gestures write them, the frame loop
    // reads them every frame, no UI binding needed.
    var cameraYaw: Float = 0.7
    var cameraPitch: Float = 0.38
    var cameraDistance: Float = 16

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let computePipeline: MTLComputePipelineState
    private let snowPipeline: MTLRenderPipelineState
    private let groundPipeline: MTLRenderPipelineState

    private var particleBuffer: MTLBuffer?
    private var particleCount = 0

    // Falling volume: +/- 14 m in x/z, respawn at 13 m, floor y = 0.
    private let volumeHalfExtent: Float = 14
    private let volumeHeight: Float = 13
    private let cameraTarget = SIMD3<Float>(0, 3.2, 0)
    private let fovY: Float = .pi / 3

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

            let snowDescriptor = MTLRenderPipelineDescriptor()
            snowDescriptor.vertexFunction = library.makeFunction(name: "snowVertex")
            snowDescriptor.fragmentFunction = library.makeFunction(name: "snowFragment")
            snowDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            // Soft flakes blend over the ground and each other.
            let snowAttachment = snowDescriptor.colorAttachments[0]
            snowAttachment.isBlendingEnabled = true
            snowAttachment.rgbBlendOperation = .add
            snowAttachment.alphaBlendOperation = .add
            snowAttachment.sourceRGBBlendFactor = .sourceAlpha
            snowAttachment.sourceAlphaBlendFactor = .one
            snowAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
            snowAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            snowPipeline = try device.makeRenderPipelineState(descriptor: snowDescriptor)

            let groundDescriptor = MTLRenderPipelineDescriptor()
            groundDescriptor.vertexFunction = library.makeFunction(name: "groundVertex")
            groundDescriptor.fragmentFunction = library.makeFunction(name: "groundFragment")
            groundDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            groundPipeline = try device.makeRenderPipelineState(descriptor: groundDescriptor)
        } catch {
            fatalError("Metal pipeline setup failed: \(error)")
        }
        super.init()
    }

    /// (Re)build the particle buffer with fresh random flakes. Runs
    /// only when the requested count changes — never per frame.
    private func rebuildParticles(count: Int) {
        particleCount = count
        guard count > 0 else {
            particleBuffer = nil
            return
        }
        let stride = SnowParticleLayout.stride
        guard let buffer = device.makeBuffer(length: stride * count,
                                             options: .storageModeShared) else {
            particleBuffer = nil
            return
        }
        var rng = SplitMix64(state: 0x5EED_0000_0000_0001)
        let base = buffer.contents()
        for i in 0..<count {
            let p = base.advanced(by: i * stride)
            let x = (rng.nextFloat() * 2 - 1) * volumeHalfExtent
            let y = rng.nextFloat() * volumeHeight
            let z = (rng.nextFloat() * 2 - 1) * volumeHalfExtent
            let fall = 0.6 + rng.nextFloat() * 1.0        // 0.6...1.6 m/s
            let diameter = 0.02 + rng.nextFloat() * 0.045 // 2...6.5 cm
            p.storeBytes(of: x, toByteOffset: SnowParticleLayout.positionX, as: Float.self)
            p.storeBytes(of: y, toByteOffset: SnowParticleLayout.positionY, as: Float.self)
            p.storeBytes(of: z, toByteOffset: SnowParticleLayout.positionZ, as: Float.self)
            p.storeBytes(of: Float(0), toByteOffset: SnowParticleLayout.velocityX, as: Float.self)
            p.storeBytes(of: fall, toByteOffset: SnowParticleLayout.velocityY, as: Float.self)
            p.storeBytes(of: Float(0), toByteOffset: SnowParticleLayout.velocityZ, as: Float.self)
            p.storeBytes(of: rng.nextFloat() * 100, toByteOffset: SnowParticleLayout.randomOffset, as: Float.self)
            p.storeBytes(of: diameter, toByteOffset: SnowParticleLayout.size, as: Float.self)
        }
        particleBuffer = buffer
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
        if wanted != particleCount {
            rebuildParticles(count: wanted)
        }
        guard particleCount > 0,
              let buffer = particleBuffer,
              let passDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let commandBuffer = queue.makeCommandBuffer() else {
            return
        }

        // --- Pass 1 (compute): advance the whole simulation in
        // place in the shared particle buffer. ---
        var sim = SimUniforms(dt: dt,
                              time: time,
                              windStrength: Float(wind),
                              fallSpeed: Float(fallSpeed),
                              halfExtent: volumeHalfExtent,
                              height: volumeHeight)
        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.setComputePipelineState(computePipeline)
            compute.setBuffer(buffer, offset: 0, index: 0)
            compute.setBytes(&sim, length: MemoryLayout<SimUniforms>.stride, index: 1)
            compute.dispatchThreads(MTLSize(width: particleCount, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            compute.endEncoding()
        }

        // --- Camera + frame uniforms for the render passes. ---
        let drawableSize = view.drawableSize
        let aspect = Float(drawableSize.width / max(drawableSize.height, 1))
        let projection = SnowMath.perspective(fovY: fovY, aspect: aspect, near: 0.1, far: 200)
        let eye = cameraTarget + SIMD3<Float>(
            cos(cameraPitch) * sin(cameraYaw),
            sin(cameraPitch),
            cos(cameraPitch) * cos(cameraYaw)) * cameraDistance
        let viewMatrix = SnowMath.lookAt(eye: eye, target: cameraTarget, up: SIMD3<Float>(0, 1, 0))
        var frame = FrameUniforms()
        frame.viewProjection = projection * viewMatrix
        // Pixels per world unit at unit depth: drawableHeight/2 * proj[1][1].
        frame.pointScale = Float(drawableSize.height) * 0.5 / tan(fovY * 0.5)

        // --- Pass 2 (render): ground, then every flake in ONE
        // point draw call reading the same buffer the compute pass
        // just updated. ---
        if let render = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) {
            render.setRenderPipelineState(groundPipeline)
            render.setVertexBytes(&frame, length: MemoryLayout<FrameUniforms>.stride, index: 1)
            render.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            render.setRenderPipelineState(snowPipeline)
            render.setVertexBuffer(buffer, offset: 0, index: 0)
            render.setVertexBytes(&frame, length: MemoryLayout<FrameUniforms>.stride, index: 1)
            render.drawPrimitives(type: .point, vertexStart: 0, vertexCount: particleCount)
            render.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { [weak self] completed in
            guard let self = self else { return }
            let ms = (completed.GPUEndTime - completed.GPUStartTime) * 1000
            self.statsLock.lock()
            self.gpuTimeTotal += ms
            self.gpuTimeSamples += 1
            self.statsLock.unlock()
        }
        commandBuffer.commit()

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
