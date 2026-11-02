// ContentView.swift — SnowTest UI: the Metal view full-screen,
// a live stats panel top-left, and three sliders along the bottom.
// Kept deliberately minimal: this app exists to judge the snowfall
// and its frame rate at a glance. High contrast, uncluttered.

import SwiftUI
import MetalKit

struct ContentView: View {
    @StateObject private var snow = SnowRenderer()
    @State private var lastDragTranslation: CGSize = .zero
    @State private var lastMagnification: CGFloat = 1

    var body: some View {
        SnowView(renderer: snow)
            .ignoresSafeArea()
            .gesture(orbitGesture)
            .simultaneousGesture(zoomGesture)
            // Overlays only intercept touches where they have
            // content, so orbiting works everywhere else.
            .overlay(alignment: .topLeading) {
                statsPanel.padding(14)
            }
            .overlay(alignment: .bottom) {
                controlsPanel.padding(14)
            }
    }

    private var statsPanel: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("SNOWTEST")
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.65))
            Text("\(Int(snow.fps.rounded())) FPS")
                .font(.system(size: 26, weight: .bold, design: .monospaced))
                .foregroundStyle(.white)
            Text("\(snow.liveCount.formatted(.number)) flakes")
                .font(.system(size: 15, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.9))
            Text(String(format: "GPU %.1f ms", snow.gpuMs))
                .font(.system(size: 15, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.9))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
    }

    private var controlsPanel: some View {
        VStack(spacing: 10) {
            sliderRow(title: "Flakes",
                      value: snow.liveCount.formatted(.number),
                      binding: $snow.flakeCount,
                      range: 1_000...100_000,
                      step: 1_000)
            sliderRow(title: "Wind",
                      value: String(format: "%.1f m/s", snow.wind),
                      binding: $snow.wind,
                      range: 0...8,
                      step: 0.1)
            sliderRow(title: "Fall speed",
                      value: String(format: "%.2fx", snow.fallSpeed),
                      binding: $snow.fallSpeed,
                      range: 0.2...2.5,
                      step: 0.05)
            Text("Drag to orbit · pinch to zoom")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.white.opacity(0.6))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14))
    }

    private func sliderRow(title: String,
                           value: String,
                           binding: Binding<Double>,
                           range: ClosedRange<Double>,
                           step: Double) -> some View {
        HStack(spacing: 12) {
            Text(title)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 86, alignment: .leading)
            Slider(value: binding, in: range, step: step)
                .tint(.white)
            Text(value)
                .font(.system(size: 15, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.9))
                .frame(width: 92, alignment: .trailing)
        }
    }

    private var orbitGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let dx = Float(value.translation.width - lastDragTranslation.width)
                let dy = Float(value.translation.height - lastDragTranslation.height)
                lastDragTranslation = value.translation
                snow.cameraYaw -= dx * 0.005
                snow.cameraPitch = min(max(snow.cameraPitch + dy * 0.005, 0.06), 1.35)
            }
            .onEnded { _ in lastDragTranslation = .zero }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let ratio = value.magnification / lastMagnification
                lastMagnification = value.magnification
                let next = snow.cameraDistance / Float(ratio)
                snow.cameraDistance = min(max(next, 4), 45)
            }
            .onEnded { _ in lastMagnification = 1 }
    }
}

/// The Metal view, configured once: no depth buffer (nothing needs
/// one — the ground is drawn first and flakes blend over it), dark
/// clear colour matching the ground shader's fade, up to 120 Hz
/// where the panel supports it.
struct SnowView: UIViewRepresentable {
    let renderer: SnowRenderer

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = renderer.device
        view.delegate = renderer
        view.preferredFramesPerSecond = 120
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .invalid
        view.clearColor = MTLClearColor(red: 0.015, green: 0.025, blue: 0.05, alpha: 1)
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}
