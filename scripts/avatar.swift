// Portrait -> circular-ready avatar.
//
//   1. lift the subject off its background (Vision foreground-instance mask)
//   2. locate the face so the square crop is centred on it, not on the image
//   3. composite onto a flat neutral backdrop and emit a square PNG
//
// The output is square rather than pre-rounded: CSS does the circle, so the
// same asset stays usable at any size and the mask can change without
// regenerating anything.
//
// usage: avatar <input> <output> <size> <hex-backdrop>

import AppKit
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import Vision

func fail(_ m: String) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!); exit(1)
}

let a = CommandLine.arguments
guard a.count == 5 else { fail("usage: avatar <in> <out> <size> <hex>") }
let inURL = URL(fileURLWithPath: a[1]), outURL = URL(fileURLWithPath: a[2])
guard let size = Double(a[3]) else { fail("bad size") }

func colour(_ hex: String) -> CIColor {
    var s = hex.hasPrefix("#") ? String(hex.dropFirst()) : hex
    if s.count == 3 { s = s.map { "\($0)\($0)" }.joined() }
    guard let v = UInt32(s, radix: 16) else { fail("bad hex \(hex)") }
    return CIColor(red: CGFloat((v >> 16) & 255) / 255.0,
                   green: CGFloat((v >> 8) & 255) / 255.0,
                   blue: CGFloat(v & 255) / 255.0)
}
let backdrop = colour(a[4])

guard let src = CIImage(contentsOf: inURL) else { fail("cannot read \(a[1])") }
let handler = VNImageRequestHandler(ciImage: src, options: [:])

// ── 1. subject cutout ────────────────────────────────────────────────────
let maskReq = VNGenerateForegroundInstanceMaskRequest()
try? handler.perform([maskReq])
var subject = src
let measureCtx = CIContext()

/// Mean coverage of one instance's mask, used to rank instances by area.
func coverage(_ obs: VNInstanceMaskObservation, _ idx: Int) -> Float {
    guard let buf = try? obs.generateScaledMaskForImage(forInstances: IndexSet(integer: idx),
                                                        from: handler) else { return 0 }
    let mask = CIImage(cvPixelBuffer: buf)
    let avg = CIFilter.areaAverage()
    avg.inputImage = mask
    avg.extent = mask.extent
    guard let out = avg.outputImage else { return 0 }
    var px = [UInt8](repeating: 0, count: 4)
    measureCtx.render(out, toBitmap: &px, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                      format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    return Float(px[0]) / 255.0
}

if let obs = maskReq.results?.first, !obs.allInstances.isEmpty {
    // Keep only the LARGEST instance. Vision happily segments a bystander's
    // backpack or a passing cyclist as extra foreground instances; compositing
    // all of them drags those into the avatar. The portrait subject is
    // reliably the biggest thing in a head-and-shoulders shot.
    let best = obs.allInstances.max { coverage(obs, $0) < coverage(obs, $1) } ?? obs.allInstances.first!
    if let buf = try? obs.generateMaskedImage(ofInstances: IndexSet(integer: best),
                                              from: handler,
                                              croppedToInstancesExtent: false) {
        subject = CIImage(cvPixelBuffer: buf)
    }
    if obs.allInstances.count > 1 {
        FileHandle.standardError.write(
            "note: \(obs.allInstances.count) instances, kept the largest\n".data(using: .utf8)!)
    }
} else {
    FileHandle.standardError.write("warn: no subject mask, using full frame\n".data(using: .utf8)!)
}

// ── 2. face-centred square crop ──────────────────────────────────────────
// Vision reports normalised coords with origin bottom-left, matching CIImage.
let faceReq = VNDetectFaceRectanglesRequest()
try? handler.perform([faceReq])

let e = src.extent
var cx = e.midX
var cy = e.midY
var side = min(e.width, e.height)

if let face = (faceReq.results?.max { $0.boundingBox.width < $1.boundingBox.width }) {
    let b = face.boundingBox
    let fw = b.width * e.width, fh = b.height * e.height
    cx = e.minX + b.midX * e.width
    cy = e.minY + b.midY * e.height
    // Head plus a little shoulder. Kept fairly tight (1.85x the face box)
    // for two reasons: these render at 26px in the ticker, where a loose
    // crop turns the face into a smudge, and a tighter frame excludes the
    // background clutter Vision sometimes welds onto the subject instance.
    side = min(max(fw, fh) * 1.85, min(e.width, e.height))
    // bias downward so the crop keeps chin and collar rather than empty
    // space above the hair
    cy -= fh * 0.14
}

// keep the square inside the image
let half = side / 2
cx = min(max(cx, e.minX + half), e.maxX - half)
cy = min(max(cy, e.minY + half), e.maxY - half)
let crop = CGRect(x: cx - half, y: cy - half, width: side, height: side)

let cropped = subject.cropped(to: crop)
    .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))

// ── 3. flatten onto the backdrop, scale to size ──────────────────────────
let bg = CIImage(color: backdrop).cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
let composite = cropped.composited(over: bg)
let scale = size / side
let final = composite.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    .cropped(to: CGRect(x: 0, y: 0, width: size, height: size))

let ctx = CIContext()
guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { fail("no colour space") }
do {
    try ctx.writePNGRepresentation(of: final, to: outURL, format: .RGBA8, colorSpace: cs)
} catch { fail("write failed: \(error.localizedDescription)") }

print("ok \(outURL.lastPathComponent)  \(Int(size))px")
