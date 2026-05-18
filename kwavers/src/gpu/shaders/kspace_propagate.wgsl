// K-space spectral propagation kernel
//
// ## Algorithm
//
// Applies the exact free-space dispersion relation in k-space:
//
//   Ŝ(k, t+Δt) = Ŝ(k, t) · exp(−iωΔt)
//
// where ω = c₀|k| (linear dispersion, non-dispersive medium).
// This is exact for the homogeneous linear acoustic wave equation.
//
// ## Binding layout
//
// group(0) binding(0): `spectrum_re` — real parts of complex spectrum  (f32, read_write)
// group(0) binding(1): `spectrum_im` — imaginary parts of spectrum     (f32, read_write)
// group(0) binding(2): `kspace`      — packed [kx, ky, kz] per voxel  (f32, read)
//                       element i occupies kspace[3i], kspace[3i+1], kspace[3i+2]
// push_constant:       `params`      — GridParams (nx, ny, nz, dt, c0)
//
// ## Stride contract (kspace)
//
// `kspace` is declared as `array<f32>` (element stride = 4 bytes, packed).
// This matches the Rust upload code which packs [kx, ky, kz] as three
// consecutive f32 values per voxel (12 bytes/voxel).  The WGSL std430 rule
// for `array<vec3<f32>>` would impose a 16-byte stride, creating a mismatch.
//
// ## Notes
//
// FFT/IFFT are performed by `fft.wgsl` (separate shader, separate dispatch).
// This kernel is dispatched ONCE between the forward FFT and the inverse FFT.
//
// ## References
//
// - Liu QH (1998). "The PSTD algorithm: a time-domain method requiring only two
//   cells per wavelength." Microwave Opt Technol Lett 15(3):158–165.
// - Treeby BE, Cox BT (2010). J Biomed Opt 15(2):021314.

const TWO_PI: f32 = 6.28318530717958647692;

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
    c0: f32,
}

// Real parts of the complex pressure spectrum (split layout).
@group(0) @binding(0)
var<storage, read_write> spectrum_re: array<f32>;

// Imaginary parts of the complex pressure spectrum (split layout).
@group(0) @binding(1)
var<storage, read_write> spectrum_im: array<f32>;

// k-vector per voxel: packed [kx₀, ky₀, kz₀, kx₁, ky₁, kz₁, …] (f32, rad/m).
// Element stride = 4 bytes (f32 array, NOT vec3<f32> which has 16-byte stride).
@group(0) @binding(2)
var<storage, read> kspace: array<f32>;

var<push_constant> params: GridParams;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

/// Apply free-space propagation phase exp(−iω·Δt) to each spectral component.
///
/// ## Theorem (exact propagation for homogeneous media)
///
/// For the linear acoustic wave equation `∂²p/∂t² = c₀²∇²p` in a homogeneous
/// medium, the Fourier-mode solution is `p̂(k,t) = A·exp(±i|k|c₀t)`.  Advancing
/// by Δt multiplies each mode by the propagation factor `exp(−iω Δt)` where
/// ω = c₀|k|.  The in-place multiplication:
///
///   [Re' + i·Im'] = [Re + i·Im] · [cos(−ωΔt) + i·sin(−ωΔt)]
///   Re' = Re·cos(θ) − Im·sin(θ)
///   Im' = Re·sin(θ) + Im·cos(θ)     where θ = −ω·Δt
///
/// is exact (no time-stepping error) for a single non-dispersive mode.
///
/// ## Theorem (race-freedom)
///
/// Each voxel `vox = index_3d(x, y, z)` writes exclusively to
/// `spectrum_re[vox]` and `spectrum_im[vox]`.  The kspace reads are at
/// `kspace[3*vox], kspace[3*vox+1], kspace[3*vox+2]` — disjoint across
/// all threads.  No two threads share a memory address → race-free.
@compute @workgroup_size(8, 8, 8)
fn propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }

    let vox = index_3d(x, y, z);

    // Read k-vector from packed array<f32> (12 bytes/voxel, no padding).
    let kx = kspace[3u * vox];
    let ky = kspace[3u * vox + 1u];
    let kz = kspace[3u * vox + 2u];

    // |k|² = kx² + ky² + kz²
    let k2    = kx * kx + ky * ky + kz * kz;
    // ω = c₀·|k|; phase θ = −ω·Δt (forward propagation)
    let omega = params.c0 * sqrt(k2);
    let theta = -omega * params.dt;

    let cos_t = cos(theta);
    let sin_t = sin(theta);

    // Split complex layout: re and im in separate arrays.
    let re_in = spectrum_re[vox];
    let im_in = spectrum_im[vox];

    spectrum_re[vox] = re_in * cos_t - im_in * sin_t;
    spectrum_im[vox] = re_in * sin_t + im_in * cos_t;
}
