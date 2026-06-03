// K-space staggered-grid phase shift compute shader
//
// Theorem: Staggered-grid half-cell shift operator (Treeby & Cox 2010, Eq. 12)
//   For a velocity field ux stored at i+½ on a staggered grid, the spectral
//   derivative ∂ux/∂x evaluated at cell-centres i requires a phase correction:
//       FFT(ux)[kx] ← FFT(ux)[kx] · exp(−i·kx·sx)
//   where sx = ±Δx/2 is the half-cell shift distance (negative for forward
//   stagger, positive for inverse stagger).
//
//   For a 3-component shift vector (sx, sy, sz) and 3D spectrum F(kx,ky,kz):
//       F(kx,ky,kz) ← F(kx,ky,kz) · exp(−i·(kx·sx + ky·sy + kz·sz))
//
//   Complex multiplication:
//       (Re + i·Im) · exp(i·φ) = Re·cos(φ) − Im·sin(φ)
//                                + i·(Re·sin(φ) + Im·cos(φ))
//
// Algorithm
// ---------
// 1. Map global invocation ID → (ix, jy, kz)
// 2. Look up kx[ix], ky[jy], kz[kz] from per-axis wavenumber arrays
// 3. Compute phase φ = −(kx·sx + ky·sy + kz·sz)
// 4. Apply complex rotation in-place
//
// References
// ----------
// - Treeby BE, Cox BT (2010). J. Biomed. Opt. 15(2):021314.
//   (Eq. 12-13: k-space staggered shift for velocity/density updates)
// - Liu Q-H (1998). Geophysics 63(6):2082–2089.
//   (pseudospectral time-domain method with staggered grids)

struct ShiftParams {
    nx: u32,
    ny: u32,
    nz: u32,
    sx: f32,   // x-axis shift distance [m], e.g. ±Δx/2
    sy: f32,   // y-axis shift distance [m], e.g. ±Δy/2
    sz: f32,   // z-axis shift distance [m], e.g. ±Δz/2
}

// 3D complex spectrum (row-major: real and imaginary parts interleaved as separate arrays)
@group(0) @binding(0)
var<storage, read_write> spec_real: array<f32>;

@group(0) @binding(1)
var<storage, read_write> spec_imag: array<f32>;

// Per-axis wavenumber vectors (length nx, ny, nz respectively)
@group(0) @binding(2)
var<storage, read> kx_vec: array<f32>;   // length nx

@group(0) @binding(3)
var<storage, read> ky_vec: array<f32>;   // length ny

@group(0) @binding(4)
var<storage, read> kz_vec: array<f32>;   // length nz

@group(1) @binding(0)
var<uniform> params: ShiftParams;

/// Row-major 3D index
fn idx3(ix: u32, jy: u32, kz: u32) -> u32 {
    return ix + jy * params.nx + kz * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 4)
fn kspace_shift(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.x;
    let jy = id.y;
    let kz = id.z;

    if (ix >= params.nx || jy >= params.ny || kz >= params.nz) {
        return;
    }

    let n = idx3(ix, jy, kz);

    // Phase: φ = −(kx·sx + ky·sy + kz·sz)
    let phase = -(kx_vec[ix] * params.sx
               +  ky_vec[jy] * params.sy
               +  kz_vec[kz] * params.sz);

    let c = cos(phase);
    let s = sin(phase);

    let re = spec_real[n];
    let im = spec_imag[n];

    // Complex rotation: F · exp(i·φ)
    spec_real[n] = re * c - im * s;
    spec_imag[n] = re * s + im * c;
}
