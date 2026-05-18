// Absorption kernel for acoustic wave propagation
//
// ## Algorithm: Stokes-Kirchhoff multiplicative per-step decay
//
// Power-law spatial attenuation `α(f) = α₀·|f/f_ref|^y` [Np m⁻¹] is applied
// as the O(Δt) operator-splitting multiplicative decay (Pinton et al. 2009, §IIB):
//
//   p^{n+1} ← p^{n+1} · exp(−α(f)·c₀·Δt)
//
// where c₀ is the local sound speed (m/s), Δt the time step (s), and
// α(f) the Stokes–Kirchhoff attenuation coefficient at the driving frequency f.
//
// The operator-splitting error is O(α·c₀·Δt)² per step, which converges to
// the exact temporal amplitude decay in the limit α·c₀·Δt → 0.  This formulation
// is unconditionally stable for all α ≥ 0 (unlike the explicit ∂³p/∂t³ term).
//
// ## Binding layout
//
// group(0) binding(0): params          — AbsorptionParams uniform
// group(0) binding(1): pressure        — pressure field, read_write (in-place decay)
// group(0) binding(2): frequency_spectrum — dominant frequency per voxel (Hz), read-only
// group(0) binding(3): grid_size       — (nx, ny, nz), read-only
//
// ## References
//
// - Pinton GF et al. (2009). IEEE UFFC 56(3):474–488.
// - Szabo TL (1994). J Acoust Soc Am 96(1):491–500.  (power-law absorption)
// - Treeby BE, Cox BT (2010). J Biomed Opt 15(2):021314.  (fractional Laplacian)

struct AbsorptionParams {
    alpha_coeff:    f32,  // α₀  [Np m⁻¹ Hz⁻ʸ] at reference frequency
    alpha_power:    f32,  // y   frequency exponent (typical: 1.0–2.0)
    reference_freq: f32,  // f_ref [Hz] at which α₀ is specified
    dt:             f32,  // Δt  [s]  time step
    c0:             f32,  // c₀  [m/s] local sound speed (per-voxel if heterogeneous)
    _pad0:          f32,  // alignment padding (std140 requires 4-component alignment)
    _pad1:          f32,
    _pad2:          f32,
}

@group(0) @binding(0)
var<uniform> params: AbsorptionParams;

@group(0) @binding(1)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(2)
var<storage, read> frequency_spectrum: array<f32>;

@group(0) @binding(3)
var<storage, read> grid_size: vec3<u32>;

/// Power-law attenuation coefficient at frequency `freq`.
///
/// α(f) = α₀ · |f / f_ref|^y   [Np m⁻¹]
///
/// Reference: Szabo TL (1994) JASA 96(1):491–500.
fn compute_absorption(freq: f32) -> f32 {
    let normalized_freq = freq / params.reference_freq;
    return params.alpha_coeff * pow(abs(normalized_freq), params.alpha_power);
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= grid_size.x ||
        global_id.y >= grid_size.y ||
        global_id.z >= grid_size.z) {
        return;
    }

    let idx = global_id.x
            + global_id.y * grid_size.x
            + global_id.z * grid_size.x * grid_size.y;

    // Stokes-Kirchhoff multiplicative per-step decay:
    //   p *= exp(−α(f) · c₀ · Δt)
    // c₀ from params (per-dispatch scalar; heterogeneous media require per-voxel c
    // passed through a separate storage buffer and sampled here instead).
    let alpha       = compute_absorption(frequency_spectrum[idx]);
    let attenuation = exp(-alpha * params.c0 * params.dt);
    pressure[idx]   = pressure[idx] * attenuation;
}