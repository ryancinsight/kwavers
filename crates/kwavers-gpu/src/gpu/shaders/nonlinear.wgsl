// Westervelt nonlinear acoustic propagation kernel
//
// ## Wave Equation
//
// Modified Westervelt equation (Westervelt 1963; Hamilton & Blackstock 1998 §3.5):
//
//   ∇²p − (1/c₀²)∂²p/∂t² + (β/ρ₀c₀⁴)∂²p²/∂t² = 0
//
// Rearranged to isolate the acceleration:
//
//   ∂²p/∂t² = c₀²∇²p + (β/ρ₀c₀²) · ∂²p²/∂t²
//
// where β = 1 + B/(2A) is the coefficient of nonlinearity.
//
// ## Weak-Nonlinear Approximation
//
// In the weak-shock limit, `∂²p²/∂t² ≈ 2(∂p/∂t)²` (quadratic term dominates),
// so the nonlinear acceleration becomes:
//
//   a_nl = (β/ρ₀c₀²) · 2(∂p/∂t)²  = (2β/ρ₀c₀²) · (∂p/∂t)²
//
// The backward-difference approximation `∂p/∂t ≈ (pⁿ − pⁿ⁻¹)/Δt` gives:
//
//   a_nl ≈ (2β/ρ₀c₀²) · ((pⁿ − pⁿ⁻¹)/Δt)²
//
// Reference: Aanonsen et al. (1984) JASA 75(3):749–768.
//
// ## Leapfrog Time Integration
//
//   p^{n+1} = 2·pⁿ − pⁿ⁻¹ + Δt² · (c₀²·∇²pⁿ + a_nl)
//
// Note: thermoviscous absorption MUST be applied as a separate dispatch
// (multiplicative decay `p *= exp(−α·c₀·Δt)`) AFTER this kernel completes.
// The explicit ∂³p/∂t³ discretization is unconditionally unstable (Von Neumann:
// growing-mode factor (r−1)³ > 0 for r > 1); operator-splitting decay is the
// only stable approximation (Pinton et al. 2009 §IIB).
//
// ## Binding layout
//
// group(0) binding(0): params       — NonlinearParams uniform
// group(0) binding(1): grid         — GridParams uniform
// group(0) binding(2): pressure     — pⁿ   (current step), read-only
// group(0) binding(3): pressure_prev— pⁿ⁻¹ (previous step), read-only
// group(0) binding(4): pressure_next— pⁿ⁺¹ (output), write-only (read_write)
// group(0) binding(5): density      — ρ₀ per voxel [kg/m³], read-only
// group(0) binding(6): sound_speed  — c₀ per voxel [m/s], read-only
// group(0) binding(7): nonlinearity — β per voxel (= 1 + B/(2A)), read-only
//
// ## References
//
// - Westervelt PJ (1963). JASA 35(4):535–537.
// - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.
// - Aanonsen SI et al. (1984). JASA 75(3):749–768.
// - Pinton GF et al. (2009). IEEE UFFC 56(3):474–488. (stable absorption)

struct NonlinearParams {
    dt: f32,  // Δt [s]
}

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
}

@group(0) @binding(0)
var<uniform> params: NonlinearParams;

@group(0) @binding(1)
var<uniform> grid: GridParams;

// pⁿ (current), pⁿ⁻¹ (previous): separate read-only buffers — no write hazard.
@group(0) @binding(2)
var<storage, read> pressure: array<f32>;

@group(0) @binding(3)
var<storage, read> pressure_prev: array<f32>;

// pⁿ⁺¹ (output): separate write buffer — no read-write alias with input buffers.
@group(0) @binding(4)
var<storage, read_write> pressure_next: array<f32>;

@group(0) @binding(5)
var<storage, read> density: array<f32>;

@group(0) @binding(6)
var<storage, read> sound_speed: array<f32>;

// β = 1 + B/(2A), coefficient of nonlinearity per voxel.
@group(0) @binding(7)
var<storage, read> nonlinearity: array<f32>;

fn get_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * grid.nx + z * grid.nx * grid.ny;
}

/// Second-order isotropic Laplacian ∇²p at interior point (x,y,z).
///
/// Uses 6-point central-difference stencil:
///   ∇²p ≈ (p_{i-1}−2p_i+p_{i+1})/dx² + (similarly for y, z)
///
/// Returns 0 at boundary nodes (Dirichlet: p=0 on halo).
fn compute_laplacian(x: u32, y: u32, z: u32) -> f32 {
    if (x == 0u || x >= grid.nx - 1u ||
        y == 0u || y >= grid.ny - 1u ||
        z == 0u || z >= grid.nz - 1u) {
        return 0.0;
    }

    let idx = get_index(x, y, z);
    let p   = pressure[idx];

    let d2x = (pressure[get_index(x + 1u, y, z)] - 2.0 * p + pressure[get_index(x - 1u, y, z)])
              / (grid.dx * grid.dx);
    let d2y = (pressure[get_index(x, y + 1u, z)] - 2.0 * p + pressure[get_index(x, y - 1u, z)])
              / (grid.dy * grid.dy);
    let d2z = (pressure[get_index(x, y, z + 1u)] - 2.0 * p + pressure[get_index(x, y, z - 1u)])
              / (grid.dz * grid.dz);

    return d2x + d2y + d2z;
}

/// Nonlinear Westervelt update: leapfrog with weak-nonlinear correction.
///
/// ## Algorithm
///
/// 1. Compute ∇²pⁿ (6-point central differences).
/// 2. Compute backward-difference velocity: v ≈ (pⁿ−pⁿ⁻¹)/Δt.
/// 3. Nonlinear acceleration: a_nl = (2β/ρ₀c₀²)·v².
///    Derivation: ∂²p²/∂t² ≈ 2(∂p/∂t)² in weak-nonlinear limit;
///    Westervelt coefficient for acceleration is β/ρ₀c₀²; combined: 2β/ρ₀c₀².
/// 4. Leapfrog: pⁿ⁺¹ = 2pⁿ − pⁿ⁻¹ + Δt²·(c₀²·∇²pⁿ + a_nl).
///
/// Absorption is NOT applied here; dispatch a multiplicative decay kernel afterward.
@compute @workgroup_size(8, 8, 4)
fn nonlinear_propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= grid.nx || y >= grid.ny || z >= grid.nz) {
        return;
    }

    let idx = get_index(x, y, z);

    // Boundary: Dirichlet zero
    if (x == 0u || x == grid.nx - 1u ||
        y == 0u || y == grid.ny - 1u ||
        z == 0u || z == grid.nz - 1u) {
        pressure_next[idx] = 0.0;
        return;
    }

    let rho  = density[idx];
    let c    = sound_speed[idx];
    let beta = nonlinearity[idx];  // β = 1 + B/(2A)

    // c₀²·∇²pⁿ  (linear wave acceleration)
    let lin = c * c * compute_laplacian(x, y, z);

    // Backward-difference ∂p/∂t ≈ (pⁿ − pⁿ⁻¹)/Δt
    let p      = pressure[idx];
    let p_prev = pressure_prev[idx];
    let dp_dt  = (p - p_prev) / params.dt;

    // Nonlinear acceleration: (2β/ρ₀c₀²)·(∂p/∂t)²
    // Factor 2 from ∂²(p²)/∂t² ≈ 2(∂p/∂t)² in weak-nonlinear limit.
    // Reference: Aanonsen et al. (1984) JASA 75(3):749–768.
    let nl = (2.0 * beta / (rho * c * c)) * dp_dt * dp_dt;

    // Leapfrog: p^{n+1} = 2pⁿ − pⁿ⁻¹ + Δt²·(lin + nl)
    pressure_next[idx] = 2.0 * p - p_prev + params.dt * params.dt * (lin + nl);
}
