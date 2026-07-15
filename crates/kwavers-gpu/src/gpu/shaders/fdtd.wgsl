// FDTD velocity-pressure acoustic solver (collocated, first-order split)
//
// ## Algorithm
//
// Linear acoustic equations (Yee 1966, scalar-pressure form):
//
//   ∂v/∂t = −(1/ρ₀) ∇p          (momentum)
//   ∂p/∂t = −ρ₀c₀² ∇·v           (continuity + equation of state)
//
// Collocated explicit Euler time integration:
//   vⁿ⁺¹ = vⁿ − (Δt/ρ₀) ∇pⁿ     (velocity_update)
//   pⁿ⁺¹ = pⁿ − Δt·ρ₀c₀² ∇·vⁿ⁺¹ (pressure_update, reads updated velocity)
//
// Two separate entry points must be dispatched in sequence:
//   1. velocity_update  — reads pressure (read-only), writes velocity_next
//   2. pressure_update  — reads velocity_next (read-only), writes pressure_next
//
// The separation guarantees that each field is fully written before the other
// dispatch reads it, eliminating cross-workgroup memory ordering hazards.
//
// ## Limitations
//
// - First-order accuracy in time (collocated Euler, not staggered Yee).
// - CFL stability requires c₀·Δt/Δx ≤ 1/√3 for 3D.
// - No PML boundary conditions; Dirichlet zero on boundary cells.
// - No absorption or nonlinear terms; apply separate kernels afterward.
// - For second-order accuracy, use `fdtd_pressure.wgsl` (leapfrog, 3 buffers)
//   or `pstd.wgsl` (pseudospectral, PML, absorption, nonlinear).
//
// ## Binding layout
//
// group(0) binding(0): pressure     — pⁿ,   read-only (velocity_update) / pⁿ output (pressure_update)
// group(0) binding(1): velocity     — vⁿ output (velocity_update) / vⁿ⁺¹ input (pressure_update)
// group(0) binding(2): medium       — [ρ₀, c₀] per voxel, read-only
// push_constant:       params       — GridParams (nx, ny, nz, dt)
//
// ## References
//
// - Yee KS (1966). IEEE Trans Antennas Propag 14(3):302–307.
// - Virieux J (1986). Geophysics 51(4):889–901.

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
}

// pressure[idx] = scalar pressure pⁿ [Pa]
@group(0) @binding(0)
var<storage, read_write> pressure: array<f32>;

// velocity[idx] = particle velocity vⁿ [m/s], 3-component packed as vec3<f32>
@group(0) @binding(1)
var<storage, read_write> velocity: array<vec3<f32>>;

// medium[idx] = (ρ₀ [kg/m³], c₀ [m/s])
@group(0) @binding(2)
var<storage, read> medium: array<vec2<f32>>;

var<immediate> params: GridParams;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

/// Velocity update: vⁿ⁺¹[i] = vⁿ[i] − (Δt/ρ₀) ∇pⁿ
///
/// Central-difference gradient of pressure (non-staggered):
///   ∂p/∂x ≈ (p[i+1] − p[i−1]) / (2Δx)   (O(Δx²))
///
/// Dispatch BEFORE pressure_update. After this dispatch, velocity holds vⁿ⁺¹.
@compute @workgroup_size(8, 8, 4)
fn velocity_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }

    // Boundary: skip (Dirichlet v = 0 at boundary — do not update)
    if (x == 0u || x == params.nx - 1u ||
        y == 0u || y == params.ny - 1u ||
        z == 0u || z == params.nz - 1u) {
        return;
    }

    let idx     = index_3d(x, y, z);
    let density = medium[idx].x;

    // Central-difference pressure gradient (O(Δx²)):
    //   ∂p/∂x at (i,j,k) ≈ (p[i+1,j,k] − p[i−1,j,k]) / (2Δx)
    // Note: grid spacing absorbed into dt; Δx = c₀·dt/CFL — caller must scale dt.
    // For a uniform grid with equal spacing h in all directions:
    //   grad_p = (p[x+1] − p[x−1], p[y+1] − p[y−1], p[z+1] − p[z−1]) * 0.5/h
    // The factor 0.5/h is missing here — callers must either pass pre-scaled dt
    // or include the grid spacing in params. This shader uses pressure differences
    // directly and relies on params.dt absorbing the 1/h factor.
    let px_pos = pressure[index_3d(x + 1u, y, z)];
    let px_neg = pressure[index_3d(x - 1u, y, z)];
    let py_pos = pressure[index_3d(x, y + 1u, z)];
    let py_neg = pressure[index_3d(x, y - 1u, z)];
    let pz_pos = pressure[index_3d(x, y, z + 1u)];
    let pz_neg = pressure[index_3d(x, y, z - 1u)];

    let grad_p = vec3<f32>(
        (px_pos - px_neg) * 0.5,
        (py_pos - py_neg) * 0.5,
        (pz_pos - pz_neg) * 0.5
    );

    // v^{n+1} = v^n - (dt/rho) * grad_p
    velocity[idx] -= (params.dt / density) * grad_p;
}

/// Pressure update: pⁿ⁺¹[i] = pⁿ[i] − Δt·ρ₀c₀²·(∇·vⁿ⁺¹)
///
/// Central-difference divergence of velocity (non-staggered):
///   ∂vₓ/∂x ≈ (vₓ[i+1] − vₓ[i−1]) / (2Δx)   (O(Δx²))
///
/// Dispatch AFTER velocity_update. Reads the updated velocity from binding(1).
@compute @workgroup_size(8, 8, 4)
fn pressure_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }

    // Boundary: Dirichlet zero
    if (x == 0u || x == params.nx - 1u ||
        y == 0u || y == params.ny - 1u ||
        z == 0u || z == params.nz - 1u) {
        pressure[index_3d(x, y, z)] = 0.0;
        return;
    }

    let idx     = index_3d(x, y, z);
    let density = medium[idx].x;
    let c       = medium[idx].y;
    let c2      = c * c;

    // Central-difference velocity divergence:
    let vx_pos = velocity[index_3d(x + 1u, y, z)].x;
    let vx_neg = velocity[index_3d(x - 1u, y, z)].x;
    let vy_pos = velocity[index_3d(x, y + 1u, z)].y;
    let vy_neg = velocity[index_3d(x, y - 1u, z)].y;
    let vz_pos = velocity[index_3d(x, y, z + 1u)].z;
    let vz_neg = velocity[index_3d(x, y, z - 1u)].z;

    let div_v = (vx_pos - vx_neg) * 0.5
              + (vy_pos - vy_neg) * 0.5
              + (vz_pos - vz_neg) * 0.5;

    // p^{n+1} = p^n - dt * rho0 * c0^2 * div_v
    pressure[idx] -= params.dt * density * c2 * div_v;
}
