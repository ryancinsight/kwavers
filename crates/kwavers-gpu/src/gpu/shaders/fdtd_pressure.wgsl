// FDTD pressure update compute shader
//
// Theorem: Second-order scalar wave equation (Yee 1966, staggered time)
//   p^{n+1}[i,j,k] = 2·p^n[i,j,k] − p^{n-1}[i,j,k] + α·∇²p^n[i,j,k]
//
// where α = (c·dt/dx)² (CFL coefficient, assumed isotropic grid dx=dy=dz)
// and ∇² is the 6-point second-order central-difference Laplacian:
//   ∇²p = (p[i-1]+p[i+1]+p[j-1]+p[j+1]+p[k-1]+p[k+1] − 6·p[i,j,k]) / dx²
//
// Combined: p_new = 2·p_n − p_prev + α·(Σ_neighbours − 6·p_n)
//   with α = (c·dt)² / dx²
//
// Boundary treatment: Dirichlet zero (skip boundary cells).
// Interior stencil requires i,j,k ∈ [1, N−2].
//
// Algorithm
// ---------
// 1. Map global invocation ID → (i, j, k)
// 2. Early-exit if out of bounds or on boundary
// 3. Compute 6-point Laplacian of p_curr
// 4. Write p_new = 2*p_curr - p_prev + coeff * laplacian
//
// References
// ----------
// - Yee KS (1966). IEEE Trans Antennas Propag 14(3):302–307.
// - Virieux J (1986). Geophysics 51(4):889–901. (P-SV wave FDTD)
// - Moczo P, Kristek J, Galis M (2014). The Finite-Difference Modelling of
//   Earthquake Motions. Cambridge University Press. (6-point Laplacian, p. 55)

struct PressureParams {
    nx:    u32,
    ny:    u32,
    nz:    u32,
    coeff: f32,   // (c·dt/dx)² = CFL² — dimensionless
}

@group(0) @binding(0)
var<storage, read> pressure_curr: array<f32>;

@group(0) @binding(1)
var<storage, read> pressure_prev: array<f32>;

@group(0) @binding(2)
var<storage, read_write> pressure_new: array<f32>;

@group(1) @binding(0)
var<uniform> params: PressureParams;

/// Row-major 3D index: p[i, j, k] → linear index i + j·nx + k·nx·ny
fn idx3(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.nx + k * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 4)
fn fdtd_pressure_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // Bounds check
    if (i >= params.nx || j >= params.ny || k >= params.nz) {
        return;
    }

    // Dirichlet zero boundary: skip outermost layer
    if (i == 0u || i == params.nx - 1u ||
        j == 0u || j == params.ny - 1u ||
        k == 0u || k == params.nz - 1u) {
        pressure_new[idx3(i, j, k)] = 0.0;
        return;
    }

    let p_c = pressure_curr[idx3(i, j, k)];

    // 6-point isotropic Laplacian (O(Δx²) accurate)
    let lap = pressure_curr[idx3(i - 1u, j, k)]
            + pressure_curr[idx3(i + 1u, j, k)]
            + pressure_curr[idx3(i, j - 1u, k)]
            + pressure_curr[idx3(i, j + 1u, k)]
            + pressure_curr[idx3(i, j, k - 1u)]
            + pressure_curr[idx3(i, j, k + 1u)]
            - 6.0 * p_c;

    // Second-order leapfrog time update
    pressure_new[idx3(i, j, k)] =
        2.0 * p_c
        - pressure_prev[idx3(i, j, k)]
        + params.coeff * lap;
}
