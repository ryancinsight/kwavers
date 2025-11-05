//! Electromagnetic Physics GPU Kernels
//!
//! This shader implements GPU-accelerated electromagnetic field computations
//! for Maxwell's equations solving using finite difference time domain (FDTD) methods.
//!
//! ## Maxwell's Equations (Time Domain)
//!
//! ∂E/∂t = (1/ε)(∇×H - J - σE)    [Ampere-Maxwell]
//! ∂H/∂t = (1/μ)(∇×E)               [Faraday]
//! ∇·E = ρ/ε                         [Gauss (E)]
//! ∇·H = 0                           [Gauss (H)]
//!
//! ## FDTD Update Equations
//!
//! E^{n+1} = E^n + Δt/ε * (∇×H)^{n+1/2}
//! H^{n+1/2} = H^{n-1/2} + Δt/μ * (∇×E)^n

// Buffer bindings
@group(0) @binding(0)
var<storage, read_write> e_field: array<f32>;

@group(0) @binding(1)
var<storage, read_write> h_field: array<f32>;

@group(0) @binding(2)
var<storage, read> current_density: array<f32>;

@group(0) @binding(3)
var<storage, read> charge_density: array<f32>;

@group(0) @binding(4)
var<uniform> constants: EMConstants;

struct EMConstants {
    /// Electric permittivity (F/m)
    permittivity: f32,
    /// Magnetic permeability (H/m)
    permeability: f32,
    /// Electrical conductivity (S/m)
    conductivity: f32,
    /// Time step (s)
    dt: f32,
    /// Spatial step in x (m)
    dx: f32,
    /// Spatial step in y (m)
    dy: f32,
    /// Spatial step in z (m)
    dz: f32,
    /// Grid dimensions
    nx: u32,
    ny: u32,
    nz: u32,
}

/// Compute curl of H field at position (i,j,k)
fn curl_h(i: u32, j: u32, k: u32) -> vec3<f32> {
    let idx = (k * constants.ny + j) * constants.nx + i;

    // Get neighboring H field values (staggered grid)
    let hx_im1 = if i > 0 { h_field[(idx - 1) * 3] } else { 0.0 };
    let hx_ip1 = if i < constants.nx - 1 { h_field[(idx + 1) * 3] } else { 0.0 };
    let hy_jm1 = if j > 0 { h_field[(idx - constants.nx) * 3 + 1] } else { 0.0 };
    let hy_jp1 = if j < constants.ny - 1 { h_field[(idx + constants.nx) * 3 + 1] } else { 0.0 };
    let hz_km1 = if k > 0 { h_field[(idx - constants.nx * constants.ny) * 3 + 2] } else { 0.0 };
    let hz_kp1 = if k < constants.nz - 1 { h_field[(idx + constants.nx * constants.ny) * 3 + 2] } else { 0.0 };

    // ∇×H = (∂Hz/∂y - ∂Hy/∂z, ∂Hx/∂z - ∂Hz/∂x, ∂Hy/∂x - ∂Hx/∂y)
    let curl_x = (hz_jp1 - hz_jm1) / (2.0 * constants.dy) - (hy_kp1 - hy_km1) / (2.0 * constants.dz);
    let curl_y = (hx_kp1 - hx_km1) / (2.0 * constants.dz) - (hz_ip1 - hz_im1) / (2.0 * constants.dx);
    let curl_z = (hy_ip1 - hy_im1) / (2.0 * constants.dx) - (hx_jp1 - hx_jm1) / (2.0 * constants.dy);

    return vec3<f32>(curl_x, curl_y, curl_z);
}

/// Compute curl of E field at position (i,j,k)
fn curl_e(i: u32, j: u32, k: u32) -> vec3<f32> {
    let idx = (k * constants.ny + j) * constants.nx + i;

    // Get neighboring E field values
    let ex_im1 = if i > 0 { e_field[(idx - 1) * 3] } else { 0.0 };
    let ex_ip1 = if i < constants.nx - 1 { e_field[(idx + 1) * 3] } else { 0.0 };
    let ey_jm1 = if j > 0 { e_field[(idx - constants.nx) * 3 + 1] } else { 0.0 };
    let ey_jp1 = if j < constants.ny - 1 { e_field[(idx + constants.nx) * 3 + 1] } else { 0.0 };
    let ez_km1 = if k > 0 { e_field[(idx - constants.nx * constants.ny) * 3 + 2] } else { 0.0 };
    let ez_kp1 = if k < constants.nz - 1 { e_field[(idx + constants.nx * constants.ny) * 3 + 2] } else { 0.0 };

    // ∇×E = (∂Ez/∂y - ∂Ey/∂z, ∂Ex/∂z - ∂Ez/∂x, ∂Ey/∂x - ∂Ex/∂y)
    let curl_x = (ez_jp1 - ez_jm1) / (2.0 * constants.dy) - (ey_kp1 - ey_km1) / (2.0 * constants.dz);
    let curl_y = (ex_kp1 - ex_km1) / (2.0 * constants.dz) - (ez_ip1 - ez_im1) / (2.0 * constants.dx);
    let curl_z = (ey_ip1 - ey_im1) / (2.0 * constants.dx) - (ex_jp1 - ex_jm1) / (2.0 * constants.dy);

    return vec3<f32>(curl_x, curl_y, curl_z);
}

/// Update electric field using Ampere-Maxwell law
@compute @workgroup_size(8, 8, 1)
fn update_electric_field(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    let j = global_id.y;
    let k = global_id.z;

    if i >= constants.nx || j >= constants.ny || k >= constants.nz {
        return;
    }

    let idx = (k * constants.ny + j) * constants.nx + i;

    // Compute ∇×H at current position
    let curl_h = curl_h(i, j, k);

    // Get current E and J values
    let e_current = vec3<f32>(
        e_field[idx * 3],
        e_field[idx * 3 + 1],
        e_field[idx * 3 + 2]
    );

    let j_current = vec3<f32>(
        current_density[idx * 3],
        current_density[idx * 3 + 1],
        current_density[idx * 3 + 2]
    );

    // Ampere-Maxwell: ∂E/∂t = (1/ε)(∇×H - J - σE)
    let dEdt = (1.0 / constants.permittivity) * (curl_h - j_current - constants.conductivity * e_current);

    // Forward Euler integration: E^{n+1} = E^n + Δt * ∂E/∂t
    let e_new = e_current + constants.dt * dEdt;

    // Store updated E field
    e_field[idx * 3] = e_new.x;
    e_field[idx * 3 + 1] = e_new.y;
    e_field[idx * 3 + 2] = e_new.z;
}

/// Update magnetic field using Faraday's law
@compute @workgroup_size(8, 8, 1)
fn update_magnetic_field(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    let j = global_id.y;
    let k = global_id.z;

    if i >= constants.nx || j >= constants.ny || k >= constants.nz {
        return;
    }

    let idx = (k * constants.ny + j) * constants.nx + i;

    // Compute ∇×E at current position
    let curl_e = curl_e(i, j, k);

    // Get current H value
    let h_current = vec3<f32>(
        h_field[idx * 3],
        h_field[idx * 3 + 1],
        h_field[idx * 3 + 2]
    );

    // Faraday: ∂H/∂t = (1/μ)∇×E
    let dHdt = (1.0 / constants.permeability) * curl_e;

    // Forward Euler integration: H^{n+1/2} = H^{n-1/2} + Δt * ∂H/∂t
    let h_new = h_current + constants.dt * dHdt;

    // Store updated H field
    h_field[idx * 3] = h_new.x;
    h_field[idx * 3 + 1] = h_new.y;
    h_field[idx * 3 + 2] = h_new.z;
}

/// Apply absorbing boundary conditions (Mur's ABC)
@compute @workgroup_size(8, 8, 1)
fn apply_abc(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    let j = global_id.y;
    let k = global_id.z;

    if i >= constants.nx || j >= constants.ny || k >= constants.nz {
        return;
    }

    let idx = (k * constants.ny + j) * constants.nx + i;

    // Simple first-order Mur ABC for demonstration
    // In practice, more sophisticated ABCs like CPML would be used

    // X boundaries
    if i == 0 || i == constants.nx - 1 {
        e_field[idx * 3] *= 0.99; // Simple damping
        e_field[idx * 3 + 1] *= 0.99;
        h_field[idx * 3 + 1] *= 0.99; // Hx and Hz at x boundaries
        h_field[idx * 3 + 2] *= 0.99;
    }

    // Y boundaries
    if j == 0 || j == constants.ny - 1 {
        e_field[idx * 3 + 1] *= 0.99;
        e_field[idx * 3 + 2] *= 0.99;
        h_field[idx * 3] *= 0.99; // Hy and Hz at y boundaries
        h_field[idx * 3 + 2] *= 0.99;
    }

    // Z boundaries
    if k == 0 || k == constants.nz - 1 {
        e_field[idx * 3 + 2] *= 0.99;
        h_field[idx * 3] *= 0.99; // Hx and Hy at z boundaries
        h_field[idx * 3 + 1] *= 0.99;
    }
}

/// Compute divergence of E field (Gauss's law check)
@compute @workgroup_size(8, 8, 1)
fn compute_divergence_e(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    let j = global_id.y;
    let k = global_id.z;

    if i >= constants.nx || j >= constants.ny || k >= constants.nz ||
       i == 0 || i == constants.nx - 1 ||
       j == 0 || j == constants.ny - 1 ||
       k == 0 || k == constants.nz - 1 {
        return;
    }

    let idx = (k * constants.ny + j) * constants.nx + i;

    // Compute ∇·E using central differences
    let ex_im1 = e_field[((k * constants.ny + j) * constants.nx + (i - 1)) * 3];
    let ex_ip1 = e_field[((k * constants.ny + j) * constants.nx + (i + 1)) * 3];
    let ey_jm1 = e_field[((k * constants.ny + (j - 1)) * constants.nx + i) * 3 + 1];
    let ey_jp1 = e_field[((k * constants.ny + (j + 1)) * constants.nx + i) * 3 + 1];
    let ez_km1 = e_field[(((k - 1) * constants.ny + j) * constants.nx + i) * 3 + 2];
    let ez_kp1 = e_field[(((k + 1) * constants.ny + j) * constants.nx + i) * 3 + 2];

    let div_e = (ex_ip1 - ex_im1) / (2.0 * constants.dx) +
                (ey_jp1 - ey_jm1) / (2.0 * constants.dy) +
                (ez_kp1 - ez_km1) / (2.0 * constants.dz);

    // Gauss's law: ∇·E = ρ/ε
    let rho_over_eps = charge_density[idx] / constants.permittivity;
    let residual = div_e - rho_over_eps;

    // Store residual for monitoring (could be in a separate buffer)
    // For now, just ensure it's computed
}

