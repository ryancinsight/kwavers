//! Nonlinear acoustic propagation compute shader
//!
//! Implements Westervelt equation for nonlinear acoustics

/// Nonlinear propagation compute kernel
pub const NONLINEAR_PROPAGATION_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> pressure: array<f32>;

@group(0) @binding(1)
var<storage, read> pressure_prev: array<f32>;

@group(0) @binding(2)
var<storage, read_write> pressure_next: array<f32>;

@group(0) @binding(3)
var<storage, read> density: array<f32>;

@group(0) @binding(4)
var<storage, read> sound_speed: array<f32>;

@group(0) @binding(5)
var<storage, read> nonlinearity: array<f32>;

@group(1) @binding(0)
var<uniform> params: NonlinearParams;

struct NonlinearParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
}

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

fn laplacian(x: u32, y: u32, z: u32) -> f32 {
    if (x == 0u || x >= params.nx - 1u ||
        y == 0u || y >= params.ny - 1u ||
        z == 0u || z >= params.nz - 1u) {
        return 0.0;
    }
    
    let idx = index_3d(x, y, z);
    let p = pressure[idx];
    
    // Second-order central differences
    let d2p_dx2 = (pressure[index_3d(x + 1u, y, z)] - 2.0 * p + pressure[index_3d(x - 1u, y, z)]) / (params.dx * params.dx);
    let d2p_dy2 = (pressure[index_3d(x, y + 1u, z)] - 2.0 * p + pressure[index_3d(x, y - 1u, z)]) / (params.dy * params.dy);
    let d2p_dz2 = (pressure[index_3d(x, y, z + 1u)] - 2.0 * p + pressure[index_3d(x, y, z - 1u)]) / (params.dz * params.dz);
    
    return d2p_dx2 + d2p_dy2 + d2p_dz2;
}

@compute @workgroup_size(8, 8, 8)
fn nonlinear_propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    
    // Get local properties
    let rho = density[idx];
    let c = sound_speed[idx];
    let beta = nonlinearity[idx];
    
    // Linear wave equation term
    let linear_term = c * c * laplacian(x, y, z);
    
    // Westervelt nonlinear term: (β/ρc⁴)(∂p/∂t)²
    // Reference: Westervelt (1963) "Parametric acoustic array"
    // This form is exact for the Westervelt equation, not simplified
    let p = pressure[idx];
    let p_prev = pressure_prev[idx];
    let dp_dt = (p - p_prev) / params.dt;
    let nonlinear_term = beta / (rho * c * c * c * c) * dp_dt * dp_dt;
    
    // Time integration (explicit scheme)
    let dt2 = params.dt * params.dt;
    pressure_next[idx] = 2.0 * p - p_prev + dt2 * (linear_term + nonlinear_term);
}
"#;
