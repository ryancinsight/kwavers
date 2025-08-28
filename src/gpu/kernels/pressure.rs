//! Pressure field kernels

/// WGSL shader for nonlinear pressure computation
pub const NONLINEAR_PRESSURE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> density: array<f32>;
@group(0) @binding(2) var<storage, read> sound_speed: array<f32>;
@group(0) @binding(3) var<storage, read> nonlinearity: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
    dx: f32,
}

@group(1) @binding(0) var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

fn laplacian(x: u32, y: u32, z: u32) -> f32 {
    let idx = index_3d(x, y, z);
    let dx2 = params.dx * params.dx;
    
    var lap = -6.0 * pressure[idx];
    
    if (x > 0u) { lap = lap + pressure[index_3d(x - 1u, y, z)]; }
    if (x < params.nx - 1u) { lap = lap + pressure[index_3d(x + 1u, y, z)]; }
    if (y > 0u) { lap = lap + pressure[index_3d(x, y - 1u, z)]; }
    if (y < params.ny - 1u) { lap = lap + pressure[index_3d(x, y + 1u, z)]; }
    if (z > 0u) { lap = lap + pressure[index_3d(x, y, z - 1u)]; }
    if (z < params.nz - 1u) { lap = lap + pressure[index_3d(x, y, z + 1u)]; }
    
    return lap / dx2;
}

@compute @workgroup_size(8, 8, 1)
fn nonlinear_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    if (x == 0u || x >= params.nx - 1u ||
        y == 0u || y >= params.ny - 1u ||
        z == 0u || z >= params.nz - 1u) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    let rho = density[idx];
    let c = sound_speed[idx];
    let beta = nonlinearity[idx];
    
    // Wave equation with nonlinearity
    let linear_term = c * c * laplacian(x, y, z);
    let nonlinear_term = beta * pressure[idx] * pressure[idx] / (rho * c * c);
    
    pressure[idx] = pressure[idx] + params.dt * (linear_term + nonlinear_term);
}
"#;
