//! FDTD kernels for GPU acceleration

/// WGSL shader for pressure update in FDTD
pub const PRESSURE_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> velocity_x: array<f32>;
@group(0) @binding(2) var<storage, read> velocity_y: array<f32>;
@group(0) @binding(3) var<storage, read> velocity_z: array<f32>;
@group(0) @binding(4) var<storage, read> density: array<f32>;
@group(0) @binding(5) var<storage, read> sound_speed: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

@group(1) @binding(0) var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 1)
fn pressure_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    
    // Compute velocity divergence
    let dvx_dx = (velocity_x[index_3d(x + 1u, y, z)] - velocity_x[index_3d(x - 1u, y, z)]) / (2.0 * params.dx);
    let dvy_dy = (velocity_y[index_3d(x, y + 1u, z)] - velocity_y[index_3d(x, y - 1u, z)]) / (2.0 * params.dy);
    let dvz_dz = (velocity_z[index_3d(x, y, z + 1u)] - velocity_z[index_3d(x, y, z - 1u)]) / (2.0 * params.dz);
    
    let divergence = dvx_dx + dvy_dy + dvz_dz;
    
    // Update pressure using wave equation
    let rho = density[idx];
    let c = sound_speed[idx];
    let bulk_modulus = rho * c * c;
    
    pressure[idx] = pressure[idx] - bulk_modulus * divergence * params.dt;
}
"#;

/// WGSL shader for velocity update in FDTD
pub const VELOCITY_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_z: array<f32>;
@group(0) @binding(4) var<storage, read> density: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

@group(1) @binding(0) var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 1)
fn velocity_update_x(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx - 1u || y >= params.ny || z >= params.nz) {
        return;
    }
    
    if (x == 0u || y == 0u || y >= params.ny - 1u ||
        z == 0u || z >= params.nz - 1u) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    let idx_next = index_3d(x + 1u, y, z);
    
    // Compute pressure gradient
    let dp_dx = (pressure[idx_next] - pressure[idx]) / params.dx;
    
    // Average density at staggered location
    let rho = 0.5 * (density[idx] + density[idx_next]);
    
    // Update velocity
    velocity_x[idx] = velocity_x[idx] - (params.dt / rho) * dp_dx;
}
"#;
