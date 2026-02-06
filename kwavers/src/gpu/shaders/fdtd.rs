//! FDTD compute shader for acoustic field updates

/// FDTD pressure update compute kernel
pub const FDTD_PRESSURE_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> pressure_prev: array<f32>;

@group(0) @binding(1)
var<storage, read> velocity_x: array<f32>;

@group(0) @binding(2)
var<storage, read> velocity_y: array<f32>;

@group(0) @binding(3)
var<storage, read> velocity_z: array<f32>;

@group(0) @binding(4)
var<storage, read_write> pressure: array<f32>;

@group(1) @binding(0)
var<uniform> params: SimParams;

struct SimParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    c0: f32,
    rho0: f32,
}

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 4)
fn fdtd_pressure_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    if (x == 0u || x == params.nx - 1u ||
        y == 0u || y == params.ny - 1u ||
        z == 0u || z == params.nz - 1u) {
        return; // Skip boundaries
    }
    
    let idx = index_3d(x, y, z);
    
    // Compute velocity divergence
    let dvx_dx = (velocity_x[index_3d(x + 1u, y, z)] - velocity_x[idx]) / params.dx;
    let dvy_dy = (velocity_y[index_3d(x, y + 1u, z)] - velocity_y[idx]) / params.dy;
    let dvz_dz = (velocity_z[index_3d(x, y, z + 1u)] - velocity_z[idx]) / params.dz;
    
    let divergence = dvx_dx + dvy_dy + dvz_dz;
    
    // Update pressure using equation of state
    let bulk_modulus = params.rho0 * params.c0 * params.c0;
    pressure[idx] = pressure_prev[idx] - bulk_modulus * params.dt * divergence;
}
"#;
