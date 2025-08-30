// FDTD compute shader for acoustic wave propagation

struct Params {
    dt: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    nx: u32,
    ny: u32,
    nz: u32,
}

@group(0) @binding(0)
var<storage, read_write> pressure_new: array<f32>;

@group(0) @binding(1)
var<storage, read> pressure: array<f32>;

@group(0) @binding(2)
var<storage, read> pressure_prev: array<f32>;

@group(0) @binding(3)
var<storage, read> density: array<f32>;

@group(0) @binding(4)
var<storage, read> sound_speed: array<f32>;

@group(0) @binding(5)
var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 8)
fn fdtd_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Boundary check
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    // Skip boundaries (handled separately)
    if (x == 0u || x == params.nx - 1u ||
        y == 0u || y == params.ny - 1u ||
        z == 0u || z == params.nz - 1u) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    
    // Get material properties
    let c = sound_speed[idx];
    let rho = density[idx];
    let c2 = c * c;
    
    // Compute spatial derivatives (2nd order central difference)
    let idx_xm = index_3d(x - 1u, y, z);
    let idx_xp = index_3d(x + 1u, y, z);
    let idx_ym = index_3d(x, y - 1u, z);
    let idx_yp = index_3d(x, y + 1u, z);
    let idx_zm = index_3d(x, y, z - 1u);
    let idx_zp = index_3d(x, y, z + 1u);
    
    let d2p_dx2 = (pressure[idx_xm] - 2.0 * pressure[idx] + pressure[idx_xp]) / (params.dx * params.dx);
    let d2p_dy2 = (pressure[idx_ym] - 2.0 * pressure[idx] + pressure[idx_yp]) / (params.dy * params.dy);
    let d2p_dz2 = (pressure[idx_zm] - 2.0 * pressure[idx] + pressure[idx_zp]) / (params.dz * params.dz);
    
    let laplacian = d2p_dx2 + d2p_dy2 + d2p_dz2;
    
    // FDTD update equation
    let dt2 = params.dt * params.dt;
    pressure_new[idx] = 2.0 * pressure[idx] - pressure_prev[idx] + c2 * dt2 * laplacian;
}