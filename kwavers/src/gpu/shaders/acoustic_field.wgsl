// Acoustic field propagation compute shader
// Implements finite difference wave equation solver on GPU

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    _padding: u32,
    dt: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    c: f32,
    _padding2: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Boundary check
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    // Skip boundary points (need at least 1 neighbor on each side)
    if (x == 0u || x >= params.nx - 1u ||
        y == 0u || y >= params.ny - 1u ||
        z == 0u || z >= params.nz - 1u) {
        let idx = index_3d(x, y, z);
        pressure_out[idx] = pressure_in[idx];
        return;
    }
    
    let idx = index_3d(x, y, z);
    
    // Get neighboring values for Laplacian
    let p_xm = pressure_in[index_3d(x - 1u, y, z)];
    let p_xp = pressure_in[index_3d(x + 1u, y, z)];
    let p_ym = pressure_in[index_3d(x, y - 1u, z)];
    let p_yp = pressure_in[index_3d(x, y + 1u, z)];
    let p_zm = pressure_in[index_3d(x, y, z - 1u)];
    let p_zp = pressure_in[index_3d(x, y, z + 1u)];
    let p_center = pressure_in[idx];
    
    // Compute Laplacian using central differences
    let laplacian_x = (p_xp - 2.0 * p_center + p_xm) / (params.dx * params.dx);
    let laplacian_y = (p_yp - 2.0 * p_center + p_ym) / (params.dy * params.dy);
    let laplacian_z = (p_zp - 2.0 * p_center + p_zm) / (params.dz * params.dz);
    let laplacian = laplacian_x + laplacian_y + laplacian_z;
    
    // Wave equation update: p_new = p_old + c^2 * dt^2 * laplacian
    let c2_dt2 = params.c * params.c * params.dt * params.dt;
    pressure_out[idx] = p_center + c2_dt2 * laplacian;
}

// Additional kernel for velocity update
@compute @workgroup_size(8, 8, 8)
fn update_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    // Velocity update based on pressure gradient
    // v_new = v_old - (dt/rho) * grad(p)
    // Implementation would go here
}