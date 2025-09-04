// FDTD compute shader for acoustic wave propagation
// This implements the finite difference time domain method on GPU

struct SimulationUniforms {
    grid_nx: u32,
    grid_ny: u32,
    grid_nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    time: f32,
    sound_speed: f32,
    density: f32,
}

@group(0) @binding(0) var<uniform> uniforms: SimulationUniforms;
@group(0) @binding(1) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> velocity_z: array<f32>;

// Convert 3D indices to 1D array index
fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return z * uniforms.grid_nx * uniforms.grid_ny + y * uniforms.grid_nx + x;
}

// Check if indices are within bounds
fn is_valid_index(x: u32, y: u32, z: u32) -> bool {
    return x < uniforms.grid_nx && y < uniforms.grid_ny && z < uniforms.grid_nz;
}

// Safe array access with bounds checking
fn get_pressure(x: u32, y: u32, z: u32) -> f32 {
    if (is_valid_index(x, y, z)) {
        return pressure[index_3d(x, y, z)];
    }
    return 0.0;
}

fn get_velocity_x(x: u32, y: u32, z: u32) -> f32 {
    if (is_valid_index(x, y, z)) {
        return velocity_x[index_3d(x, y, z)];
    }
    return 0.0;
}

fn get_velocity_y(x: u32, y: u32, z: u32) -> f32 {
    if (is_valid_index(x, y, z)) {
        return velocity_y[index_3d(x, y, z)];
    }
    return 0.0;
}

fn get_velocity_z(x: u32, y: u32, z: u32) -> f32 {
    if (is_valid_index(x, y, z)) {
        return velocity_z[index_3d(x, y, z)];
    }
    return 0.0;
}

// Update pressure using finite differences
fn update_pressure_at(x: u32, y: u32, z: u32) {
    // Skip boundary points
    if (x == 0 || x >= uniforms.grid_nx - 1 ||
        y == 0 || y >= uniforms.grid_ny - 1 ||
        z == 0 || z >= uniforms.grid_nz - 1) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    
    // Compute velocity divergence using central differences
    let dvx_dx = (get_velocity_x(x + 1, y, z) - get_velocity_x(x - 1, y, z)) / (2.0 * uniforms.dx);
    let dvy_dy = (get_velocity_y(x, y + 1, z) - get_velocity_y(x, y - 1, z)) / (2.0 * uniforms.dy);
    let dvz_dz = (get_velocity_z(x, y, z + 1) - get_velocity_z(x, y, z - 1)) / (2.0 * uniforms.dz);
    
    let div_v = dvx_dx + dvy_dy + dvz_dz;
    
    // Update pressure: dp/dt = -K * div(v)
    // For simplicity, use K = rho * c^2
    let bulk_modulus = uniforms.density * uniforms.sound_speed * uniforms.sound_speed;
    pressure[idx] = pressure[idx] - uniforms.dt * bulk_modulus * div_v;
}

// Update velocity using finite differences  
fn update_velocity_at(x: u32, y: u32, z: u32) {
    // Skip boundary points
    if (x == 0 || x >= uniforms.grid_nx - 1 ||
        y == 0 || y >= uniforms.grid_ny - 1 ||
        z == 0 || z >= uniforms.grid_nz - 1) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    
    // Compute pressure gradients using central differences
    let dp_dx = (get_pressure(x + 1, y, z) - get_pressure(x - 1, y, z)) / (2.0 * uniforms.dx);
    let dp_dy = (get_pressure(x, y + 1, z) - get_pressure(x, y - 1, z)) / (2.0 * uniforms.dy);
    let dp_dz = (get_pressure(x, y, z + 1) - get_pressure(x, y, z - 1)) / (2.0 * uniforms.dz);
    
    // Update velocities: dv/dt = -(1/rho) * grad(p)
    let inv_density = 1.0 / uniforms.density;
    velocity_x[idx] = velocity_x[idx] - uniforms.dt * inv_density * dp_dx;
    velocity_y[idx] = velocity_y[idx] - uniforms.dt * inv_density * dp_dy;
    velocity_z[idx] = velocity_z[idx] - uniforms.dt * inv_density * dp_dz;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Check bounds
    if (!is_valid_index(x, y, z)) {
        return;
    }
    
    // FDTD leapfrog scheme:
    // 1. Update pressure using current velocities
    // 2. Update velocities using new pressure
    
    update_pressure_at(x, y, z);
    workgroupBarrier(); // Ensure all pressure updates complete
    
    update_velocity_at(x, y, z);
}