// Nonlinear acoustic propagation kernel
// Implements Westervelt equation with B/A nonlinearity

struct NonlinearParams {
    beta: f32,           // Nonlinearity parameter B/A
    c0: f32,             // Sound speed
    rho0: f32,           // Density
    dt: f32,             // Time step
    diffusivity: f32,    // Sound diffusivity
}

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
}

@group(0) @binding(0)
var<uniform> params: NonlinearParams;

@group(0) @binding(1)
var<uniform> grid: GridParams;

@group(0) @binding(2)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(3)
var<storage, read> pressure_prev: array<f32>;

@group(0) @binding(4)
var<storage, read_write> velocity: array<vec3<f32>>;

fn get_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * grid.nx + z * grid.nx * grid.ny;
}

fn compute_laplacian(x: u32, y: u32, z: u32) -> f32 {
    let idx = get_index(x, y, z);
    let p_center = pressure[idx];
    
    var laplacian = 0.0;
    
    // X direction
    if (x > 0u && x < grid.nx - 1u) {
        let p_xm = pressure[get_index(x - 1u, y, z)];
        let p_xp = pressure[get_index(x + 1u, y, z)];
        laplacian += (p_xp - 2.0 * p_center + p_xm) / (grid.dx * grid.dx);
    }
    
    // Y direction
    if (y > 0u && y < grid.ny - 1u) {
        let p_ym = pressure[get_index(x, y - 1u, z)];
        let p_yp = pressure[get_index(x, y + 1u, z)];
        laplacian += (p_yp - 2.0 * p_center + p_ym) / (grid.dy * grid.dy);
    }
    
    // Z direction
    if (z > 0u && z < grid.nz - 1u) {
        let p_zm = pressure[get_index(x, y, z - 1u)];
        let p_zp = pressure[get_index(x, y, z + 1u)];
        laplacian += (p_zp - 2.0 * p_center + p_zm) / (grid.dz * grid.dz);
    }
    
    return laplacian;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Boundary check
    if (x >= grid.nx || y >= grid.ny || z >= grid.nz) {
        return;
    }
    
    // Skip boundary points
    if (x == 0u || x == grid.nx - 1u ||
        y == 0u || y == grid.ny - 1u ||
        z == 0u || z == grid.nz - 1u) {
        return;
    }
    
    let idx = get_index(x, y, z);
    
    // Compute spatial derivatives
    let laplacian = compute_laplacian(x, y, z);
    
    // Get current and previous pressure
    let p = pressure[idx];
    let p_prev = pressure_prev[idx];
    
    // Compute time derivative
    let dp_dt = (p - p_prev) / params.dt;
    
    // Westervelt equation:
    // ∂²p/∂t² = c²∇²p + (β/ρc²)(∂p/∂t)² + δ∇²(∂p/∂t)
    
    // Linear term
    let linear_term = params.c0 * params.c0 * laplacian;
    
    // Nonlinear term (B/A nonlinearity)
    let nonlinear_term = (params.beta / (params.rho0 * params.c0 * params.c0)) * dp_dt * dp_dt;
    
    // Diffusion term (thermoviscous losses)
    let diffusion_term = params.diffusivity * laplacian * dp_dt;
    
    // Update pressure using leapfrog integration
    let d2p_dt2 = linear_term + nonlinear_term + diffusion_term;
    let p_new = 2.0 * p - p_prev + d2p_dt2 * params.dt * params.dt;
    
    // Store updated pressure
    pressure[idx] = p_new;
}