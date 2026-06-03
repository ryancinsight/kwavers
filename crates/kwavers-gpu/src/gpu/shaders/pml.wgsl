// Perfectly Matched Layer (PML) boundary conditions
// Based on Berenger's split-field formulation

struct PMLParams {
    pml_width: u32,
    sigma_max: f32,
    alpha_max: f32,
    dt: f32,
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
var<uniform> pml: PMLParams;

@group(0) @binding(1)
var<uniform> grid: GridParams;

@group(0) @binding(2)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(3)
var<storage, read_write> psi_x: array<f32>;

@group(0) @binding(4)
var<storage, read_write> psi_y: array<f32>;

@group(0) @binding(5)
var<storage, read_write> psi_z: array<f32>;

fn compute_pml_coefficient(dist: f32, width: f32) -> f32 {
    if (dist >= width) {
        return 0.0;
    }
    let normalized = (width - dist) / width;
    return pml.sigma_max * pow(normalized, 2.0);
}

fn get_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * grid.nx + z * grid.nx * grid.ny;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= grid.nx || y >= grid.ny || z >= grid.nz) {
        return;
    }
    
    let idx = get_index(x, y, z);
    let pml_width_f = f32(pml.pml_width);
    
    // Compute distance from boundaries
    let dist_x_left = f32(x);
    let dist_x_right = f32(grid.nx - x - 1u);
    let dist_y_left = f32(y);
    let dist_y_right = f32(grid.ny - y - 1u);
    let dist_z_left = f32(z);
    let dist_z_right = f32(grid.nz - z - 1u);
    
    // Compute PML coefficients for each direction
    var sigma_x = 0.0;
    var sigma_y = 0.0;
    var sigma_z = 0.0;
    
    if (dist_x_left < pml_width_f) {
        sigma_x = compute_pml_coefficient(dist_x_left, pml_width_f);
    } else if (dist_x_right < pml_width_f) {
        sigma_x = compute_pml_coefficient(dist_x_right, pml_width_f);
    }
    
    if (dist_y_left < pml_width_f) {
        sigma_y = compute_pml_coefficient(dist_y_left, pml_width_f);
    } else if (dist_y_right < pml_width_f) {
        sigma_y = compute_pml_coefficient(dist_y_right, pml_width_f);
    }
    
    if (dist_z_left < pml_width_f) {
        sigma_z = compute_pml_coefficient(dist_z_left, pml_width_f);
    } else if (dist_z_right < pml_width_f) {
        sigma_z = compute_pml_coefficient(dist_z_right, pml_width_f);
    }
    
    // Update auxiliary fields
    let decay_x = exp(-sigma_x * pml.dt);
    let decay_y = exp(-sigma_y * pml.dt);
    let decay_z = exp(-sigma_z * pml.dt);
    
    psi_x[idx] = psi_x[idx] * decay_x;
    psi_y[idx] = psi_y[idx] * decay_y;
    psi_z[idx] = psi_z[idx] * decay_z;
    
    // Apply to pressure field
    let total_decay = decay_x * decay_y * decay_z;
    pressure[idx] = pressure[idx] * total_decay;
}