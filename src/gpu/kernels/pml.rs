//! PML boundary kernels for GPU

/// WGSL shader for PML absorption
pub const PML_ABSORPTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read> sigma_x: array<f32>;
@group(0) @binding(2) var<storage, read> sigma_y: array<f32>;
@group(0) @binding(3) var<storage, read> sigma_z: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
}

@group(1) @binding(0) var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 1)
fn pml_absorb(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    
    // Apply PML absorption
    let sigma = sigma_x[x] + sigma_y[y] + sigma_z[z];
    field[idx] = field[idx] * exp(-sigma * params.dt);
}
"#;
