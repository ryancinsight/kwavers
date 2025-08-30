//! Acoustic absorption compute shader

/// Acoustic absorption compute kernel
pub const ABSORPTION_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(1)
var<storage, read> absorption_coeff: array<f32>;

@group(1) @binding(0)
var<uniform> params: AbsorptionParams;

struct AbsorptionParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
}

@compute @workgroup_size(8, 8, 8)
fn apply_absorption(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    let idx = x + y * params.nx + z * params.nx * params.ny;
    
    // Apply exponential decay
    let decay = exp(-absorption_coeff[idx] * params.dt);
    pressure[idx] = pressure[idx] * decay;
}
"#;
