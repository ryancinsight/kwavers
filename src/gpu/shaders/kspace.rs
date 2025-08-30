//! K-space propagation compute shader

/// K-space propagation compute kernel
pub const KSPACE_PROPAGATE_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read_write> field_real: array<f32>;

@group(0) @binding(1)
var<storage, read_write> field_imag: array<f32>;

@group(0) @binding(2)
var<storage, read> kx: array<f32>;

@group(0) @binding(3)
var<storage, read> ky: array<f32>;

@group(0) @binding(4)
var<storage, read> kz: array<f32>;

@group(1) @binding(0)
var<uniform> params: KSpaceParams;

struct KSpaceParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
    c0: f32,
}

@compute @workgroup_size(8, 8, 8)
fn kspace_propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    let idx = x + y * params.nx + z * params.nx * params.ny;
    
    // Compute k^2
    let k_squared = kx[x] * kx[x] + ky[y] * ky[y] + kz[z] * kz[z];
    
    // Compute phase shift
    let omega = params.c0 * sqrt(k_squared);
    let phase = omega * params.dt;
    
    // Apply phase shift (complex multiplication)
    let cos_phase = cos(phase);
    let sin_phase = sin(phase);
    
    let real = field_real[idx];
    let imag = field_imag[idx];
    
    field_real[idx] = real * cos_phase - imag * sin_phase;
    field_imag[idx] = real * sin_phase + imag * cos_phase;
}
"#;
