// Absorption kernel for acoustic wave propagation
// Implements power law absorption: α = α₀ * |ω|^y

struct AbsorptionParams {
    alpha_coeff: f32,
    alpha_power: f32,
    reference_freq: f32,
    dt: f32,
}

@group(0) @binding(0)
var<uniform> params: AbsorptionParams;

@group(0) @binding(1)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(2)
var<storage, read> frequency_spectrum: array<f32>;

@group(0) @binding(3)
var<storage, read> grid_size: vec3<u32>;

fn compute_absorption(freq: f32) -> f32 {
    let normalized_freq = freq / params.reference_freq;
    return params.alpha_coeff * pow(abs(normalized_freq), params.alpha_power);
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= grid_size.x || 
        global_id.y >= grid_size.y || 
        global_id.z >= grid_size.z) {
        return;
    }
    
    let idx = global_id.x + 
              global_id.y * grid_size.x + 
              global_id.z * grid_size.x * grid_size.y;
    
    // Apply frequency-dependent absorption
    let freq = frequency_spectrum[idx];
    let alpha = compute_absorption(freq);
    
    // Apply absorption using exponential decay
    // p(t+dt) = p(t) * exp(-alpha * c * dt)
    let attenuation = exp(-alpha * 1500.0 * params.dt); // c = 1500 m/s
    pressure[idx] = pressure[idx] * attenuation;
}