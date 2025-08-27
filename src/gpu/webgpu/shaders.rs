//! WebGPU shader management

use crate::error::KwaversResult;

#[cfg(feature = "wgpu")]
use wgpu::{ComputePipeline, Device};

/// Acoustic field update shader in WGSL
pub const ACOUSTIC_SHADER: &str = r#"
struct SimulationParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: SimulationParams;
@group(0) @binding(1) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> velocity_z: array<f32>;
@group(0) @binding(5) var<storage, read> density: array<f32>;
@group(0) @binding(6) var<storage, read> sound_speed: array<f32>;

fn get_index(x: u32, y: u32, z: u32) -> u32 {
    return z * params.ny * params.nx + y * params.nx + x;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.nx || 
        global_id.y >= params.ny || 
        global_id.z >= params.nz) {
        return;
    }
    
    let idx = get_index(global_id.x, global_id.y, global_id.z);
    let rho = density[idx];
    let c = sound_speed[idx];
    let c2 = c * c;
    
    // Update pressure using velocity divergence
    var div_v: f32 = 0.0;
    
    if (global_id.x > 0u && global_id.x < params.nx - 1u) {
        let idx_xp = get_index(global_id.x + 1u, global_id.y, global_id.z);
        let idx_xm = get_index(global_id.x - 1u, global_id.y, global_id.z);
        div_v += (velocity_x[idx_xp] - velocity_x[idx_xm]) / (2.0 * params.dx);
    }
    
    if (global_id.y > 0u && global_id.y < params.ny - 1u) {
        let idx_yp = get_index(global_id.x, global_id.y + 1u, global_id.z);
        let idx_ym = get_index(global_id.x, global_id.y - 1u, global_id.z);
        div_v += (velocity_y[idx_yp] - velocity_y[idx_ym]) / (2.0 * params.dy);
    }
    
    if (global_id.z > 0u && global_id.z < params.nz - 1u) {
        let idx_zp = get_index(global_id.x, global_id.y, global_id.z + 1u);
        let idx_zm = get_index(global_id.x, global_id.y, global_id.z - 1u);
        div_v += (velocity_z[idx_zp] - velocity_z[idx_zm]) / (2.0 * params.dz);
    }
    
    // Update pressure
    pressure[idx] -= rho * c2 * div_v * params.dt;
}
"#;

/// Thermal field update shader in WGSL
pub const THERMAL_SHADER: &str = r#"
struct ThermalParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: ThermalParams;
@group(0) @binding(1) var<storage, read_write> temperature: array<f32>;
@group(0) @binding(2) var<storage, read> heat_rate: array<f32>;
@group(0) @binding(3) var<storage, read> thermal_conductivity: array<f32>;
@group(0) @binding(4) var<storage, read> specific_heat: array<f32>;
@group(0) @binding(5) var<storage, read> density: array<f32>;

fn get_index(x: u32, y: u32, z: u32) -> u32 {
    return z * params.ny * params.nx + y * params.nx + x;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.nx || 
        global_id.y >= params.ny || 
        global_id.z >= params.nz) {
        return;
    }
    
    let idx = get_index(global_id.x, global_id.y, global_id.z);
    
    // Get material properties
    let k = thermal_conductivity[idx];
    let cp = specific_heat[idx];
    let rho = density[idx];
    let q = heat_rate[idx];
    
    // Compute thermal diffusivity
    let alpha = k / (rho * cp);
    
    // Compute Laplacian of temperature
    var laplacian: f32 = 0.0;
    
    // X direction
    if (global_id.x > 0u && global_id.x < params.nx - 1u) {
        let idx_xp = get_index(global_id.x + 1u, global_id.y, global_id.z);
        let idx_xm = get_index(global_id.x - 1u, global_id.y, global_id.z);
        let T_xp = temperature[idx_xp];
        let T_xm = temperature[idx_xm];
        let T = temperature[idx];
        laplacian += (T_xp - 2.0 * T + T_xm) / (params.dx * params.dx);
    }
    
    // Y direction
    if (global_id.y > 0u && global_id.y < params.ny - 1u) {
        let idx_yp = get_index(global_id.x, global_id.y + 1u, global_id.z);
        let idx_ym = get_index(global_id.x, global_id.y - 1u, global_id.z);
        let T_yp = temperature[idx_yp];
        let T_ym = temperature[idx_ym];
        let T = temperature[idx];
        laplacian += (T_yp - 2.0 * T + T_ym) / (params.dy * params.dy);
    }
    
    // Z direction
    if (global_id.z > 0u && global_id.z < params.nz - 1u) {
        let idx_zp = get_index(global_id.x, global_id.y, global_id.z + 1u);
        let idx_zm = get_index(global_id.x, global_id.y, global_id.z - 1u);
        let T_zp = temperature[idx_zp];
        let T_zm = temperature[idx_zm];
        let T = temperature[idx];
        laplacian += (T_zp - 2.0 * T + T_zm) / (params.dz * params.dz);
    }
    
    // Update temperature using heat equation
    temperature[idx] += params.dt * (alpha * laplacian + q / (rho * cp));
}
"#;

/// Create acoustic compute pipeline
#[cfg(feature = "wgpu")]
pub fn create_acoustic_pipeline(device: &Device) -> KwaversResult<ComputePipeline> {
    use wgpu::util::DeviceExt;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Acoustic Shader"),
        source: wgpu::ShaderSource::Wgsl(ACOUSTIC_SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Acoustic Pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(pipeline)
}

/// Create thermal compute pipeline
#[cfg(feature = "wgpu")]
pub fn create_thermal_pipeline(device: &Device) -> KwaversResult<ComputePipeline> {
    use wgpu::util::DeviceExt;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Thermal Shader"),
        source: wgpu::ShaderSource::Wgsl(THERMAL_SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Thermal Pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(pipeline)
}
