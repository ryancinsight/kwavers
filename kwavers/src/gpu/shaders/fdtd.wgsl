// FDTD compute shader for acoustic wave propagation

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
}

@group(0) @binding(0)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(1)
var<storage, read_write> velocity: array<vec3<f32>>;

@group(0) @binding(2)
var<storage, read> medium: array<vec2<f32>>; // density, sound_speed

var<push_constant> params: GridParams;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Bounds check
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    let props = medium[idx];
    let density = props.x;
    let c2 = props.y * props.y; // c^2
    
    // Update velocity using pressure gradient
    if (x > 0u && x < params.nx - 1u &&
        y > 0u && y < params.ny - 1u &&
        z > 0u && z < params.nz - 1u) {
        
        // Compute pressure gradients
        let px_plus = pressure[index_3d(x + 1u, y, z)];
        let px_minus = pressure[index_3d(x - 1u, y, z)];
        let py_plus = pressure[index_3d(x, y + 1u, z)];
        let py_minus = pressure[index_3d(x, y - 1u, z)];
        let pz_plus = pressure[index_3d(x, y, z + 1u)];
        let pz_minus = pressure[index_3d(x, y, z - 1u)];
        
        let grad_p = vec3<f32>(
            (px_plus - px_minus) * 0.5,
            (py_plus - py_minus) * 0.5,
            (pz_plus - pz_minus) * 0.5
        );
        
        // Update velocity: v += -dt/ρ * ∇p
        velocity[idx] -= (params.dt / density) * grad_p;
    }
    
    // Update pressure using velocity divergence
    if (x > 0u && x < params.nx - 1u &&
        y > 0u && y < params.ny - 1u &&
        z > 0u && z < params.nz - 1u) {
        
        // Compute velocity divergence
        let vx_plus = velocity[index_3d(x + 1u, y, z)].x;
        let vx_minus = velocity[index_3d(x - 1u, y, z)].x;
        let vy_plus = velocity[index_3d(x, y + 1u, z)].y;
        let vy_minus = velocity[index_3d(x, y - 1u, z)].y;
        let vz_plus = velocity[index_3d(x, y, z + 1u)].z;
        let vz_minus = velocity[index_3d(x, y, z - 1u)].z;
        
        let div_v = (vx_plus - vx_minus) * 0.5 +
                   (vy_plus - vy_minus) * 0.5 +
                   (vz_plus - vz_minus) * 0.5;
        
        // Update pressure: p += -dt * ρc² * ∇·v
        pressure[idx] -= params.dt * density * c2 * div_v;
    }
}