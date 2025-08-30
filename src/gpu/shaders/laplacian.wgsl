// 3D Laplacian compute shader for wave propagation

struct Params {
    dx: f32,
    dy: f32,
    dz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    let nx = u32(params.nx);
    let ny = u32(params.ny);
    return x + y * nx + z * nx * ny;
}

@compute @workgroup_size(8, 8, 8)
fn laplacian_3d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let nx = u32(params.nx);
    let ny = u32(params.ny);
    let nz = u32(params.nz);
    
    // Boundary check
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    
    // Handle boundaries with zero-flux
    var laplacian = 0.0;
    
    // X direction
    if (x > 0u && x < nx - 1u) {
        let idx_xm = index_3d(x - 1u, y, z);
        let idx_xp = index_3d(x + 1u, y, z);
        laplacian += (input[idx_xm] - 2.0 * input[idx] + input[idx_xp]) / (params.dx * params.dx);
    }
    
    // Y direction
    if (y > 0u && y < ny - 1u) {
        let idx_ym = index_3d(x, y - 1u, z);
        let idx_yp = index_3d(x, y + 1u, z);
        laplacian += (input[idx_ym] - 2.0 * input[idx] + input[idx_yp]) / (params.dy * params.dy);
    }
    
    // Z direction
    if (z > 0u && z < nz - 1u) {
        let idx_zm = index_3d(x, y, z - 1u);
        let idx_zp = index_3d(x, y, z + 1u);
        laplacian += (input[idx_zm] - 2.0 * input[idx] + input[idx_zp]) / (params.dz * params.dz);
    }
    
    output[idx] = laplacian;
}