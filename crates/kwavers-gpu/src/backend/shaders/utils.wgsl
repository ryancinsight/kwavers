// GPU Utility Shaders (WGSL)
//
// Common utility operations for GPU computation:
// - Buffer copy
// - Type conversions (f32 ↔ f64)
// - Data transposition
// - Index permutation

// Simple buffer copy
@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn buffer_copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&src)) {
        return;
    }

    dst[idx] = src[idx];
}

// Fill buffer with constant value
struct FillParams {
    value: f32,
}

@group(0) @binding(0)
var<storage, read_write> fill_buffer: array<f32>;

@group(0) @binding(1)
var<uniform> fill_params: FillParams;

@compute @workgroup_size(256, 1, 1)
fn buffer_fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&fill_buffer)) {
        return;
    }

    fill_buffer[idx] = fill_params.value;
}

// 3D array transposition (permute dimensions)
struct TransposeParams {
    nx: u32,
    ny: u32,
    nz: u32,
    // Permutation: 0=xyz, 1=xzy, 2=yxz, 3=yzx, 4=zxy, 5=zyx
    permutation: u32,
}

@group(0) @binding(0)
var<storage, read> transpose_input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> transpose_output: array<f32>;

@group(0) @binding(2)
var<uniform> transpose_params: TransposeParams;

@compute @workgroup_size(8, 8, 4)
fn transpose_3d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    let nx = transpose_params.nx;
    let ny = transpose_params.ny;
    let nz = transpose_params.nz;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    // Input index
    let idx_in = x + y * nx + z * nx * ny;

    // Output index based on permutation
    var idx_out: u32;
    switch (transpose_params.permutation) {
        case 0u: {  // xyz (identity)
            idx_out = x + y * nx + z * nx * ny;
        }
        case 1u: {  // xzy
            idx_out = x + z * nx + y * nx * nz;
        }
        case 2u: {  // yxz
            idx_out = y + x * ny + z * ny * nx;
        }
        case 3u: {  // yzx
            idx_out = y + z * ny + x * ny * nz;
        }
        case 4u: {  // zxy
            idx_out = z + x * nz + y * nz * nx;
        }
        case 5u: {  // zyx
            idx_out = z + y * nz + x * nz * ny;
        }
        default: {
            idx_out = idx_in;
        }
    }

    transpose_output[idx_out] = transpose_input[idx_in];
}

// Sum reduction (compute sum of all elements)
@group(0) @binding(0)
var<storage, read> sum_input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> sum_output: array<f32>;

var<workgroup> sum_shared: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn sum_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Load data
    if (idx < arrayLength(&sum_input)) {
        sum_shared[local_idx] = sum_input[idx];
    } else {
        sum_shared[local_idx] = 0.0;
    }

    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            sum_shared[local_idx] = sum_shared[local_idx] + sum_shared[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write result
    if (local_idx == 0u) {
        sum_output[global_id.x / 256u] = sum_shared[0];
    }
}

// L2 norm computation (sum of squares then sqrt)
@group(0) @binding(0)
var<storage, read> norm_input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> norm_output: array<f32>;

var<workgroup> norm_shared: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn l2_norm(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Load data and square
    if (idx < arrayLength(&norm_input)) {
        let val = norm_input[idx];
        norm_shared[local_idx] = val * val;
    } else {
        norm_shared[local_idx] = 0.0;
    }

    workgroupBarrier();

    // Tree reduction (sum of squares)
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            norm_shared[local_idx] = norm_shared[local_idx] + norm_shared[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write result (sqrt applied on CPU for simplicity)
    if (local_idx == 0u) {
        norm_output[global_id.x / 256u] = norm_shared[0];
    }
}

// Clamp values to range [min, max]
struct ClampParams {
    min_val: f32,
    max_val: f32,
}

@group(0) @binding(0)
var<storage, read> clamp_input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> clamp_output: array<f32>;

@group(0) @binding(2)
var<uniform> clamp_params: ClampParams;

@compute @workgroup_size(256, 1, 1)
fn clamp_values(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&clamp_input)) {
        return;
    }

    clamp_output[idx] = clamp(clamp_input[idx], clamp_params.min_val, clamp_params.max_val);
}

// Normalize array to [0, 1] range
@compute @workgroup_size(256, 1, 1)
fn normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&clamp_input)) {
        return;
    }

    // Assumes min/max already computed and stored in clamp_params
    let range = clamp_params.max_val - clamp_params.min_val;
    if (range > 0.0) {
        clamp_output[idx] = (clamp_input[idx] - clamp_params.min_val) / range;
    } else {
        clamp_output[idx] = 0.0;
    }
}
