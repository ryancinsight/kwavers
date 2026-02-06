// GPU Operators Compute Shader (WGSL)
//
// Implements common array operations for ultrasound simulation:
// - Element-wise multiplication
// - Element-wise addition/subtraction
// - Scalar multiplication
// - Spatial derivatives (k-space operators)
//
// Optimized for parallel execution on GPU

// Element-wise multiply: out = a * b
@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn elementwise_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Bounds check
    if (idx >= arrayLength(&input_a)) {
        return;
    }

    // Perform multiplication
    output[idx] = input_a[idx] * input_b[idx];
}

// Element-wise addition: out = a + b
@compute @workgroup_size(256, 1, 1)
fn elementwise_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input_a)) {
        return;
    }

    output[idx] = input_a[idx] + input_b[idx];
}

// Element-wise subtraction: out = a - b
@compute @workgroup_size(256, 1, 1)
fn elementwise_subtract(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input_a)) {
        return;
    }

    output[idx] = input_a[idx] - input_b[idx];
}

// Scalar multiplication: out = a * scalar
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_scalar: array<f32>;

struct ScalarParams {
    value: f32,
}

@group(0) @binding(2)
var<uniform> params: ScalarParams;

@compute @workgroup_size(256, 1, 1)
fn scalar_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input)) {
        return;
    }

    output_scalar[idx] = input[idx] * params.value;
}

// Spatial derivative using k-space operator
// This is a placeholder - full implementation requires FFT integration
@compute @workgroup_size(256, 1, 1)
fn spatial_derivative(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input_a)) {
        return;
    }

    // Placeholder: k-space derivative = FFT → multiply by ik → IFFT
    // For now, just copy input to output
    output[idx] = input_a[idx];
}

// 3D element-wise multiply with proper indexing
struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
}

@group(0) @binding(3)
var<uniform> grid: GridParams;

@compute @workgroup_size(8, 8, 4)
fn elementwise_multiply_3d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    // Bounds check
    if (x >= grid.nx || y >= grid.ny || z >= grid.nz) {
        return;
    }

    // 3D to 1D index conversion
    let idx = x + y * grid.nx + z * grid.nx * grid.ny;

    // Perform multiplication
    output[idx] = input_a[idx] * input_b[idx];
}

// Fused multiply-add: out = a * b + c
@group(0) @binding(0)
var<storage, read> fma_a: array<f32>;

@group(0) @binding(1)
var<storage, read> fma_b: array<f32>;

@group(0) @binding(2)
var<storage, read> fma_c: array<f32>;

@group(0) @binding(3)
var<storage, read_write> fma_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn fused_multiply_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&fma_a)) {
        return;
    }

    // FMA is often a single instruction on GPU
    fma_out[idx] = fma(fma_a[idx], fma_b[idx], fma_c[idx]);
}

// Absolute value
@compute @workgroup_size(256, 1, 1)
fn absolute_value(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input)) {
        return;
    }

    output_scalar[idx] = abs(input[idx]);
}

// Maximum reduction (find max value in array)
@group(0) @binding(0)
var<storage, read> reduction_input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> reduction_output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn max_reduction(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Load data into shared memory
    if (idx < arrayLength(&reduction_input)) {
        shared_data[local_idx] = reduction_input[idx];
    } else {
        shared_data[local_idx] = -3.40282e38;  // -FLT_MAX
    }

    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_data[local_idx] = max(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write result
    if (local_idx == 0u) {
        reduction_output[global_id.x / 256u] = shared_data[0];
    }
}
