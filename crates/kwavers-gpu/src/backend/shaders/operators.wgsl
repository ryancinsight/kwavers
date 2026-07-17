// GPU spatial-derivative compute shader.
//
// Elementwise and generic storage operations are dispatched by Hephaestus.
// This source retains only the finite-difference operation specific to Kwavers.

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

struct DerivativeParams {
    nx: u32,
    ny: u32,
    nz: u32,
    direction: u32,
}

@group(0) @binding(2)
var<uniform> derivative_params: DerivativeParams;

// Second-order central differences in the interior and first-order one-sided
// differences at domain boundaries. Grid spacing is intentionally absent: this
// backend contract computes an index-space derivative.
@compute @workgroup_size(256, 1, 1)
fn spatial_derivative(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = derivative_params.nx * derivative_params.ny * derivative_params.nz;

    if (idx >= total) {
        return;
    }

    let yz = derivative_params.ny * derivative_params.nz;
    let x = idx / yz;
    let rem = idx - x * yz;
    let y = rem / derivative_params.nz;
    let z = rem - y * derivative_params.nz;

    if (derivative_params.direction == 0u) {
        if (derivative_params.nx < 2u) {
            output[idx] = 0.0;
        } else if (x == 0u) {
            output[idx] = input[idx + yz] - input[idx];
        } else if (x + 1u == derivative_params.nx) {
            output[idx] = input[idx] - input[idx - yz];
        } else {
            output[idx] = 0.5 * (input[idx + yz] - input[idx - yz]);
        }
        return;
    }

    if (derivative_params.direction == 1u) {
        if (derivative_params.ny < 2u) {
            output[idx] = 0.0;
        } else if (y == 0u) {
            output[idx] = input[idx + derivative_params.nz] - input[idx];
        } else if (y + 1u == derivative_params.ny) {
            output[idx] = input[idx] - input[idx - derivative_params.nz];
        } else {
            output[idx] = 0.5 * (input[idx + derivative_params.nz] - input[idx - derivative_params.nz]);
        }
        return;
    }

    if (derivative_params.direction == 2u) {
        if (derivative_params.nz < 2u) {
            output[idx] = 0.0;
        } else if (z == 0u) {
            output[idx] = input[idx + 1u] - input[idx];
        } else if (z + 1u == derivative_params.nz) {
            output[idx] = input[idx] - input[idx - 1u];
        } else {
            output[idx] = 0.5 * (input[idx + 1u] - input[idx - 1u]);
        }
        return;
    }

    output[idx] = 0.0;
}
