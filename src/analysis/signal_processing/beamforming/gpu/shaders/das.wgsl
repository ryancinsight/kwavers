// Delay-and-Sum (DAS) Beamforming Compute Shader
//
// Mathematical Foundation:
//   y(r, t) = Σᵢ₌₁ᴺ wᵢ · xᵢ(t - τᵢ(r))
//
// where:
//   - N = number of sensors
//   - wᵢ = apodization weight for sensor i
//   - xᵢ(t) = received RF signal at sensor i
//   - τᵢ(r) = time-of-flight delay from focal point r to sensor i
//   - y(r, t) = beamformed output
//
// Time-of-flight calculation:
//   τᵢ(r) = ||rᵢ - r|| / c
//
// Implementation:
//   - Parallelization: Each workgroup processes multiple sensors for one focal point
//   - Interpolation: Nearest-neighbor (integer sample shifts)
//   - Accumulation: Atomic addition for thread-safe summation
//
// References:
//   - Van Trees, H. L. (2002). Optimum Array Processing.
//   - Treeby, B. E. & Cox, B. T. (2010). k-Wave: MATLAB toolbox.

// Push constants for per-focal-point parameters
struct PushConstants {
    n_sensors: u32,
    n_samples: u32,
    sampling_rate: f32,
    sound_speed: f32,
    focal_x: f32,
    focal_y: f32,
    focal_z: f32,
    _padding: u32,  // Align to 32 bytes
}

var<push_constant> params: PushConstants;

// Storage buffers
@group(0) @binding(0)
var<storage, read> rf_data: array<f32>;  // Input RF data: [n_sensors × n_samples]

@group(0) @binding(1)
var<storage, read> sensor_positions: array<vec3<f32>>;  // Sensor positions [m]: [n_sensors × 3]

@group(0) @binding(2)
var<storage, read> apodization: array<f32>;  // Apodization weights: [n_sensors]

@group(0) @binding(3)
var<storage, read_write> output: array<atomic<i32>>;  // Beamformed output (atomic for thread safety)

// Helper function: Compute 1D index from sensor and sample indices
fn rf_index(sensor_idx: u32, sample_idx: u32) -> u32 {
    return sensor_idx * params.n_samples + sample_idx;
}

// Helper function: Compute Euclidean distance
fn distance_3d(p1: vec3<f32>, p2: vec3<f32>) -> f32 {
    let diff = p1 - p2;
    return sqrt(dot(diff, diff));
}

// Main compute kernel
@compute @workgroup_size(256, 1, 1)
fn das_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sensor_idx = global_id.x;

    // Bounds check
    if (sensor_idx >= params.n_sensors) {
        return;
    }

    // 1. Compute time-of-flight delay
    let focal_point = vec3<f32>(params.focal_x, params.focal_y, params.focal_z);
    let sensor_pos = sensor_positions[sensor_idx];
    let distance = distance_3d(focal_point, sensor_pos);
    let delay_seconds = distance / params.sound_speed;
    let delay_samples = delay_seconds * params.sampling_rate;

    // 2. Convert to integer sample index (nearest-neighbor interpolation)
    let sample_idx = u32(delay_samples + 0.5);  // Round to nearest

    // 3. Apply delay and weighted sum
    if (sample_idx < params.n_samples) {
        let rf_idx = rf_index(sensor_idx, sample_idx);
        let rf_value = rf_data[rf_idx];
        let weight = apodization[sensor_idx];
        let weighted_value = rf_value * weight;

        // 4. Atomic accumulation (thread-safe summation)
        // Convert float to fixed-point integer for atomic operations
        // Scale by 1000 to preserve 3 decimal places
        let fixed_value = i32(weighted_value * 1000.0);
        atomicAdd(&output[0], fixed_value);
    }
}

// Alternative kernel with linear interpolation (higher accuracy)
@compute @workgroup_size(256, 1, 1)
fn das_kernel_linear_interp(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sensor_idx = global_id.x;

    // Bounds check
    if (sensor_idx >= params.n_sensors) {
        return;
    }

    // 1. Compute time-of-flight delay
    let focal_point = vec3<f32>(params.focal_x, params.focal_y, params.focal_z);
    let sensor_pos = sensor_positions[sensor_idx];
    let distance = distance_3d(focal_point, sensor_pos);
    let delay_seconds = distance / params.sound_speed;
    let delay_samples = delay_seconds * params.sampling_rate;

    // 2. Linear interpolation
    let sample_idx_low = u32(floor(delay_samples));
    let sample_idx_high = sample_idx_low + 1u;
    let alpha = fract(delay_samples);  // Interpolation factor

    // 3. Interpolate RF value
    var interpolated_value = 0.0;
    if (sample_idx_high < params.n_samples) {
        let rf_low = rf_data[rf_index(sensor_idx, sample_idx_low)];
        let rf_high = rf_data[rf_index(sensor_idx, sample_idx_high)];
        interpolated_value = rf_low * (1.0 - alpha) + rf_high * alpha;
    } else if (sample_idx_low < params.n_samples) {
        // Boundary case: only low sample is valid
        interpolated_value = rf_data[rf_index(sensor_idx, sample_idx_low)];
    }

    // 4. Apply apodization and accumulate
    let weight = apodization[sensor_idx];
    let weighted_value = interpolated_value * weight;
    let fixed_value = i32(weighted_value * 1000.0);
    atomicAdd(&output[0], fixed_value);
}

// Kernel with workgroup reduction (optimized for large arrays)
var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn das_kernel_optimized(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let sensor_idx = global_id.x;
    let local_idx = local_id.x;

    // Initialize shared memory
    shared_sums[local_idx] = 0.0;

    // Compute contribution if within bounds
    if (sensor_idx < params.n_sensors) {
        // 1. Compute time-of-flight delay
        let focal_point = vec3<f32>(params.focal_x, params.focal_y, params.focal_z);
        let sensor_pos = sensor_positions[sensor_idx];
        let distance = distance_3d(focal_point, sensor_pos);
        let delay_seconds = distance / params.sound_speed;
        let delay_samples = delay_seconds * params.sampling_rate;

        // 2. Get RF value with nearest-neighbor interpolation
        let sample_idx = u32(delay_samples + 0.5);

        if (sample_idx < params.n_samples) {
            let rf_idx = rf_index(sensor_idx, sample_idx);
            let rf_value = rf_data[rf_idx];
            let weight = apodization[sensor_idx];
            shared_sums[local_idx] = rf_value * weight;
        }
    }

    // Synchronize workgroup
    workgroupBarrier();

    // Parallel reduction within workgroup
    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_sums[local_idx] += shared_sums[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread in workgroup writes to global memory
    if (local_idx == 0u) {
        let fixed_value = i32(shared_sums[0] * 1000.0);
        atomicAdd(&output[0], fixed_value);
    }
}
