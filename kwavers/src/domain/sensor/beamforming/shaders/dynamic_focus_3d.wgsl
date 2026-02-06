//! 3D Dynamic Focusing Beamforming Compute Shader
//!
//! Advanced beamforming with depth-dependent focusing for improved resolution
//! in volumetric ultrasound imaging. Implements dynamic receive focusing with
//! GPU acceleration.
//!
//! # Features
//! - Depth-dependent focal points
//! - Variable aperture sizing
//! - F-number control
//! - Real-time focusing adjustments
//!
//! # Performance Optimizations
//! - Pre-computed delay tables
//! - Hierarchical focusing zones
//! - Adaptive aperture control

struct DynamicFocusParams {
    // Volume dimensions
    volume_dims: vec3<u32>,
    _padding1: u32,

    // Voxel spacing (meters)
    voxel_spacing: vec3<f32>,
    _padding2: u32,

    // Array configuration
    num_elements: vec3<u32>,
    _padding3: u32,

    // Element spacing (meters)
    element_spacing: vec3<f32>,
    _padding4: u32,

    // Acoustic parameters
    sound_speed: f32,
    sampling_freq: f32,
    center_freq: f32,
    f_number: f32, // F-number for aperture control

    // Dynamic focusing parameters
    min_depth: f32,
    max_depth: f32,
    num_focus_zones: u32,
    _padding5: u32,

    // Processing parameters
    num_frames: u32,
    num_samples: u32,
    enable_variable_aperture: u32, // 0 = false, 1 = true
    _padding6: u32,
};

@group(0) @binding(0)
var<storage, read> rf_data: array<f32>; // [frames][elements][samples]

@group(0) @binding(1)
var<storage, read_write> output_volume: array<f32>; // [x][y][z]

@group(0) @binding(2)
var<uniform> params: DynamicFocusParams;

@group(0) @binding(3)
var<storage, read> apodization_weights: array<f32>; // [elements_x][elements_y][elements_z]

@group(0) @binding(4)
var<storage, read> element_positions: array<vec3<f32>>; // [elements][3]

@group(0) @binding(5)
var<storage, read> focus_delays: array<f32>; // Pre-computed delay tables [zones][elements]

@group(0) @binding(6)
var<storage, read> aperture_masks: array<u32>; // Aperture masks [zones][elements] (bitmask)

/// Convert 3D volume coordinates to linear index
fn volume_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.volume_dims.x + z * params.volume_dims.x * params.volume_dims.y;
}

/// Convert 3D element coordinates to linear index
fn element_index(ex: u32, ey: u32, ez: u32) -> u32 {
    return ex + ey * params.num_elements.x + ez * params.num_elements.x * params.num_elements.y;
}

/// Convert RF data coordinates to linear index
fn rf_data_index(frame: u32, element: u32, sample: u32) -> u32 {
    return frame * (params.num_elements.x * params.num_elements.y * params.num_elements.z * params.num_samples) +
           element * params.num_samples + sample;
}

/// Determine focus zone for given depth
fn get_focus_zone(depth: f32) -> u32 {
    if (depth <= params.min_depth) {
        return 0u;
    }
    if (depth >= params.max_depth) {
        return params.num_focus_zones - 1u;
    }

    let zone_range = (params.max_depth - params.min_depth) / f32(params.num_focus_zones);
    return u32((depth - params.min_depth) / zone_range);
}

/// Calculate dynamic aperture size based on F-number and depth
fn calculate_aperture_size(depth: f32) -> f32 {
    return depth / params.f_number;
}

/// Check if element is within aperture for given voxel
fn is_element_in_aperture(element_pos: vec3<f32>, voxel_pos: vec3<f32>, aperture_size: f32) -> bool {
    let lateral_distance = length(element_pos.xy - voxel_pos.xy);
    return lateral_distance <= aperture_size * 0.5;
}

/// Get pre-computed delay for element and focus zone
fn get_focus_delay(zone: u32, element_idx: u32) -> f32 {
    let total_elements = params.num_elements.x * params.num_elements.y * params.num_elements.z;
    return focus_delays[zone * total_elements + element_idx];
}

/// Calculate variable apodization weight based on distance from focus
fn calculate_variable_apodization(element_pos: vec3<f32>, voxel_pos: vec3<f32>, aperture_size: f32) -> f32 {
    let lateral_distance = length(element_pos.xy - voxel_pos.xy);
    let normalized_distance = lateral_distance / (aperture_size * 0.5);

    if (normalized_distance > 1.0) {
        return 0.0; // Outside aperture
    }

    // Gaussian roll-off at aperture edges
    let sigma = 0.3;
    return exp(-0.5 * normalized_distance * normalized_distance / (sigma * sigma));
}

/// Linear interpolation for sub-sample delays
fn interpolate_sample(rf_samples: array<f32, 1024>, delay: f32, num_samples: u32) -> f32 {
    let sample_idx = floor(delay);
    let fraction = delay - sample_idx;

    if (u32(sample_idx) >= num_samples - 1u) {
        return 0.0; // Out of bounds
    }

    let idx0 = u32(sample_idx);
    let idx1 = idx0 + 1u;

    let sample0 = rf_samples[idx0];
    let sample1 = rf_samples[idx1];

    // Linear interpolation
    return sample0 + fraction * (sample1 - sample0);
}

/// Main dynamic focusing beamforming kernel
@compute @workgroup_size(8, 8, 8)
fn dynamic_focus_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_x = global_id.x;
    let voxel_y = global_id.y;
    let voxel_z = global_id.z;

    // Check bounds
    if (voxel_x >= params.volume_dims.x || voxel_y >= params.volume_dims.y || voxel_z >= params.volume_dims.z) {
        return;
    }

    // Calculate voxel position in world coordinates
    let voxel_pos = vec3<f32>(
        f32(voxel_x) * params.voxel_spacing.x,
        f32(voxel_y) * params.voxel_spacing.y,
        f32(voxel_z) * params.voxel_spacing.z
    );

    // Determine focus zone based on depth
    let focus_zone = get_focus_zone(voxel_pos.z);

    // Calculate aperture size for this depth
    let aperture_size = calculate_aperture_size(voxel_pos.z);

    var sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    let total_elements = params.num_elements.x * params.num_elements.y * params.num_elements.z;

    // Process all elements
    for (var ex: u32 = 0u; ex < params.num_elements.x; ex = ex + 1u) {
        for (var ey: u32 = 0u; ey < params.num_elements.y; ey = ey + 1u) {
            for (var ez: u32 = 0u; ez < params.num_elements.z; ez = ez + 1u) {
                let element_idx = element_index(ex, ey, ez);
                let element_pos = element_positions[element_idx];

                // Check if element is within aperture
                var in_aperture = true;
                if (params.enable_variable_aperture == 1u) {
                    in_aperture = is_element_in_aperture(element_pos, voxel_pos, aperture_size);
                }

                if (!in_aperture) {
                    continue;
                }

                // Get pre-computed delay for this focus zone
                let delay = get_focus_delay(focus_zone, element_idx);

                // Calculate variable apodization weight
                var weight = apodization_weights[element_idx];
                if (params.enable_variable_aperture == 1u) {
                    let variable_weight = calculate_variable_apodization(element_pos, voxel_pos, aperture_size);
                    weight = weight * variable_weight;
                }

                // Sum across all frames and samples with delay
                for (var frame: u32 = 0u; frame < params.num_frames; frame = frame + 1u) {
                    // Collect RF samples for this element
                    var rf_samples: array<f32, 1024>;
                    for (var s: u32 = 0u; s < min(1024u, params.num_samples); s = s + 1u) {
                        rf_samples[s] = rf_data[rf_data_index(frame, element_idx, s)];
                    }

                    // Interpolate delayed sample
                    let delayed_sample = interpolate_sample(rf_samples, delay, params.num_samples);

                    // Accumulate with apodization
                    sum = sum + delayed_sample * weight;
                    weight_sum = weight_sum + weight;
                }
            }
        }
    }

    // Normalize by total weight and write to output volume
    let volume_idx = volume_index(voxel_x, voxel_y, voxel_z);
    if (weight_sum > 0.0) {
        output_volume[volume_idx] = sum / weight_sum;
    } else {
        output_volume[volume_idx] = 0.0;
    }
}

/// Kernel for pre-computing focus delays for all zones and elements
@compute @workgroup_size(64, 1, 1)
fn precompute_focus_delays_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let zone = global_id.y;
    let element_idx = global_id.x;

    if (zone >= params.num_focus_zones) {
        return;
    }

    let total_elements = params.num_elements.x * params.num_elements.y * params.num_elements.z;
    if (element_idx >= total_elements) {
        return;
    }

    // Calculate focus depth for this zone
    let zone_depth = params.min_depth +
        f32(zone) * (params.max_depth - params.min_depth) / f32(params.num_focus_zones - 1u);

    // Focus point at center laterally, at calculated depth
    let focus_pos = vec3<f32>(0.0, 0.0, zone_depth);
    let element_pos = element_positions[element_idx];

    // Calculate time-of-flight delay in samples
    let distance = length(focus_pos - element_pos);
    let time_of_flight = distance / params.sound_speed;
    let delay_samples = time_of_flight * params.sampling_freq;

    // Store delay
    focus_delays[zone * total_elements + element_idx] = delay_samples;
}

/// Kernel for computing aperture masks for variable aperture control
@compute @workgroup_size(64, 1, 1)
fn compute_aperture_masks_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let zone = global_id.y;
    let element_group = global_id.x; // Process elements in groups of 32

    if (zone >= params.num_focus_zones) {
        return;
    }

    let total_elements = params.num_elements.x * params.num_elements.y * params.num_elements.z;
    let elements_per_group = 32u;

    // Calculate focus depth for this zone
    let zone_depth = params.min_depth +
        f32(zone) * (params.max_depth - params.min_depth) / f32(params.num_focus_zones - 1u);

    let aperture_size = calculate_aperture_size(zone_depth);
    let focus_pos = vec3<f32>(0.0, 0.0, zone_depth);

    // Process elements in this group
    for (var i: u32 = 0u; i < elements_per_group; i = i + 1u) {
        let element_idx = element_group * elements_per_group + i;
        if (element_idx >= total_elements) {
            break;
        }

        let element_pos = element_positions[element_idx];
        let in_aperture = is_element_in_aperture(element_pos, focus_pos, aperture_size);

        // Update aperture mask (bitmask)
        let mask_idx = zone * ((total_elements + 31u) / 32u) + element_group;
        if (in_aperture) {
            aperture_masks[mask_idx] = aperture_masks[mask_idx] | (1u << i);
        }
    }
}
