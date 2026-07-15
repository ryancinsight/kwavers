// 3D Delay-and-Sum with dynamic (per-depth-zone) transmit focusing.
//
// Shares the validated receive geometry and RF interpolation of the static
// `beamforming_3d.wgsl` (which carries the CPU differential test). The
// dynamic-focus extension selects a depth zone per voxel and adds the
// precomputed transmit-focus delay for that zone, forming the round-trip
// time-of-flight for a zone-focused transmit:
//
//   τ_i(r_v) = τ_tx(zone, i) + |r_v − r_i| / c · f_s
//
// where τ_tx(zone, i) = focus_delays[zone·M + i] is prepared on the host as
// |focus(zone) − r_i| / c · f_s with focus(zone) = (0, 0, depth(zone)).
//
// Buffer layout (must match `dynamic_focus_dispatch.rs`):
//   rf_data           : f32[ (frame*channels + ch) * num_samples + sample ]
//   output            : f32[ x + y*VX + z*VX*VY ]
//   apodization       : f32[ ch ]
//   element_positions : f32[ 3*ch + {0,1,2} ]
//   focus_delays      : f32[ zone*channels + ch ]
//   aperture_masks    : u32[ zone*ceil(channels/32) + ch/32 ]  (bit ch%32)

struct DfParams {
    // vec4 (not vec3) so each block is exactly 16 bytes, matching the host
    // struct's `[T; 3] + _padding` layout (see beamforming_3d.wgsl).
    volume_dims: vec4<u32>,
    voxel_spacing: vec4<f32>,
    num_elements: vec4<u32>,
    element_spacing: vec4<f32>,
    sound_speed: f32,
    sampling_freq: f32,
    center_freq: f32,
    f_number: f32,
    min_depth: f32,
    max_depth: f32,
    num_focus_zones: u32,
    pad5: u32,
    num_frames: u32,
    num_samples: u32,
    enable_variable_aperture: u32,
    pad6: u32,
}

@group(0) @binding(0) var<storage, read> rf_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: DfParams;
@group(0) @binding(3) var<storage, read> apodization: array<f32>;
@group(0) @binding(4) var<storage, read> element_positions: array<f32>;
@group(0) @binding(5) var<storage, read> focus_delays: array<f32>;
@group(0) @binding(6) var<storage, read> aperture_masks: array<u32>;

fn rf_interp(frame: u32, ch: u32, channels: u32, tau_s: f32) -> f32 {
    if (tau_s < 0.0) {
        return 0.0;
    }
    let n0 = u32(floor(tau_s));
    if (n0 + 1u >= params.num_samples) {
        return 0.0;
    }
    let alpha = tau_s - f32(n0);
    let base = (frame * channels + ch) * params.num_samples;
    let s0 = rf_data[base + n0];
    let s1 = rf_data[base + n0 + 1u];
    return mix(s0, s1, alpha);
}

@compute @workgroup_size(8, 8, 8)
fn dynamic_focus_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = params.volume_dims.xyz;
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let idxf = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z));
    let dimsf = vec3<f32>(f32(dims.x), f32(dims.y), f32(dims.z));
    let pv = (idxf - (dimsf - vec3<f32>(1.0)) * 0.5) * params.voxel_spacing.xyz;

    let channels = params.num_elements.x * params.num_elements.y * params.num_elements.z;
    let c = params.sound_speed;
    let fs = params.sampling_freq;

    // Depth-zone selection (focus(zone) lies on the z axis, depth = pv.z).
    var zone = 0u;
    if (params.num_focus_zones > 1u) {
        let span = max(params.max_depth - params.min_depth, 1e-9);
        let t = clamp((pv.z - params.min_depth) / span, 0.0, 1.0);
        zone = u32(round(t * f32(params.num_focus_zones - 1u)));
        if (zone >= params.num_focus_zones) {
            zone = params.num_focus_zones - 1u;
        }
    }
    let mask_words = (channels + 31u) / 32u;

    var acc = 0.0;
    for (var frame = 0u; frame < params.num_frames; frame = frame + 1u) {
        for (var ch = 0u; ch < channels; ch = ch + 1u) {
            if (params.enable_variable_aperture == 1u) {
                let word = aperture_masks[zone * mask_words + ch / 32u];
                if ((word & (1u << (ch % 32u))) == 0u) {
                    continue;
                }
            }
            let ep = vec3<f32>(
                element_positions[3u * ch],
                element_positions[3u * ch + 1u],
                element_positions[3u * ch + 2u],
            );
            let receive = length(pv - ep) / c * fs;
            let transmit = focus_delays[zone * channels + ch];
            acc = acc + apodization[ch] * rf_interp(frame, ch, channels, transmit + receive);
        }
    }

    let denom = max(f32(params.num_frames), 1.0);
    let out_idx = gid.x + gid.y * dims.x + gid.z * dims.x * dims.y;
    output[out_idx] = acc / denom;
}
