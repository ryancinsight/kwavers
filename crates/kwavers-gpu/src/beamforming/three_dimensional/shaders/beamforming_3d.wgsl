// 3D Delay-and-Sum (DAS) beamformer — static (plane-wave transmit) path.
//
// GPU port of `cpu::das::delay_and_sum_cpu`. The differential test
// `three_dimensional::tests` asserts this shader and the CPU reference agree to
// an epsilon bound; keep the two algorithms identical.
//
//   I_DAS(r_v) = (1/N_f) Σ_f Σ_i w_i · x_i^f(τ_i · f_s),  τ_i = |r_v − r_i| / c
//
// Buffer layout (must match `processor.rs` / `dispatch.rs`):
//   rf_data           : f32[ (frame*channels + ch) * num_samples + sample ]
//   output            : f32[ x + y*VX + z*VX*VY ]
//   element_positions : f32[ 3*ch + {0,1,2} ]  (centred element coordinates, m)
//   apodization       : f32[ ch ]              (per-element weight)

struct Params {
    // vec4 (not vec3) so each block is exactly 16 bytes, matching the host
    // struct's `[T; 3] + _padding` layout (WGSL packs vec3 as 12 bytes, which
    // would shift every following scalar by 4 and misread num_frames). The `.w`
    // lane is the host `_paddingN`.
    volume_dims: vec4<u32>,
    voxel_spacing: vec4<f32>,
    num_elements: vec4<u32>,
    element_spacing: vec4<f32>,
    sound_speed: f32,
    sampling_freq: f32,
    center_freq: f32,
    pad5: f32,
    num_frames: u32,
    num_samples: u32,
    dynamic_focusing: u32,
    apodization_window: u32,
}

@group(0) @binding(0) var<storage, read> rf_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> apodization: array<f32>;
@group(0) @binding(4) var<storage, read> element_positions: array<f32>;

// Linearly-interpolated RF sample at fractional index `tau_s`; 0 outside the
// recorded window (no wrap-around), matching the CPU `rf_get` closure.
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
fn delay_and_sum_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = params.volume_dims.xyz;
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    // Voxel centre in physical coordinates (origin at array centre):
    //   p_v = (idx − (dims − 1)/2) · voxel_spacing
    let idxf = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z));
    let dimsf = vec3<f32>(f32(dims.x), f32(dims.y), f32(dims.z));
    let pv = (idxf - (dimsf - vec3<f32>(1.0)) * 0.5) * params.voxel_spacing.xyz;

    let channels = params.num_elements.x * params.num_elements.y * params.num_elements.z;
    let c = params.sound_speed;
    let fs = params.sampling_freq;

    var acc = 0.0;
    for (var frame = 0u; frame < params.num_frames; frame = frame + 1u) {
        for (var ch = 0u; ch < channels; ch = ch + 1u) {
            let ep = vec3<f32>(
                element_positions[3u * ch],
                element_positions[3u * ch + 1u],
                element_positions[3u * ch + 2u],
            );
            let tau_s = length(pv - ep) / c * fs;
            acc = acc + apodization[ch] * rf_interp(frame, ch, channels, tau_s);
        }
    }

    let denom = max(f32(params.num_frames), 1.0);
    let out_idx = gid.x + gid.y * dims.x + gid.z * dims.x * dims.y;
    output[out_idx] = acc / denom;
}
