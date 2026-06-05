/// Per-channel scale factors used to map physical Pa to the
/// network's `[-1, 1]` output space.
#[derive(Debug, Clone, Copy)]
pub struct OutputScales {
    pub p_min_pa: f32,
    pub p_max_pa: f32,
    pub p_rms_pa: f32,
}

/// Per-axis spatial half-extents (m) used to map physical
/// coordinates to the network's `[-1, 1]` input space.
#[derive(Debug, Clone, Copy)]
pub struct CoordHalves {
    pub hx_m: f32,
    pub hy_m: f32,
    pub hz_m: f32,
}

/// `(f0_min, f0_max)` and `(pnp_min, pnp_max)` for input-side
/// parameter normalisation.
#[derive(Debug, Clone, Copy)]
pub struct ParamRanges {
    pub f0_hz: (f32, f32),
    pub pnp_pa: (f32, f32),
}

/// How [`super::sampler::KernelCubeSampler::batch`] selects voxels.
///
/// `Uniform` — every voxel equally likely. Simple but biases the
/// network toward predicting near-zero everywhere because most
/// voxels are far from focus and have `|p| ≈ 0`.
///
/// `ImportanceByMagnitude { exponent }` — voxel probability ∝
/// `|p|^exponent`. With `exponent = 1.0` the network sees each
/// voxel weighted by its envelope magnitude (so the focal peak is
/// sampled ~the same total amount as the far rim); `2.0` further
/// concentrates samples near the peak. Closes the focal-peak
/// underprediction observed under uniform sampling.
#[derive(Debug, Clone, Copy)]
pub enum SamplingMode {
    Uniform,
    ImportanceByMagnitude { exponent: f32 },
}

impl Default for SamplingMode {
    fn default() -> Self {
        SamplingMode::Uniform
    }
}
