/// Canonical GPU workspace descriptor for photoacoustic acoustic propagation.
#[derive(Debug, Clone, Default)]
pub struct AcousticGpuWorkspace {
    pub spectrum_len: usize,
}

/// Report whether the canonical FFT-backed GPU dependency is available.
#[must_use]
pub fn gpu_acoustic_available() -> bool {
    kwavers_math::fft::gpu_fft_available()
}
