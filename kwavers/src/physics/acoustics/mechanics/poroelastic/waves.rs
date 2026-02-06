//! Wave mode definitions for poroelastic media

/// Wave speeds in poroelastic media
#[derive(Debug, Clone)]
pub struct WaveSpeeds {
    /// Fast compressional wave (P1) speed (m/s)
    pub fast_wave: f64,
    /// Slow compressional wave (P2) speed (m/s)
    pub slow_wave: f64,
    /// Shear wave speed (m/s)
    pub shear_wave: f64,
}

/// Wave mode type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveMode {
    /// Fast compressional wave (in-phase)
    FastP,
    /// Slow compressional wave (out-of-phase)
    SlowP,
    /// Shear wave
    Shear,
}
