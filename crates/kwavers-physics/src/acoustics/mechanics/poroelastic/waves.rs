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
pub enum PoroelasticWaveMode {
    /// Fast compressional wave (in-phase)
    FastP,
    /// Slow compressional wave (out-of-phase)
    SlowP,
    /// Shear wave
    Shear,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `WaveSpeeds` stores and returns the three modal speeds verbatim.
    ///
    /// Analytical: constructed values must be readable via public fields
    /// without any conversion or mutation.
    #[test]
    fn wave_speeds_stores_all_three_modal_speeds() {
        let ws = WaveSpeeds {
            fast_wave: 2500.0,
            slow_wave: 200.0,
            shear_wave: 1200.0,
        };
        assert_eq!(
            ws.fast_wave, 2500.0,
            "fast_wave must match constructed value"
        );
        assert_eq!(
            ws.slow_wave, 200.0,
            "slow_wave must match constructed value"
        );
        assert_eq!(
            ws.shear_wave, 1200.0,
            "shear_wave must match constructed value"
        );
    }

    /// Physical ordering: fast P-wave speed > slow P-wave speed (Biot theory).
    ///
    /// In Biot poroelastic media the fast (in-phase) mode is always faster
    /// than the slow (diffusion-dominated, out-of-phase) mode.
    #[test]
    fn wave_speeds_fast_greater_than_slow_for_physical_values() {
        let ws = WaveSpeeds {
            fast_wave: 2500.0,
            slow_wave: 200.0,
            shear_wave: 1200.0,
        };
        assert!(
            ws.fast_wave > ws.slow_wave,
            "fast_wave ({}) must exceed slow_wave ({}) in Biot media",
            ws.fast_wave,
            ws.slow_wave
        );
    }

    /// `PoroelasticWaveMode` derives `PartialEq` and `Eq`: variants compare equal to
    /// themselves and unequal to each other.
    #[test]
    fn wave_mode_equality_is_variant_identity() {
        assert_eq!(PoroelasticWaveMode::FastP, PoroelasticWaveMode::FastP);
        assert_eq!(PoroelasticWaveMode::SlowP, PoroelasticWaveMode::SlowP);
        assert_eq!(PoroelasticWaveMode::Shear, PoroelasticWaveMode::Shear);
        assert_ne!(PoroelasticWaveMode::FastP, PoroelasticWaveMode::SlowP);
        assert_ne!(PoroelasticWaveMode::FastP, PoroelasticWaveMode::Shear);
        assert_ne!(PoroelasticWaveMode::SlowP, PoroelasticWaveMode::Shear);
    }

    /// `PoroelasticWaveMode` derives `Clone` and `Copy`: a copied variant equals the original.
    #[test]
    fn wave_mode_copy_clone_preserves_variant() {
        let m = PoroelasticWaveMode::SlowP;
        let copied = m;
        let cloned = m.clone();
        assert_eq!(copied, m);
        assert_eq!(cloned, m);
    }
}
