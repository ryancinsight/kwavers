//! `PMLConfig` — PML absorbing boundary configuration.

/// PML configuration parameters.
///
/// Controls the absorbing boundary layer properties for elastic wave propagation.
///
/// ## Profile Formula
///
/// `σ(d) = σ_max * (d / L_pml)^n`
///
/// where d is distance into the PML, L_pml is its thickness, and n is the profile order.
#[derive(Debug, Clone)]
pub struct PMLConfig {
    /// Thickness of PML region in grid points.
    pub thickness: usize,

    /// Maximum attenuation coefficient (Np/m).
    ///
    /// Typical values: 50–200 Np/m for ultrasound applications.
    pub sigma_max: f64,

    /// Power-law exponent for attenuation profile.
    ///
    /// Standard value: 2 (quadratic). Higher values produce steeper absorption.
    pub profile_order: u32,

    /// Theoretical reflection coefficient target (dimensionless).
    ///
    /// Typical: 1e-5 to 1e-8 (−100 to −160 dB).
    pub reflection_target: f64,
}

impl Default for PMLConfig {
    /// Default PML configuration providing <−80 dB reflection for ultrasound.
    fn default() -> Self {
        Self {
            thickness: 10,
            sigma_max: 100.0,
            profile_order: 2,
            reflection_target: 1e-4,
        }
    }
}
