//! Hounsfield unit conversions for CT data
//!
//! Implements piecewise linear fits to experimental CT density data,
//! matching k-wave-python's `hounsfield2density` and `hounsfield2soundspeed`.
//!
//! References:
//! - Mast, T. D. (2000) "Empirical relationships between acoustic parameters
//!   in human soft tissues," Acoust. Res. Lett. Online, 1(2), 37-42.

/// Hounsfield unit conversions for CT data
#[derive(Debug)]
pub struct HounsfieldUnits;

impl HounsfieldUnits {
    /// Convert Hounsfield units to density (kg/m³).
    ///
    /// Uses piecewise linear fits to experimental data (k-wave compatible).
    ///
    /// Regions:
    /// - HU < 930:         ρ = 1.025793·HU − 5.680404
    /// - 930 ≤ HU ≤ 1098:  ρ = 0.908271·HU + 103.615
    /// - 1098 < HU < 1260: ρ = 0.510837·HU + 539.998
    /// - HU ≥ 1260:        ρ = 0.662537·HU + 348.856
    #[must_use]
    pub fn to_density(hu: f64) -> f64 {
        if hu < 930.0 {
            1.025793065681423 * hu + (-5.680404011488714)
        } else if hu <= 1098.0 {
            0.9082709691264 * hu + 103.6151457847139
        } else if hu < 1260.0 {
            0.5108369316599 * hu + 539.9977189228704
        } else {
            0.6625370912451 * hu + 348.8555178455294
        }
    }

    /// Convert density to Hounsfield units (approximate inverse).
    ///
    /// Uses the soft-tissue region (930 ≤ HU ≤ 1098) inverse for the
    /// typical clinical range.  For extreme values this is approximate.
    #[must_use]
    pub fn from_density(density: f64) -> f64 {
        // Inverse of the primary soft-tissue region
        (density - 103.6151457847139) / 0.9082709691264
    }

    /// Convert Hounsfield units to sound speed (m/s).
    ///
    /// Uses the Mast (2000) empirical relationship:
    ///   c = (ρ(HU) + 349) / 0.893
    ///
    /// where ρ(HU) is computed from [`to_density`].
    /// Matches k-wave-python's `hounsfield2soundspeed`.
    #[must_use]
    pub fn to_sound_speed(hu: f64) -> f64 {
        (Self::to_density(hu) + 349.0) / 0.893
    }

    /// Convert Hounsfield units to acoustic impedance
    #[must_use]
    pub fn to_impedance(hu: f64) -> f64 {
        let density = Self::to_density(hu);
        let sound_speed = Self::to_sound_speed(hu);
        density * sound_speed
    }

    /// Get typical tissue properties from HU value
    #[must_use]
    pub fn classify_tissue(hu: f64) -> &'static str {
        match hu {
            h if h < -1000.0 => "Air",
            h if h < -100.0 => "Fat",
            h if h < -10.0 => "Water",
            h if h < 40.0 => "Soft Tissue",
            h if h < 100.0 => "Muscle",
            h if h < 300.0 => "Liver",
            h if h < 700.0 => "Trabecular Bone",
            _ => "Cortical Bone",
        }
    }
}
