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
            1.025793065681423f64.mul_add(hu, -5.680404011488714)
        } else if hu <= 1098.0 {
            0.9082709691264f64.mul_add(hu, 103.6151457847139)
        } else if hu < 1260.0 {
            0.5108369316599f64.mul_add(hu, 539.9977189228704)
        } else {
            0.6625370912451f64.mul_add(hu, 348.8555178455294)
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
    /// where ρ(HU) is computed from [`Self::to_density`].
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

#[cfg(test)]
mod tests {
    use super::HounsfieldUnits;

    // The k-wave-python `hounsfield2density` fit uses the CT-number convention
    // (water ≈ 1000 HU). The four piecewise segments must join continuously
    // (C⁰) at the breakpoints HU = 930, 1098, 1260 — otherwise a CT volume
    // would produce density discontinuities the segmentation never intended.
    #[test]
    fn density_is_continuous_at_segment_breakpoints() {
        for &bp in &[930.0_f64, 1098.0, 1260.0] {
            // Approach the breakpoint from below and above; the fit is C⁰, so
            // the one-sided limits must agree to within rounding of the linear
            // evaluation (≪ 1 kg/m³).
            let below = HounsfieldUnits::to_density(bp - 1e-6);
            let above = HounsfieldUnits::to_density(bp + 1e-6);
            assert!(
                (below - above).abs() < 1e-3,
                "density discontinuity at HU={bp}: {below} vs {above}"
            );
        }
    }

    // Water (HU ≈ 1000 in the CT-number convention) and cortical bone (HU ≈
    // 1300) must land in their physiological density bands — an independent
    // physical check, not a restatement of the fit coefficients.
    #[test]
    fn density_lands_in_physiological_bands() {
        let rho_water = HounsfieldUnits::to_density(1000.0);
        assert!(
            (1000.0..=1025.0).contains(&rho_water),
            "water density {rho_water} kg/m³ outside [1000, 1025]"
        );
        let rho_bone = HounsfieldUnits::to_density(1300.0);
        assert!(
            (1150.0..=1300.0).contains(&rho_bone),
            "cortical-bone density {rho_bone} kg/m³ outside [1150, 1300]"
        );
    }

    // Density is strictly increasing in HU across the full clinical range:
    // denser tissue ⇒ higher CT attenuation ⇒ higher HU.
    #[test]
    fn density_is_strictly_increasing() {
        let mut prev = HounsfieldUnits::to_density(0.0);
        for hu_centi in 1..=400 {
            let hu = hu_centi as f64 * 5.0; // 5 … 2000 HU
            let rho = HounsfieldUnits::to_density(hu);
            assert!(
                rho > prev,
                "density not increasing at HU={hu}: {rho} ≤ {prev}"
            );
            prev = rho;
        }
    }

    // Sound speed is the Mast (2000) affine image of density, c = (ρ+349)/0.893,
    // and the round-trip density↔HU inverse recovers the soft-tissue segment.
    #[test]
    fn sound_speed_and_inverse_are_consistent() {
        let hu = 1000.0_f64;
        let rho = HounsfieldUnits::to_density(hu);
        let c = HounsfieldUnits::to_sound_speed(hu);
        assert!((c - (rho + 349.0) / 0.893).abs() < 1e-9);
        assert!(
            (1480.0..=1560.0).contains(&c),
            "water speed {c} m/s out of band"
        );
        // from_density inverts the soft-tissue segment (930 ≤ HU ≤ 1098).
        assert!((HounsfieldUnits::from_density(rho) - hu).abs() < 1e-6);
        // Impedance identity.
        assert!((HounsfieldUnits::to_impedance(hu) - rho * c).abs() < 1e-6);
    }

    #[test]
    fn tissue_classification_boundaries() {
        assert_eq!(HounsfieldUnits::classify_tissue(-1500.0), "Air");
        assert_eq!(HounsfieldUnits::classify_tissue(0.0), "Soft Tissue");
        assert_eq!(HounsfieldUnits::classify_tissue(1000.0), "Cortical Bone");
    }
}
