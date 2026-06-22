//! Transducer Materials Module
//!
//! Defines piezoelectric materials, backing layers, matching layers,
//! and acoustic lenses used in transducer construction.

use super::{PZT_DIELECTRIC_CONSTANT, PZT_SOUND_SPEED};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// Piezoelectric material properties
///
/// Based on material data from:
/// - Berlincourt et al. (1964): "Piezoelectric and Piezomagnetic Materials"
/// - IEEE Standard 176-1987: "IEEE Standard on Piezoelectricity"
#[derive(Debug, Clone)]
pub struct PiezoMaterial {
    /// Material type
    pub material_type: PiezoType,
    /// Thickness mode coupling coefficient (k33)
    pub coupling_k33: f64,
    /// Lateral mode coupling coefficient (k31)
    pub coupling_k31: f64,
    /// Mechanical quality factor
    pub mechanical_q: f64,
    /// Dielectric constant (relative)
    pub dielectric_constant: f64,
    /// Density (kg/m³)
    pub density: f64,
    /// Speed of sound (m/s)
    pub sound_speed: f64,
    /// Acoustic impedance (`MRayl`)
    pub acoustic_impedance: f64,
    /// Curie temperature (°C)
    pub curie_temperature: f64,
}

/// Common piezoelectric material types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PiezoType {
    /// Lead Zirconate Titanate (Navy Type I)
    PZT4,
    /// Lead Zirconate Titanate (Navy Type VI)
    PZT5H,
    /// Lead Zirconate Titanate (Navy Type II)
    PZT5A,
    /// Lead Magnesium Niobate - Lead Titanate
    PMNPT,
    /// Polyvinylidene Fluoride
    PVDF,
    /// Lead-free Bismuth Sodium Titanate
    BNT,
    /// Custom material
    Custom,
}

impl PiezoMaterial {
    /// Create PZT-5H material (most common for medical transducers)
    #[must_use]
    pub fn pzt_5h() -> Self {
        Self {
            material_type: PiezoType::PZT5H,
            coupling_k33: 0.75,
            coupling_k31: 0.39,
            mechanical_q: 65.0,
            dielectric_constant: PZT_DIELECTRIC_CONSTANT,
            density: 7500.0,
            sound_speed: PZT_SOUND_SPEED,
            acoustic_impedance: 34.5,
            curie_temperature: 193.0,
        }
    }

    /// Create PZT-4 material (higher Q, lower coupling)
    #[must_use]
    pub fn pzt_4() -> Self {
        Self {
            material_type: PiezoType::PZT4,
            coupling_k33: 0.70,
            coupling_k31: 0.33,
            mechanical_q: 500.0,
            dielectric_constant: 1300.0,
            density: 7500.0,
            sound_speed: PZT_SOUND_SPEED,
            acoustic_impedance: 34.5,
            curie_temperature: 328.0,
        }
    }

    /// Create PMN-PT single crystal (highest coupling)
    #[must_use]
    pub fn pmn_pt() -> Self {
        Self {
            material_type: PiezoType::PMNPT,
            coupling_k33: 0.90,
            coupling_k31: 0.45,
            mechanical_q: 100.0,
            dielectric_constant: 5000.0,
            density: 8100.0,
            sound_speed: 4500.0,
            acoustic_impedance: 36.5,
            curie_temperature: 130.0,
        }
    }

    /// Create PVDF polymer (flexible, broadband)
    #[must_use]
    pub fn pvdf() -> Self {
        Self {
            material_type: PiezoType::PVDF,
            coupling_k33: 0.20,
            coupling_k31: 0.12,
            mechanical_q: 10.0,
            dielectric_constant: 12.0,
            density: 1780.0,
            sound_speed: 2200.0,
            acoustic_impedance: 3.9,
            curie_temperature: 100.0,
        }
    }

    /// Calculate electromechanical coupling factor
    #[must_use]
    pub fn effective_coupling(&self) -> f64 {
        self.coupling_k33.powi(2)
    }

    /// Calculate bandwidth based on coupling and Q
    ///
    /// Fractional bandwidth ≈ k² / Q^0.5
    #[must_use]
    pub fn bandwidth_estimate(&self) -> f64 {
        self.effective_coupling() / self.mechanical_q.sqrt() * 100.0
    }
}

/// Backing layer for damping and bandwidth control
#[derive(Debug, Clone)]
pub struct BackingLayer {
    /// Backing material type
    pub material: BackingMaterial,
    /// Acoustic impedance (`MRayl`)
    pub acoustic_impedance: f64,
    /// Attenuation coefficient (dB/mm at 1 `MHz`)
    pub attenuation: f64,
    /// Thickness (m)
    pub thickness: f64,
}

/// Common backing materials
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackingMaterial {
    /// Tungsten-loaded epoxy
    TungstenEpoxy,
    /// Air backing (undamped)
    Air,
    /// Custom composite
    Custom,
}

impl BackingLayer {
    /// Create tungsten-epoxy backing (standard for broadband)
    #[must_use]
    pub fn tungsten_epoxy(thickness: f64) -> Self {
        Self {
            material: BackingMaterial::TungstenEpoxy,
            acoustic_impedance: 5.0,
            attenuation: 5.0,
            thickness,
        }
    }

    /// Create air backing (narrow band, high sensitivity)
    #[must_use]
    pub fn air_backed() -> Self {
        Self {
            material: BackingMaterial::Air,
            acoustic_impedance: 0.0004,
            attenuation: 0.0,
            thickness: 0.0,
        }
    }

    /// Calculate reflection coefficient at piezo-backing interface
    #[must_use]
    pub fn reflection_coefficient(&self, piezo_impedance: f64) -> f64 {
        // From the piezo backing (incident) into the matching/acoustic layer.
        kwavers_medium::properties::reflection_coefficient(piezo_impedance, self.acoustic_impedance)
    }
}

/// Matching layer for impedance matching
#[derive(Debug, Clone)]
pub struct MatchingLayer {
    /// Acoustic impedance (`MRayl`)
    pub acoustic_impedance: f64,
    /// Thickness (m)
    pub thickness: f64,
    /// Number of layers
    pub num_layers: usize,
}

impl MatchingLayer {
    /// Design quarter-wave matching layer
    ///
    /// Optimal impedance: `Z_match` = `sqrt(Z_piezo` * `Z_medium`)
    #[must_use]
    pub fn quarter_wave(frequency: f64, piezo_impedance: f64, medium_impedance: f64) -> Self {
        let optimal_impedance = (piezo_impedance * medium_impedance).sqrt();
        let sound_speed = 2500.0; // Typical for matching layer materials
        let wavelength = sound_speed / frequency;
        let thickness = wavelength / 4.0;

        Self {
            acoustic_impedance: optimal_impedance,
            thickness,
            num_layers: 1,
        }
    }

    /// Design multi-layer matching for broader bandwidth
    ///
    /// Uses binomial transformer design
    #[must_use]
    pub fn multi_layer(
        frequency: f64,
        piezo_impedance: f64,
        medium_impedance: f64,
        num_layers: usize,
    ) -> Vec<Self> {
        let mut layers = Vec::new();
        let impedance_ratio = (medium_impedance / piezo_impedance).ln();

        for i in 1..=num_layers {
            let fraction = i as f64 / (num_layers + 1) as f64;
            let layer_impedance = piezo_impedance * (fraction * impedance_ratio).exp();
            let sound_speed = 500.0f64.mul_add(fraction, 2500.0); // Varies with material
            let wavelength = sound_speed / frequency;
            let thickness = wavelength / 4.0;

            layers.push(Self {
                acoustic_impedance: layer_impedance,
                thickness,
                num_layers: 1,
            });
        }

        layers
    }

    /// Calculate power transmission coefficient
    /// Single quarter-wave matching layer: T = 4Z₁Z₃/(Z₁+Z₃)²
    /// For optimal matching: Z₂ = √(Z₁Z₃) per Kinsler et al. (2000) §10.3
    #[must_use]
    pub fn transmission_coefficient(&self, piezo_impedance: f64, medium_impedance: f64) -> f64 {
        // Quarter-wave layer transmission (reflections cancel at design frequency)
        let _r1 = (self.acoustic_impedance - piezo_impedance)
            / (self.acoustic_impedance + piezo_impedance);
        let _r2 = (medium_impedance - self.acoustic_impedance)
            / (medium_impedance + self.acoustic_impedance);

        let numerator = 4.0 * piezo_impedance * medium_impedance;
        let denominator = (piezo_impedance + medium_impedance).powi(2);
        numerator / denominator
    }
}

/// Acoustic lens for beam focusing
#[derive(Debug, Clone)]
pub struct AcousticLens {
    /// Lens material
    pub material: LensMaterial,
    /// Radius of curvature (m)
    pub radius_of_curvature: f64,
    /// Lens thickness at center (m)
    pub center_thickness: f64,
    /// Speed of sound in lens (m/s)
    pub sound_speed: f64,
    /// Acoustic impedance (`MRayl`)
    pub acoustic_impedance: f64,
    /// Attenuation (dB/mm at 1 `MHz`)
    pub attenuation: f64,
}

/// Common lens materials
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LensMaterial {
    /// Silicone rubber (RTV)
    Silicone,
    /// Polyurethane
    Polyurethane,
    /// Custom material
    Custom,
}

impl AcousticLens {
    /// Create silicone lens (standard for medical transducers)
    #[must_use]
    pub fn silicone(focal_length: f64, aperture: f64) -> Self {
        let sound_speed_lens = 1000.0; // m/s in silicone
        let sound_speed_tissue = SOUND_SPEED_TISSUE; // m/s in tissue

        // Calculate radius of curvature using lens equation
        let radius = focal_length * (sound_speed_tissue - sound_speed_lens) / sound_speed_tissue;

        // Calculate center thickness
        let sagitta = aperture.powi(2) / (8.0 * radius.abs());
        let center_thickness = sagitta + 0.5e-3; // Add minimum thickness

        Self {
            material: LensMaterial::Silicone,
            radius_of_curvature: radius,
            center_thickness,
            sound_speed: sound_speed_lens,
            acoustic_impedance: 1.0,
            attenuation: 1.0,
        }
    }

    /// Calculate focal length in the medium
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn focal_length(&self, medium_sound_speed: f64) -> f64 {
        let _speed_ratio = medium_sound_speed / self.sound_speed;
        self.radius_of_curvature * medium_sound_speed / (medium_sound_speed - self.sound_speed)
    }

    /// Calculate f-number (focal length / aperture)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn f_number(&self, aperture: f64, medium_sound_speed: f64) -> f64 {
        self.focal_length(medium_sound_speed) / aperture
    }

    /// Focusing **delay profile** the lens imposes across its aperture.
    ///
    /// A passive refractive lens of focal length `F` (see [`Self::focal_length`])
    /// curves the wavefront exactly like the phased-array delay law (Sources &
    /// Transducers §6.5): the geometric focusing delay from aperture radius `r`
    /// to the focus, relative to the lens centre, is
    ///
    /// ```text
    /// τ(r) = (√(F² + r²) − F) / c_medium       [s],
    /// ```
    ///
    /// which is `0` at the centre, monotone-increasing in `r`, and reduces to the
    /// paraxial `r² / (2 c_medium F)` for `r ≪ F`. This is the static-lens
    /// analogue of the per-element delay `τ_i` of a phased array — a fixed lens is
    /// a passive, non-steerable delay law. `F = |focal_length|` so the magnitude is
    /// returned for both converging and diverging designs.
    ///
    /// Returns one delay per supplied aperture radius.
    #[must_use]
    pub fn aperture_delay_profile(&self, radii_m: &[f64], medium_sound_speed: f64) -> Vec<f64> {
        let f = self.focal_length(medium_sound_speed).abs();
        let inv_c = 1.0 / medium_sound_speed;
        radii_m
            .iter()
            .map(|&r| (f.mul_add(f, r * r).sqrt() - f) * inv_c)
            .collect()
    }

    /// Validate lens design
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.center_thickness <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "center_thickness".to_owned(),
                value: self.center_thickness.to_string(),
                constraint: "Lens thickness must be positive".to_owned(),
            }));
        }

        if self.radius_of_curvature == 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius_of_curvature".to_owned(),
                value: "0".to_owned(),
                constraint: "Radius of curvature cannot be zero".to_owned(),
            }));
        }

        Ok(())
    }
}

#[cfg(test)]
mod lens_tests {
    use super::*;

    fn custom_lens(radius_m: f64, c_lens: f64) -> AcousticLens {
        AcousticLens {
            material: LensMaterial::Custom,
            radius_of_curvature: radius_m,
            center_thickness: 1.0e-3,
            sound_speed: c_lens,
            acoustic_impedance: 1.0,
            attenuation: 1.0,
        }
    }

    #[test]
    fn lensmaker_focal_length_formula() {
        // Plano-concave slow lens (c_lens = 1000) in c_medium = 1500, R = 10 mm.
        // Lensmaker (single refractive surface): F = R·c_m/(c_m − c_lens).
        let lens = custom_lens(10.0e-3, 1000.0);
        let f = lens.focal_length(1500.0);
        let expected = 10.0e-3 * 1500.0 / (1500.0 - 1000.0); // = 30 mm
        assert!((f - expected).abs() < 1e-12, "F {f} != {expected}");
        assert!((f - 0.030).abs() < 1e-9, "expected 30 mm focus");
    }

    #[test]
    fn silicone_constructor_round_trips_focal_length_and_f_number() {
        let (f_design, aperture) = (50.0e-3, 20.0e-3);
        let lens = AcousticLens::silicone(f_design, aperture);
        let f = lens.focal_length(SOUND_SPEED_TISSUE);
        // silicone() sets R = F(c_t − c_l)/c_t, which focal_length inverts back to F.
        assert!(
            (f - f_design).abs() < 1e-9,
            "focal length {f} != design {f_design}"
        );
        assert!(
            (lens.f_number(aperture, SOUND_SPEED_TISSUE) - f_design / aperture).abs() < 1e-9,
            "f-number must equal F/aperture"
        );
    }

    #[test]
    fn aperture_delay_profile_is_zero_centred_monotone_and_paraxial() {
        let (f_design, c_m) = (50.0e-3, SOUND_SPEED_TISSUE);
        let lens = AcousticLens::silicone(f_design, 20.0e-3);
        let radii: Vec<f64> = (0..6).map(|i| i as f64 * 1.0e-3).collect(); // 0..5 mm
        let tau = lens.aperture_delay_profile(&radii, c_m);

        assert!(tau[0].abs() < 1e-18, "τ(0) must be 0");
        for w in tau.windows(2) {
            assert!(w[1] > w[0], "τ(r) must strictly increase with r");
        }
        // Paraxial limit τ(r) ≈ r²/(2 c F): relative error ~ r²/(4F²) ≈ 0.25% at r=5mm.
        let r = 5.0e-3;
        let exact = lens.aperture_delay_profile(&[r], c_m)[0];
        let paraxial = r * r / (2.0 * c_m * f_design);
        assert!(
            (exact - paraxial).abs() / paraxial < 0.02,
            "τ {exact} vs paraxial {paraxial}"
        );
    }

    #[test]
    fn validate_rejects_zero_radius() {
        assert!(custom_lens(0.0, 1000.0).validate().is_err());
    }
}
