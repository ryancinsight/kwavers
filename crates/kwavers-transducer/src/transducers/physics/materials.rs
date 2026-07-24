//! Transducer Materials Module
//!
//! Defines piezoelectric materials, backing layers, matching layers,
//! and acoustic lenses used in transducer construction.

use super::{PZT_DIELECTRIC_CONSTANT, PZT_SOUND_SPEED};
use aequitas::systems::si::{
    quantities::{
        AcousticImpedance, Frequency, Length, MassDensity, ThermodynamicTemperature, Time, Velocity,
    },
    units::Kelvin,
};
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
    pub density: MassDensity,
    /// Speed of sound (m/s)
    pub sound_speed: Velocity,
    /// Acoustic impedance (Rayl)
    pub acoustic_impedance: AcousticImpedance,
    /// Curie temperature (K)
    pub curie_temperature: ThermodynamicTemperature,
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
            density: MassDensity::from_base(7500.0),
            sound_speed: Velocity::from_base(PZT_SOUND_SPEED),
            acoustic_impedance: AcousticImpedance::from_base(34.5e6),
            curie_temperature: ThermodynamicTemperature::from_unit::<Kelvin>(466.15),
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
            density: MassDensity::from_base(7500.0),
            sound_speed: Velocity::from_base(PZT_SOUND_SPEED),
            acoustic_impedance: AcousticImpedance::from_base(34.5e6),
            curie_temperature: ThermodynamicTemperature::from_unit::<Kelvin>(601.15),
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
            density: MassDensity::from_base(8100.0),
            sound_speed: Velocity::from_base(4500.0),
            acoustic_impedance: AcousticImpedance::from_base(36.5e6),
            curie_temperature: ThermodynamicTemperature::from_unit::<Kelvin>(403.15),
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
            density: MassDensity::from_base(1780.0),
            sound_speed: Velocity::from_base(2200.0),
            acoustic_impedance: AcousticImpedance::from_base(3.9e6),
            curie_temperature: ThermodynamicTemperature::from_unit::<Kelvin>(373.15),
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
    /// Acoustic impedance (Rayl)
    pub acoustic_impedance: AcousticImpedance,
    /// Attenuation coefficient (dB/mm at 1 `MHz`)
    pub attenuation: f64,
    /// Thickness (m)
    pub thickness: Length,
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
            acoustic_impedance: AcousticImpedance::from_base(5.0e6),
            attenuation: 5.0,
            thickness: Length::from_base(thickness),
        }
    }

    /// Create air backing (narrow band, high sensitivity)
    #[must_use]
    pub fn air_backed() -> Self {
        Self {
            material: BackingMaterial::Air,
            acoustic_impedance: AcousticImpedance::from_base(400.0),
            attenuation: 0.0,
            thickness: Length::from_base(0.0),
        }
    }

    /// Calculate reflection coefficient at piezo-backing interface
    #[must_use]
    pub fn reflection_coefficient(&self, piezo_impedance: AcousticImpedance) -> f64 {
        // From the piezo backing (incident) into the matching/acoustic layer.
        kwavers_medium::properties::reflection_coefficient(
            piezo_impedance.into_base() * 1.0e-6,
            self.acoustic_impedance.into_base() * 1.0e-6,
        )
    }
}

/// Matching layer for impedance matching
#[derive(Debug, Clone)]
pub struct MatchingLayer {
    /// Acoustic impedance (Rayl)
    pub acoustic_impedance: AcousticImpedance,
    /// Thickness (m)
    pub thickness: Length,
    /// Number of layers
    pub num_layers: usize,
}

impl MatchingLayer {
    /// Design quarter-wave matching layer
    ///
    /// Optimal impedance: `Z_match` = `sqrt(Z_piezo` * `Z_medium`)
    #[must_use]
    pub fn quarter_wave(
        frequency: Frequency,
        piezo_impedance: AcousticImpedance,
        medium_impedance: AcousticImpedance,
    ) -> Self {
        let optimal_impedance = AcousticImpedance::from_base(
            (piezo_impedance.into_base() * medium_impedance.into_base()).sqrt(),
        );
        let sound_speed = 2500.0; // Typical for matching layer materials
        let thickness = Length::from_base(sound_speed / (4.0 * frequency.into_base()));

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
        frequency: Frequency,
        piezo_impedance: AcousticImpedance,
        medium_impedance: AcousticImpedance,
        num_layers: usize,
    ) -> Vec<Self> {
        let mut layers = Vec::new();
        let impedance_ratio = (medium_impedance.into_base() / piezo_impedance.into_base()).ln();

        for i in 1..=num_layers {
            let fraction = i as f64 / (num_layers + 1) as f64;
            let layer_impedance = AcousticImpedance::from_base(
                piezo_impedance.into_base() * (fraction * impedance_ratio).exp(),
            );
            let sound_speed = 500.0f64.mul_add(fraction, 2500.0); // Varies with material
            let thickness = Length::from_base(sound_speed / (4.0 * frequency.into_base()));

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
    pub fn transmission_coefficient(
        &self,
        piezo_impedance: AcousticImpedance,
        medium_impedance: AcousticImpedance,
    ) -> f64 {
        let piezo_impedance = piezo_impedance.into_base();
        let medium_impedance = medium_impedance.into_base();
        let layer_impedance = self.acoustic_impedance.into_base();
        // Quarter-wave layer transmission (reflections cancel at design frequency)
        let _r1 = (layer_impedance - piezo_impedance) / (layer_impedance + piezo_impedance);
        let _r2 = (medium_impedance - layer_impedance) / (medium_impedance + layer_impedance);

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
    pub radius_of_curvature: Length,
    /// Lens thickness at center (m)
    pub center_thickness: Length,
    /// Speed of sound in lens (m/s)
    pub sound_speed: Velocity,
    /// Acoustic impedance (Rayl)
    pub acoustic_impedance: AcousticImpedance,
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
    pub fn silicone(focal_length: Length, aperture: Length) -> Self {
        let sound_speed_lens = 1000.0; // m/s in silicone
        let sound_speed_tissue = SOUND_SPEED_TISSUE; // m/s in tissue
        let focal_length_m = focal_length.into_base();
        let aperture_m = aperture.into_base();

        // Calculate radius of curvature using lens equation
        let radius = focal_length_m * (sound_speed_tissue - sound_speed_lens) / sound_speed_tissue;

        // Calculate center thickness
        let sagitta = aperture_m.powi(2) / (8.0 * radius.abs());
        let center_thickness = sagitta + 0.5e-3; // Add minimum thickness

        Self {
            material: LensMaterial::Silicone,
            radius_of_curvature: Length::from_base(radius),
            center_thickness: Length::from_base(center_thickness),
            sound_speed: Velocity::from_base(sound_speed_lens),
            acoustic_impedance: AcousticImpedance::from_base(1.0e6),
            attenuation: 1.0,
        }
    }

    /// Calculate focal length in the medium
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn focal_length(&self, medium_sound_speed: Velocity) -> Length {
        let medium_speed = medium_sound_speed.into_base();
        let lens_speed = self.sound_speed.into_base();
        Length::from_base(
            self.radius_of_curvature.into_base() * medium_speed / (medium_speed - lens_speed),
        )
    }

    /// Calculate f-number (focal length / aperture)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn f_number(&self, aperture: Length, medium_sound_speed: Velocity) -> f64 {
        self.focal_length(medium_sound_speed).into_base() / aperture.into_base()
    }

    /// Focusing **delay profile** the lens imposes across its aperture.
    ///
    /// A passive refractive lens of focal length `F` (see [`Self::focal_length`])
    /// curves the wavefront exactly like the phased-array delay law (Sources &
    /// Transducers §6.5): the geometric focusing delay from aperture radius `r`
    /// to the focus, relative to the lens centre, is
    ///
    /// ```text
    /// τ(r) = (√(F² + r²) − F) / c_medium       `s`,
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
    pub fn aperture_delay_profile(
        &self,
        radii: &[Length],
        medium_sound_speed: Velocity,
    ) -> Vec<Time> {
        let f = self.focal_length(medium_sound_speed).into_base().abs();
        let inv_c = 1.0 / medium_sound_speed.into_base();
        radii
            .iter()
            .map(|radius| {
                Time::from_base((f.mul_add(f, radius.into_base().powi(2)).sqrt() - f) * inv_c)
            })
            .collect()
    }

    /// Validate lens design
    /// # Errors
    /// - Returns `KwaversError::Config` if the precondition for a Config-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.center_thickness.into_base() <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "center_thickness".to_owned(),
                value: self.center_thickness.into_base().to_string(),
                constraint: "Lens thickness must be positive".to_owned(),
            }));
        }

        if self.radius_of_curvature.into_base() == 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius_of_curvature".to_owned(),
                value: "0".to_owned(),
                constraint: "Radius of curvature cannot be zero".to_owned(),
            }));
        }

        Ok(())
    }
}

/// Acoustic **Fresnel zone plate**: focuses by *diffraction* from concentric
/// zones rather than refraction through bulk material.
///
/// The zone boundaries sit at radii where the path from the aperture to the
/// focus grows by half a wavelength, so the central and even/odd zones interfere
/// constructively at `F`. A Soret (amplitude) plate blocks alternate zones; a
/// phase-reversal plate inverts them for higher efficiency. The plate is thin and
/// flat (no lens sagitta), at the cost of secondary foci at `F/3, F/5, …` and
/// reduced efficiency. Conventions follow Hecht, *Optics*, §10.3.5 (the acoustic
/// case is identical with `λ = c/f`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FresnelZonePlate {
    /// Primary focal length `F` \`m`.
    pub focal_length: Length,
    /// Design wavelength `λ = c/f` \`m`.
    pub wavelength: Length,
    /// Outer aperture radius \`m`.
    pub aperture_radius: Length,
}

impl FresnelZonePlate {
    /// Create a zone plate for a target focal length, wavelength, and aperture.
    #[must_use]
    pub fn new(focal_length: Length, wavelength: Length, aperture_radius: Length) -> Self {
        Self {
            focal_length,
            wavelength,
            aperture_radius,
        }
    }

    /// `n`-th zone boundary radius (exact, not paraxial):
    ///
    /// ```text
    /// r_n = √(n λ F + (n λ / 2)²),
    /// ```
    ///
    /// from the half-wave path condition `√(r_n² + F²) − F = n λ / 2`. Reduces to
    /// the familiar `√(n λ F)` when `F ≫ n λ`.
    #[must_use]
    pub fn zone_radius(&self, n: usize) -> Length {
        let nl = n as f64 * self.wavelength.into_base();
        Length::from_base(
            nl.mul_add(self.focal_length.into_base(), (0.5 * nl).powi(2))
                .sqrt(),
        )
    }

    /// All zone-boundary radii that fall within the aperture (`r_n ≤ a`),
    /// in increasing order.
    #[must_use]
    pub fn zone_radii(&self) -> Vec<Length> {
        let mut radii = Vec::new();
        let mut n = 1usize;
        loop {
            let r = self.zone_radius(n);
            if r > self.aperture_radius || !r.into_base().is_finite() {
                break;
            }
            radii.push(r);
            n += 1;
        }
        radii
    }

    /// Number of full Fresnel zones inside the aperture.
    #[must_use]
    pub fn num_zones(&self) -> usize {
        self.zone_radii().len()
    }

    /// f-number `F / D` of the primary focus (`D = 2·aperture_radius`).
    #[must_use]
    pub fn f_number(&self) -> f64 {
        self.focal_length.into_base() / (2.0 * self.aperture_radius.into_base())
    }
}

/// **Corrective acoustic-lens thickness** from a per-point aberration phase
/// (Maimbourg et al. 2020, *IEEE TBME*, Eq. 1).
///
/// A single-element transducer is made target/skull-specific by casting a lens
/// whose local thickness `p(M)` imposes the correction phase `φ̃(M)` (the
/// unwrapped skull-aberration phase, e.g. from
/// `kwavers_physics::…::aberration_correction::phase_correction`). A slab of
/// thickness `p` and speed `c_lens` replacing the coupling medium `c_water`
/// advances the phase by `Δφ = 2π f₀ p (1/c_water − 1/c_lens)`, so inverting,
///
/// ```text
/// p(M) = φ̃(M) / (2π f₀) · 1 / (1/c_water − 1/c_lens) + K,
/// ```
///
/// where `K` (here `min_thickness_m`) sets the *minimum* lens thickness so the
/// whole profile is castable (`p ≥ min_thickness_m`). Returns one thickness per
/// supplied phase sample; the relative profile is the physics, the constant
/// offset is fabrication headroom.
///
/// # Panics
/// Never; an empty `phase_rad` yields an empty vector.
#[must_use]
pub fn corrective_lens_thickness(
    phase_rad: &[f64],
    frequency: Frequency,
    c_water: Velocity,
    c_lens: Velocity,
    min_thickness: Length,
) -> Vec<Length> {
    let frequency_hz = frequency.into_base();
    let c_water = c_water.into_base();
    let c_lens = c_lens.into_base();
    let min_thickness_m = min_thickness.into_base();
    let denom = 2.0 * std::f64::consts::PI * frequency_hz * (c_water.recip() - c_lens.recip());
    if denom == 0.0 || phase_rad.is_empty() {
        return vec![Length::from_base(min_thickness_m); phase_rad.len()];
    }
    let raw: Vec<f64> = phase_rad.iter().map(|&phi| phi / denom).collect();
    let min_raw = raw.iter().copied().fold(f64::INFINITY, f64::min);
    // Offset so the thinnest point equals the minimal castable thickness.
    raw.iter()
        .map(|&p| Length::from_base(p - min_raw + min_thickness_m))
        .collect()
}

/// **Isoplanatic mechanical-steering pose** for a lens-coupled single-element
/// transducer (Maimbourg et al. 2020, Eq. 2).
///
/// A lens corrected for the on-axis target steers the focus to a transverse
/// offset `x` by *mechanically* rotating the transducer/lens pair by `θ_y` and
/// translating it along the beam axis by `T_z`, exploiting the skull's
/// isoplanatic angle (nearby targets share the same aberration):
///
/// ```text
/// θ_y = arcsin(x / F),    T_z = F − √(F² − x²),
/// ```
///
/// for focal length `F`. Returns `(θ_y `rad`, T_z `m`)`, or `None` for the
/// unphysical `|x| > F`.
#[must_use]
pub fn isoplanatic_steering_pose(x_offset: Length, focal_length: Length) -> Option<(f64, Length)> {
    let x_offset_m = x_offset.into_base();
    let focal_length_m = focal_length.into_base();
    if focal_length_m <= 0.0 || x_offset_m.abs() > focal_length_m {
        return None;
    }
    let theta_y = (x_offset_m / focal_length_m).asin();
    let t_z = focal_length_m - (focal_length_m * focal_length_m - x_offset_m * x_offset_m).sqrt();
    Some((theta_y, Length::from_base(t_z)))
}

#[cfg(test)]
mod lens_tests {
    use super::*;

    fn custom_lens(radius_m: f64, c_lens: f64) -> AcousticLens {
        AcousticLens {
            material: LensMaterial::Custom,
            radius_of_curvature: Length::from_base(radius_m),
            center_thickness: Length::from_base(1.0e-3),
            sound_speed: Velocity::from_base(c_lens),
            acoustic_impedance: AcousticImpedance::from_base(1.0e6),
            attenuation: 1.0,
        }
    }

    #[test]
    fn lensmaker_focal_length_formula() {
        // Plano-concave slow lens (c_lens = 1000) in c_medium = 1500, R = 10 mm.
        // Lensmaker (single refractive surface): F = R·c_m/(c_m − c_lens).
        let lens = custom_lens(10.0e-3, 1000.0);
        let f = lens.focal_length(Velocity::from_base(1500.0));
        let expected = 10.0e-3 * 1500.0 / (1500.0 - 1000.0); // = 30 mm
        assert!(
            (f.into_base() - expected).abs() < 1e-12,
            "F {f:?} != {expected}"
        );
        assert!((f.into_base() - 0.030).abs() < 1e-9, "expected 30 mm focus");
    }

    #[test]
    fn silicone_constructor_round_trips_focal_length_and_f_number() {
        let (f_design, aperture) = (50.0e-3, 20.0e-3);
        let lens = AcousticLens::silicone(Length::from_base(f_design), Length::from_base(aperture));
        let f = lens.focal_length(Velocity::from_base(SOUND_SPEED_TISSUE));
        // silicone() sets R = F(c_t − c_l)/c_t, which focal_length inverts back to F.
        assert!(
            (f.into_base() - f_design).abs() < 1e-9,
            "focal length {:?} != design {f_design}",
            f
        );
        assert!(
            (lens.f_number(
                Length::from_base(aperture),
                Velocity::from_base(SOUND_SPEED_TISSUE)
            ) - f_design / aperture)
                .abs()
                < 1e-9,
            "f-number must equal F/aperture"
        );
    }

    #[test]
    fn aperture_delay_profile_is_zero_centred_monotone_and_paraxial() {
        let (f_design, c_m) = (50.0e-3, SOUND_SPEED_TISSUE);
        let lens = AcousticLens::silicone(Length::from_base(f_design), Length::from_base(20.0e-3));
        let radii: Vec<Length> = (0..6)
            .map(|i| Length::from_base(i as f64 * 1.0e-3))
            .collect(); // 0..5 mm
        let tau = lens.aperture_delay_profile(&radii, Velocity::from_base(c_m));

        assert!(tau[0].into_base().abs() < 1e-18, "τ(0) must be 0");
        for w in tau.windows(2) {
            assert!(w[1] > w[0], "τ(r) must strictly increase with r");
        }
        // Paraxial limit τ(r) ≈ r²/(2 c F): relative error ~ r²/(4F²) ≈ 0.25% at r=5mm.
        let r = 5.0e-3;
        let exact = lens.aperture_delay_profile(&[Length::from_base(r)], Velocity::from_base(c_m))
            [0]
        .into_base();
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

    #[test]
    fn fresnel_zone_radius_satisfies_half_wave_path_condition() {
        // λ = 0.5 mm (≈3 MHz, water), F = 40 mm, aperture 15 mm.
        let (lambda, f) = (0.5e-3, 40.0e-3);
        let zp = FresnelZonePlate::new(
            Length::from_base(f),
            Length::from_base(lambda),
            Length::from_base(15.0e-3),
        );
        // Closed form for n = 1..4.
        for n in 1..=4 {
            let r = zp.zone_radius(n).into_base();
            // The defining condition: √(r_n² + F²) − F = n λ/2.
            let path_excess = (r * r + f * f).sqrt() - f;
            assert!(
                (path_excess - n as f64 * lambda / 2.0).abs() < 1e-12,
                "zone {n}: path excess {path_excess} != nλ/2"
            );
        }
    }

    #[test]
    fn fresnel_zone_radii_are_paraxial_sqrt_and_increasing() {
        let (lambda, f) = (0.5e-3, 40.0e-3);
        let zp = FresnelZonePlate::new(
            Length::from_base(f),
            Length::from_base(lambda),
            Length::from_base(15.0e-3),
        );
        let radii = zp.zone_radii();
        // Several zones fit (paraxial count ≈ a²/(λF) = 0.0225/2e-5 ≈ 11).
        assert!(radii.len() >= 10, "expected ≥10 zones, got {}", radii.len());
        for w in radii.windows(2) {
            assert!(w[1] > w[0], "zone radii must increase");
        }
        // Paraxial r_n ≈ √(n λ F) within 1% for the inner zones (n ≪ F/λ).
        let r1_paraxial = (lambda * f).sqrt();
        assert!(
            (radii[0].into_base() - r1_paraxial).abs() / r1_paraxial < 0.01,
            "r₁ {} vs paraxial {r1_paraxial}",
            radii[0].into_base()
        );
        // All within the aperture; the next zone would exceed it.
        assert!(radii.iter().all(|&r| r <= Length::from_base(15.0e-3)));
        assert!(zp.zone_radius(radii.len() + 1) > Length::from_base(15.0e-3));
    }

    #[test]
    fn fresnel_f_number_is_focal_over_diameter() {
        let zp = FresnelZonePlate::new(
            Length::from_base(40.0e-3),
            Length::from_base(0.5e-3),
            Length::from_base(10.0e-3),
        );
        assert!((zp.f_number() - 40.0e-3 / 20.0e-3).abs() < 1e-12);
    }

    #[test]
    fn corrective_lens_thickness_inverts_phase_to_thickness() {
        // Maimbourg 2020: f₀=914 kHz, c_water=1485, c_lens=1000, K=2 mm.
        let (f0, c_w, c_l, k) = (914.0e3, 1485.0, 1000.0, 2.0e-3);
        // Uniform phase ⇒ uniform thickness equal to the minimal thickness K.
        let flat = corrective_lens_thickness(
            &[1.3; 5],
            Frequency::from_base(f0),
            Velocity::from_base(c_w),
            Velocity::from_base(c_l),
            Length::from_base(k),
        );
        assert!(flat.iter().all(|p| (p.into_base() - k).abs() < 1e-15));

        // A 2π phase difference maps to a one-wavelength-equivalent thickness
        // step |Δp| = 1/(f₀·|1/c_w − 1/c_l|).
        let two_pi = 2.0 * std::f64::consts::PI;
        let p = corrective_lens_thickness(
            &[0.0, two_pi],
            Frequency::from_base(f0),
            Velocity::from_base(c_w),
            Velocity::from_base(c_l),
            Length::from_base(k),
        );
        let expected_step = 1.0 / (f0 * (c_w.recip() - c_l.recip()).abs());
        assert!(
            ((p[1].into_base() - p[0].into_base()).abs() - expected_step).abs() < 1e-9,
            "Δthickness {} != {expected_step}",
            (p[1].into_base() - p[0].into_base()).abs()
        );
        // Thinnest point is exactly the minimal castable thickness.
        let pmin = p
            .iter()
            .map(|value| value.into_base())
            .fold(f64::INFINITY, f64::min);
        assert!((pmin - k).abs() < 1e-15);
    }

    #[test]
    fn isoplanatic_steering_pose_matches_maimbourg_table() {
        let f = 61.0e-3; // 61 mm focal length (H101 transducer)
                         // On-axis: no rotation, no pullback.
        let (th0, tz0) = isoplanatic_steering_pose(Length::from_base(0.0), Length::from_base(f))
            .expect("on-axis");
        assert!(th0.abs() < 1e-15 && tz0.into_base().abs() < 1e-15);

        // Paper Figure 2, x = 11.2 mm ⇒ θ_y = 10°35′ ≈ 10.583°, z-stage ≈ 1.0 mm.
        let (theta, t_z) =
            isoplanatic_steering_pose(Length::from_base(11.2e-3), Length::from_base(f))
                .expect("steered");
        let theta_deg = theta.to_degrees();
        assert!(
            (theta_deg - (10.0 + 35.0 / 60.0)).abs() < 0.02,
            "θ_y {theta_deg}° != 10°35′"
        );
        assert!(
            (t_z.into_base() - 1.0e-3).abs() < 0.05e-3,
            "T_z {} mm != ~1.0 mm",
            t_z.into_base() * 1e3
        );

        // |x| > F is unphysical.
        assert!(
            isoplanatic_steering_pose(Length::from_base(70.0e-3), Length::from_base(f)).is_none()
        );
    }
}
