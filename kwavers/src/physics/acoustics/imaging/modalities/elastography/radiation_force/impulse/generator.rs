use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::physics::acoustics::mechanics::elastic_wave::ElasticBodyForceConfig;
use ndarray::Array3;

use super::parameters::PushPulseParameters;
use super::super::patterns::MultiDirectionalPush;

/// Acoustic radiation force generator
#[derive(Debug)]
pub struct AcousticRadiationForce {
    /// Push pulse configuration
    parameters: PushPulseParameters,
    /// Medium sound speed (m/s)
    sound_speed: f64,
    /// Medium absorption coefficient (Np/m)
    absorption: f64,
    /// Medium density (kg/m³)
    #[allow(dead_code)]
    density: f64,
    /// Computational grid
    #[allow(dead_code)]
    grid: Grid,
}

impl AcousticRadiationForce {
    /// Create new acoustic radiation force generator
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `medium` - Tissue medium properties
    pub fn new(grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        // Get medium properties at center
        let (nx, ny, nz) = grid.dimensions();
        let ci = nx / 2;
        let cj = ny / 2;
        let ck = nz / 2;

        let sound_speed = medium.sound_speed(ci, cj, ck);
        let density = medium.density(ci, cj, ck);

        // Estimate absorption coefficient
        // For soft tissue at 5 MHz: α ≈ 0.5 dB/cm/MHz ≈ 5.8 Np/m
        let absorption = 5.8; // Np/m, typical value for soft tissue at 1 MHz
                              // Reference: Duck (1990), Physical Properties of Tissue

        Ok(Self {
            parameters: PushPulseParameters::default(),
            sound_speed,
            absorption,
            density,
            grid: grid.clone(),
        })
    }

    /// Set custom push pulse parameters
    pub fn set_parameters(&mut self, parameters: PushPulseParameters) {
        self.parameters = parameters;
    }

    /// Get current push pulse parameters
    #[must_use]
    pub fn parameters(&self) -> &PushPulseParameters {
        &self.parameters
    }

    /// Create an elastic body-force configuration for an ARFI push pulse.
    ///
    /// # Arguments
    ///
    /// * `push_location` - Focal point [x, y, z] in meters
    ///
    /// # Returns
    ///
    /// Elastic body-force configuration to be consumed by the elastic solver as a source term.
    ///
    /// # Correctness invariant
    ///
    /// This returns a *forcing term* `f(x,t)` with units N/m³, to be used in:
    ///
    ///   ρ ∂v/∂t = ∇·σ + f
    ///
    /// This is intentionally not an “initial displacement” API.
    ///
    /// # References
    ///
    /// Nightingale et al. (2002): Radiation force density f ≈ (2αI)/c.
    pub fn push_pulse_body_force(
        &self,
        push_location: [f64; 3],
    ) -> KwaversResult<ElasticBodyForceConfig> {
        // Calculate radiation force density magnitude (N/m³)
        // f = (2αI)/c
        let force_density = (2.0 * self.absorption * self.parameters.intensity) / self.sound_speed;

        // Convert pulse duration into an impulse density J = ∫ f(t) dt (N·s/m³).
        // We model the temporal envelope as a unit-area Gaussian in the solver, so this is the
        // time integral of the force density.
        let impulse_n_per_m3_s = force_density * self.parameters.duration;

        // Spatial envelope: use Gaussian standard deviations derived from FWHM heuristics.
        // Lateral: FWHM ≈ 1.2 × λ × F-number
        // Axial:   FWHM ≈ 6 × λ × F-number²
        //
        // For a Gaussian exp(-0.5 (r/σ)²), FWHM = 2 √(2 ln 2) σ.
        let wavelength = self.sound_speed / self.parameters.frequency;
        let lateral_fwhm = 1.2 * wavelength * self.parameters.f_number;
        let axial_fwhm = 6.0 * wavelength * self.parameters.f_number * self.parameters.f_number;

        let fwhm_to_sigma = 1.0 / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt());
        let sigma_lateral = (lateral_fwhm * fwhm_to_sigma).max(1e-12);
        let sigma_axial = (axial_fwhm * fwhm_to_sigma).max(1e-12);

        // Temporal envelope: use a Gaussian with σ_t chosen so that ~99% of mass lies within the
        // push duration. For a Gaussian, ±3σ covers ~99.7%, so take σ_t = duration / 6.
        let sigma_t_s = (self.parameters.duration / 6.0).max(1e-12);

        // Direction: ARFI primarily pushes along the beam axis. In this simplified geometry, we
        // TODO_AUDIT: P2 - Advanced Radiation Force Elastography - Implement full acoustic radiation force with absorption, scattering, and streaming effects
        // DEPENDS ON: physics/acoustics/imaging/elastography/radiation_force_exact.rs, physics/acoustics/imaging/elastography/streaming.rs
        // MISSING: Complete radiation force: F = (α I / c) + (3V₀/4π) Re[(p₁p₂*)/(ρc²)] for monopole + dipole terms
        // MISSING: Acoustic streaming effects with Reynolds stress tensor contributions
        // MISSING: Nonlinear absorption and scattering contributions to radiation force
        // MISSING: Frequency-dependent radiation force with dispersion effects
        // MISSING: Multi-frequency radiation force for improved SNR and depth penetration
        // THEOREM: Acoustic radiation pressure: P_rad = (α I₀)/(ρc) for plane waves in absorbing media
        // THEOREM: Gor'kov potential: U = -V₀ [ (p²/(2ρ₀c₀²)) - (3/2) ρ₀ v² ] for particle displacement
        // REFERENCES: Wu & Nyborg (1990) JASA 87, 84; Sarvazyan et al. (1998) Ultrasound Med Biol
        // model the beam axis as +z.
        let direction = [0.0, 0.0, 1.0];

        Ok(ElasticBodyForceConfig::GaussianImpulse {
            center_m: push_location,
            sigma_m: [sigma_lateral, sigma_lateral, sigma_axial],
            direction,
            t0_s: 0.0,
            sigma_t_s,
            impulse_n_per_m3_s,
        })
    }

    #[allow(dead_code)]
    pub fn push_pulse_pseudo_displacement(
        &self,
        push_location: [f64; 3],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut displacement = Array3::zeros((nx, ny, nz));

        // Calculate radiation force density
        // f = (2αI)/c (N/m³)
        let force_density = (2.0 * self.absorption * self.parameters.intensity) / self.sound_speed;

        // NOTE: This computes a quantity with units [m/s], not [m]. Historically this was used as a
        // displacement initializer; we keep it only for backward compatibility.
        let pseudo_displacement_scale = (force_density * self.parameters.duration) / self.density;

        // FWHM heuristics
        let wavelength = self.sound_speed / self.parameters.frequency;
        let lateral_width = 1.2 * wavelength * self.parameters.f_number;
        let axial_length = 6.0 * wavelength * self.parameters.f_number * self.parameters.f_number;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;

                    let dx = x - push_location[0];
                    let dy = y - push_location[1];
                    let dz = z - push_location[2];

                    let r_lateral = (dx * dx + dy * dy).sqrt();
                    let r_axial = dz.abs();

                    let lateral_profile = (-4.0 * (r_lateral / lateral_width).powi(2)).exp();
                    let axial_profile = (-4.0 * (r_axial / axial_length).powi(2)).exp();

                    displacement[[i, j, k]] =
                        pseudo_displacement_scale * lateral_profile * axial_profile;
                }
            }
        }

        Ok(displacement)
    }

    /// Create per-push body-force configs for a multi-directional push sequence.
    ///
    /// This is the correctness-first replacement for summing scalar “initial displacements”.
    pub fn multi_directional_body_forces(
        &self,
        push_sequence: &MultiDirectionalPush,
    ) -> KwaversResult<Vec<ElasticBodyForceConfig>> {
        let mut forces = Vec::with_capacity(push_sequence.pushes.len());
        for push in &push_sequence.pushes {
            let mut cfg = self.push_pulse_body_force(push.location)?;
            // Apply amplitude weighting by scaling impulse density (impulse density is ∫ f dt).
            match &mut cfg {
                ElasticBodyForceConfig::GaussianImpulse {
                    impulse_n_per_m3_s, ..
                } => {
                    *impulse_n_per_m3_s *= push.amplitude_weight;
                }
            }
            forces.push(cfg);
        }
        Ok(forces)
    }
}
