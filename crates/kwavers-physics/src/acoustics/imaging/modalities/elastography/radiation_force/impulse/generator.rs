use crate::acoustics::mechanics::elastic_wave::ElasticBodyForceConfig;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::absorption::PowerLawAbsorption;
use kwavers_medium::Medium;
use leto::Array3;

use super::super::patterns::MultiDirectionalPush;
use super::parameters::PushPulseParameters;

/// Acoustic radiation force generator
#[derive(Debug)]
pub struct AcousticRadiationForce {
    /// Push pulse configuration
    parameters: PushPulseParameters,
    /// Medium sound speed (m/s)
    sound_speed: f64,
    /// Power-law absorption model (α(f) = α₀·f^y for soft tissue) — evaluated
    /// at the current push frequency in `absorption_np_per_m()`.
    absorption_model: PowerLawAbsorption,
    /// Medium density (kg/m³)
    density: f64,
    /// Computational grid
    grid: Grid,
}

impl AcousticRadiationForce {
    /// Create new acoustic radiation force generator
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `medium` - Tissue medium properties
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        // Get medium properties at center
        let (nx, ny, nz) = grid.dimensions();
        let ci = nx / 2;
        let cj = ny / 2;
        let ck = nz / 2;

        let sound_speed = medium.sound_speed(ci, cj, ck);
        let density = medium.density(ci, cj, ck);

        // Soft-tissue power-law absorption α(f) = α₀·f^y from SSOT
        // (Duck 1990, Physical Properties of Tissue). Evaluated at the active
        // push frequency in `absorption_np_per_m()`. The previous code stored
        // a single scalar 5.8 Np/m — the 1 MHz value — even though the
        // default push frequency is 5 MHz, under-predicting the radiation-
        // force density `f = 2αI/c` by ~5× whenever the caller used the
        // default 5 MHz pulse.
        let absorption_model = PowerLawAbsorption::soft_tissue();

        Ok(Self {
            parameters: PushPulseParameters::default(),
            sound_speed,
            absorption_model,
            density,
            grid: grid.clone(),
        })
    }

    /// Power-law absorption coefficient evaluated at the active push frequency
    /// (Np/m).
    #[inline]
    fn absorption_np_per_m(&self) -> f64 {
        self.absorption_model
            .absorption_at_frequency(self.parameters.frequency)
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn push_pulse_body_force(
        &self,
        push_location: [f64; 3],
    ) -> KwaversResult<ElasticBodyForceConfig> {
        // Calculate radiation force density magnitude (N/m³)
        // f = (2·α(f)·I)/c — α is frequency-dependent power-law absorption.
        let force_density =
            (2.0 * self.absorption_np_per_m() * self.parameters.intensity) / self.sound_speed;

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
        // model the beam axis as +z.
        //
        // Not yet implemented: complete radiation force physics. Absent: monopole + dipole
        // force terms F = (αI/c) + (3V₀/4π)Re[(p₁p₂*)/(ρc²)] (Wu & Nyborg 1990, JASA 87);
        // Reynolds stress tensor acoustic streaming; nonlinear absorption and scattering
        // contributions; frequency-dependent dispersion corrections; and multi-frequency
        // excitation for improved SNR and depth penetration (Sarvazyan et al. 1998).
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
    /// Push pulse pseudo displacement.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn push_pulse_pseudo_displacement(
        &self,
        push_location: [f64; 3],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut displacement = Array3::zeros((nx, ny, nz));

        // Calculate radiation force density
        // f = (2·α(f)·I)/c (N/m³) — α evaluated at the active push frequency.
        let force_density =
            (2.0 * self.absorption_np_per_m() * self.parameters.intensity) / self.sound_speed;

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

                    let r_lateral = dx.hypot(dy);
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
