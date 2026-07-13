//! Fast acoustic intensity and bioheat calculations

use super::planner::TreatmentPlanner;
use super::types::{TranscranialTargetVolume, TransducerSetup};
use eunomia::Complex;
use kwavers_core::constants::fundamental::ACOUSTIC_ABSORPTION_TISSUE;
use kwavers_core::constants::medical::{
    BLOOD_PERFUSION_RATE_BRAIN, BLOOD_SPECIFIC_HEAT, THERMAL_DOSE_THRESHOLD,
};
use kwavers_core::constants::numerical::CM_TO_M;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::tissue_acoustics::{DENSITY_BLOOD, DENSITY_BRAIN, SOUND_SPEED_BRAIN};
use kwavers_core::constants::{BODY_TEMPERATURE_C, NP_TO_DB};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::Array3;

const MILLIMETERS_TO_METERS: f64 = 1.0e-3;
// SSOT: SOUND_SPEED_BRAIN and DENSITY_BRAIN imported from core::constants::tissue_acoustics
// dB/(cm·MHz) → Np/m at 1 MHz: divide by NP_TO_DB then by CM_TO_M (SSOT).
const BRAIN_ABSORPTION_NP_PER_M: f64 = ACOUSTIC_ABSORPTION_TISSUE / NP_TO_DB / CM_TO_M;
const ABSORPTION_RATE_PER_INTENSITY: f64 = 0.5;

impl TreatmentPlanner {
    /// Simulate acoustic intensity field from a phased-array transducer using coherent
    /// superposition of spherical waves (Rayleigh-Sommerfeld approximation).
    ///
    /// # Theory
    ///
    /// Each element `n` at position `r_n` radiates a monopole spherical wave with
    /// complex pressure amplitude:
    ///
    /// ```text
    /// p_n(r) = A_n · exp(i φ_n) · exp(i k |r − r_n|) / (4π |r − r_n|)
    /// ```
    ///
    /// where:
    /// - `k = 2π f / c` is the wavenumber [rad/m]
    /// - `A_n` is the element amplitude (unity assumed)
    /// - `φ_n` is the phase delay applied to element `n`
    ///
    /// The total acoustic pressure is the coherent sum over all elements:
    ///
    /// ```text
    /// p(r) = Σ_n p_n(r)
    /// ```
    ///
    /// and the acoustic intensity (time-averaged) is:
    ///
    /// ```text
    /// I(r) = |p(r)|² / (2 ρ c)
    /// ```
    ///
    /// Reference: O'Neil HT (1949), *J Acoust Soc Am* 21(5):516–526;
    /// Daum DR & Hynynen K (1999), *IEEE Trans Biomed Eng* 46(9):1070–1082.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn simulate_acoustic_field(
        &self,
        setup: &TransducerSetup,
    ) -> KwaversResult<Array3<f64>> {
        validate_transducer_setup(setup)?;

        let (nx, ny, nz) = self.brain_grid.dimensions();
        let mut acoustic_field = Array3::zeros([nx, ny, nz]);

        // Wavenumber [rad/m]: k = 2π f / c_brain
        let k_wave = TWO_PI * setup.frequency / SOUND_SPEED_BRAIN;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let xv = i as f64 * self.brain_grid.dx;
                    let yv = j as f64 * self.brain_grid.dy;
                    let zv = k as f64 * self.brain_grid.dz;

                    acoustic_field[[i, j, k]] =
                        harmonic_intensity_at_point([xv, yv, zv], setup, k_wave);
                }
            }
        }

        Ok(acoustic_field)
    }

    /// Calculate thermal response using the Pennes bioheat equation in steady state.
    ///
    /// # Theory — Pennes Bioheat Equation (1948)
    ///
    /// The steady-state bioheat equation (no diffusion, spatially uniform):
    ///
    /// ```text
    /// 0 = Q(r) − W_b ρ_b c_b [T(r) − T_a]
    /// ```
    ///
    /// where:
    /// - `Q = 2α I` is the volumetric heat source [W/m³]
    /// - `α` = amplitude absorption coefficient [Np/m]
    /// - `W_b` = blood perfusion rate [m³/(m³·s)] = 1/s
    /// - `ρ_b c_b` = blood heat capacity density [J/(m³·K)]
    ///
    /// Solving:
    /// ```text
    /// ΔT = Q / (W_b · ρ_b · c_b)  (K)  (steady-state, no diffusion)
    /// ```
    ///
    /// Reference: Pennes HH (1948). "Analysis of tissue and arterial blood temperatures
    /// in the resting human forearm." *J Appl Physiol* 1(2):93–122.
    /// Nyborg WL (1988). *Phys Med Biol* 33(7):785–792.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn calculate_thermal_response(
        &self,
        acoustic_field: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = acoustic_field.shape();
        let mut temperature_field = Array3::zeros([nx, ny, nz]);

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let intensity = acoustic_field[[i, j, k]]; // [W/m²]
                    let Some(temperature) = steady_state_temperature_from_intensity(intensity)
                    else {
                        return Err(validation_error(format!(
                            "Acoustic intensity {intensity} W/m^2 is outside the finite nonnegative domain"
                        )));
                    };
                    temperature_field[[i, j, k]] = temperature;
                }
            }
        }

        Ok(temperature_field)
    }

    /// Estimate treatment time
    pub(crate) fn estimate_treatment_time(
        &self,
        _targets: &[TranscranialTargetVolume],
        acoustic_field: &Array3<f64>,
    ) -> f64 {
        estimate_treatment_time_from_intensity_field(acoustic_field)
    }
}

fn validation_error(message: impl Into<String>) -> KwaversError {
    KwaversError::Validation(ValidationError::ConstraintViolation {
        message: message.into(),
    })
}

fn validate_transducer_setup(setup: &TransducerSetup) -> KwaversResult<()> {
    if setup.num_elements == 0 {
        return Err(validation_error(
            "Transducer setup requires at least one element",
        ));
    }
    if !setup.frequency.is_finite() || setup.frequency <= 0.0 {
        return Err(validation_error(format!(
            "Transducer frequency {} Hz is outside the positive finite domain",
            setup.frequency
        )));
    }
    if setup.element_positions.len() != setup.num_elements
        || setup.element_phases.len() != setup.num_elements
        || setup.element_amplitudes.len() != setup.num_elements
    {
        return Err(validation_error(format!(
            "Transducer vectors must match num_elements {}; got positions {}, phases {}, amplitudes {}",
            setup.num_elements,
            setup.element_positions.len(),
            setup.element_phases.len(),
            setup.element_amplitudes.len()
        )));
    }

    for (idx, element) in setup.element_positions.iter().enumerate() {
        if !element.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(validation_error(format!(
                "Transducer element {idx} has a nonfinite position"
            )));
        }
    }
    for (idx, &phase) in setup.element_phases.iter().enumerate() {
        if !phase.is_finite() {
            return Err(validation_error(format!(
                "Transducer element {idx} has a nonfinite phase"
            )));
        }
    }
    for (idx, &amplitude) in setup.element_amplitudes.iter().enumerate() {
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(validation_error(format!(
                "Transducer element {idx} amplitude {amplitude} is outside the finite nonnegative domain"
            )));
        }
    }

    Ok(())
}

fn harmonic_intensity_at_point(point_m: [f64; 3], setup: &TransducerSetup, k_wave: f64) -> f64 {
    let mut p_total = Complex::new(0.0_f64, 0.0_f64);

    for idx in 0..setup.num_elements {
        let element = setup.element_positions[idx];
        let element_m = [
            element[0] * MILLIMETERS_TO_METERS,
            element[1] * MILLIMETERS_TO_METERS,
            element[2] * MILLIMETERS_TO_METERS,
        ];
        let dx = point_m[0] - element_m[0];
        let dy = point_m[1] - element_m[1];
        let dz = point_m[2] - element_m[2];
        let r = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
        if r < 1e-9 {
            continue;
        }

        let phase = k_wave * r + setup.element_phases[idx];
        let amplitude = setup.element_amplitudes[idx];
        p_total += amplitude * Complex::new(phase.cos(), phase.sin()) / r;
    }

    p_total.norm_sqr() / (2.0 * DENSITY_BRAIN * SOUND_SPEED_BRAIN)
}

fn steady_state_temperature_from_intensity(intensity: f64) -> Option<f64> {
    if !intensity.is_finite() || intensity < 0.0 {
        return None;
    }

    let perfusion_sink = BLOOD_PERFUSION_RATE_BRAIN * DENSITY_BLOOD * BLOOD_SPECIFIC_HEAT;
    let heat_source = 2.0 * BRAIN_ABSORPTION_NP_PER_M * intensity;
    Some(BODY_TEMPERATURE_C + heat_source / perfusion_sink)
}

fn max_nonnegative_finite_intensity(acoustic_field: &Array3<f64>) -> Option<f64> {
    let mut max_intensity = 0.0_f64;

    for &intensity in acoustic_field.iter() {
        if !intensity.is_finite() || intensity < 0.0 {
            return None;
        }
        max_intensity = max_intensity.max(intensity);
    }

    Some(max_intensity)
}

fn estimate_treatment_time_from_intensity_field(acoustic_field: &Array3<f64>) -> f64 {
    let Some(max_intensity) = max_nonnegative_finite_intensity(acoustic_field) else {
        return f64::INFINITY;
    };

    if max_intensity > 0.0 {
        THERMAL_DOSE_THRESHOLD / (ABSORPTION_RATE_PER_INTENSITY * max_intensity)
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

    fn one_element_setup(amplitude: f64) -> TransducerSetup {
        TransducerSetup {
            num_elements: 1,
            element_positions: vec![[0.0, 0.0, 0.0]],
            element_phases: vec![0.0],
            element_amplitudes: vec![amplitude],
            frequency: MHZ_TO_HZ,
            focal_distance: 1.0,
        }
    }

    #[test]
    fn validate_transducer_setup_rejects_invalid_domains() {
        let mut setup = one_element_setup(1.0);
        setup.frequency = 0.0;
        let frequency_error = validate_transducer_setup(&setup).unwrap_err();
        assert!(frequency_error.to_string().contains("frequency"));

        let mut setup = one_element_setup(1.0);
        setup.element_amplitudes[0] = f64::NAN;
        let amplitude_error = validate_transducer_setup(&setup).unwrap_err();
        assert!(amplitude_error.to_string().contains("amplitude"));

        let mut setup = one_element_setup(1.0);
        setup.element_phases.clear();
        let length_error = validate_transducer_setup(&setup).unwrap_err();
        assert!(length_error.to_string().contains("vectors"));
    }

    #[test]
    fn harmonic_intensity_uses_millimeter_element_positions() {
        let setup = TransducerSetup {
            num_elements: 1,
            element_positions: vec![[1.0, 0.0, 0.0]],
            element_phases: vec![0.0],
            element_amplitudes: vec![1.0],
            frequency: MHZ_TO_HZ,
            focal_distance: 1.0,
        };
        let point_m = [0.002, 0.0, 0.0];
        let k_wave = 2.0 * std::f64::consts::PI * setup.frequency / SOUND_SPEED_BRAIN;

        let intensity = harmonic_intensity_at_point(point_m, &setup, k_wave);
        let expected = 1.0 / (0.001_f64.powi(2) * 2.0 * DENSITY_BRAIN * SOUND_SPEED_BRAIN);

        assert!((intensity - expected).abs() < expected * 1.0e-12);
    }

    #[test]
    fn harmonic_intensity_scales_with_amplitude_squared() {
        let point_m = [0.001, 0.0, 0.0];
        let low = one_element_setup(1.0);
        let high = one_element_setup(2.0);
        let k_wave = 2.0 * std::f64::consts::PI * low.frequency / SOUND_SPEED_BRAIN;

        let low_intensity = harmonic_intensity_at_point(point_m, &low, k_wave);
        let high_intensity = harmonic_intensity_at_point(point_m, &high, k_wave);

        assert!((high_intensity / low_intensity - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn steady_state_temperature_matches_pennes_source_balance() {
        let intensity = 2.0;
        let temperature = steady_state_temperature_from_intensity(intensity)
            .expect("finite nonnegative intensity must map to temperature");
        let perfusion_sink = BLOOD_PERFUSION_RATE_BRAIN * DENSITY_BLOOD * BLOOD_SPECIFIC_HEAT;
        let expected =
            BODY_TEMPERATURE_C + (2.0 * BRAIN_ABSORPTION_NP_PER_M * intensity) / perfusion_sink;

        assert!((temperature - expected).abs() < 1.0e-14);
    }

    #[test]
    fn steady_state_temperature_rejects_invalid_intensity() {
        assert!(steady_state_temperature_from_intensity(-1.0).is_none());
        assert!(steady_state_temperature_from_intensity(f64::NAN).is_none());
        assert!(steady_state_temperature_from_intensity(f64::INFINITY).is_none());
    }

    #[test]
    fn treatment_time_matches_peak_intensity_contract() {
        let field = Array3::from_shape_vec([2, 1, 1], vec![5.0, 10.0]).expect("shape matches");
        let time = estimate_treatment_time_from_intensity_field(&field);
        let expected = THERMAL_DOSE_THRESHOLD / (ABSORPTION_RATE_PER_INTENSITY * 10.0);

        assert!((time - expected).abs() < 1.0e-12);
    }

    #[test]
    fn treatment_time_is_infinite_without_valid_heating() {
        let zero = Array3::zeros([1, 1, 1]);
        let invalid = Array3::from_elem([1, 1, 1], f64::NAN);
        let negative = Array3::from_elem([1, 1, 1], -1.0);

        assert!(estimate_treatment_time_from_intensity_field(&zero).is_infinite());
        assert!(estimate_treatment_time_from_intensity_field(&invalid).is_infinite());
        assert!(estimate_treatment_time_from_intensity_field(&negative).is_infinite());
    }
}
