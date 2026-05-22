//! Steering vector calculations for beamforming
//!
//! ## Mathematical Foundation
//! **Plane Wave**: a(θ,φ) = exp(j k r · û) where k = 2π/λ, û is unit direction vector
//! **Spherical Wave**: a(r) = exp(j k |r - r₀|) / |r - r₀| for near-field sources
//! **Focused Beam**: Combines phase delays for beam focusing at specific point

use crate::core::error::KwaversResult;
use crate::math::geometry::delays;
use crate::math::geometry::distance3;
use ndarray::Array1;

/// Steering vector calculation methods
#[derive(Debug, Clone, PartialEq)]
pub enum SteeringVectorMethod {
    /// Far-field plane wave assumption: a(θ,φ) = exp(j k r · û)
    PlaneWave,
    /// Near-field spherical wave: a(r) = exp(j k |r - r₀|) / |r - r₀|
    SphericalWave { source_position: [f64; 3] },
    /// Focused beam at specific point
    Focused { focal_point: [f64; 3] },
}

/// Steering vector computation for array processing
#[derive(Debug)]
pub struct SteeringVector;

impl SteeringVector {
    /// Compute plane wave.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compute_plane_wave(
        direction: [f64; 3],
        frequency: f64,
        sensor_positions: &[[f64; 3]],
        speed_of_sound: f64,
    ) -> KwaversResult<Array1<num_complex::Complex<f64>>> {
        use num_complex::Complex;

        let phase_delays = delays::plane_wave_phase_delays(
            sensor_positions,
            direction,
            frequency,
            speed_of_sound,
        )?;

        let mut steering_vector = Array1::zeros(sensor_positions.len());
        for (i, &phase) in phase_delays.iter().enumerate() {
            steering_vector[i] = Complex::new(0.0, phase).exp();
        }

        Ok(steering_vector)
    }

    /// Compute steering vector for given direction and sensor positions
    /// Returns complex-valued steering vector as `Array1<Complex<f64>>`
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compute(
        method: &SteeringVectorMethod,
        direction: [f64; 3],
        frequency: f64,
        sensor_positions: &[[f64; 3]],
        speed_of_sound: f64,
    ) -> KwaversResult<Array1<num_complex::Complex<f64>>> {
        use num_complex::Complex;

        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "frequency must be finite and > 0".to_owned(),
            ));
        }
        if !speed_of_sound.is_finite() || speed_of_sound <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "speed_of_sound must be finite and > 0".to_owned(),
            ));
        }
        if sensor_positions.is_empty() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "sensor_positions must be non-empty".to_owned(),
            ));
        }
        if sensor_positions
            .iter()
            .any(|p| !p.iter().all(|v| v.is_finite()))
        {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "sensor_positions must be finite 3D coordinates".to_owned(),
            ));
        }

        let wavenumber = 2.0 * std::f64::consts::PI * frequency / speed_of_sound;
        let num_sensors = sensor_positions.len();

        let mut steering_vector = Array1::zeros(num_sensors);

        match method {
            SteeringVectorMethod::PlaneWave => {
                steering_vector = Self::compute_plane_wave(
                    direction,
                    frequency,
                    sensor_positions,
                    speed_of_sound,
                )?;
            }

            SteeringVectorMethod::SphericalWave { source_position } => {
                if source_position.iter().any(|v| !v.is_finite()) {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "source_position must be finite".to_owned(),
                    ));
                }
                // Spherical wave steering: a_i = exp(j k |r_i - r₀|) / |r_i - r₀|
                // where r₀ is the source position
                for (i, &pos) in sensor_positions.iter().enumerate() {
                    let distance = distance3(pos, *source_position);
                    if distance.abs() < 1e-12 {
                        return Err(crate::core::error::KwaversError::Numerical(
                            crate::core::error::NumericalError::InvalidOperation(
                                format!("Sensor at source position (distance = {:.2e}) - spherical wave steering undefined", distance)
                            )
                        ));
                    }
                    let phase = wavenumber * distance;
                    let amplitude = 1.0 / distance; // Spherical spreading
                    steering_vector[i] = Complex::new(0.0, phase).exp() * amplitude;
                }
            }

            SteeringVectorMethod::Focused { focal_point } => {
                let phase_delays = delays::focus_phase_delays(
                    sensor_positions,
                    *focal_point,
                    frequency,
                    speed_of_sound,
                )?;
                for (i, &phase) in phase_delays.iter().enumerate() {
                    steering_vector[i] = Complex::new(0.0, phase).exp();
                }
            }
        }

        Ok(steering_vector)
    }

    /// Compute real-valued steering vector (phase-only for delay-and-sum)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compute_real(
        method: &SteeringVectorMethod,
        direction: [f64; 3],
        frequency: f64,
        sensor_positions: &[[f64; 3]],
        speed_of_sound: f64,
    ) -> KwaversResult<Array1<f64>> {
        let complex_steering = Self::compute(
            method,
            direction,
            frequency,
            sensor_positions,
            speed_of_sound,
        )?;
        Ok(complex_steering.mapv(|c| c.re)) // Take real part for delay-and-sum compatibility
    }

    /// Compute broadside steering vector (perpendicular to array axis)
    /// # Panics
    /// - Panics if `broadside steering computation must succeed`.
    ///
    #[must_use]
    pub fn broadside(
        sensor_positions: &[[f64; 3]],
        frequency: f64,
        speed_of_sound: f64,
    ) -> Array1<f64> {
        // Broadside: direction perpendicular to array (typically [0, 0, 1] or [0, 1, 0])
        Self::compute_real(
            &SteeringVectorMethod::PlaneWave,
            [0.0, 0.0, 1.0], // z-direction
            frequency,
            sensor_positions,
            speed_of_sound,
        )
        .expect("broadside steering computation must succeed")
    }

    /// Compute endfire steering vector (along array axis)
    /// # Panics
    /// - Panics if `endfire steering computation must succeed`.
    ///
    #[must_use]
    pub fn endfire(
        sensor_positions: &[[f64; 3]],
        frequency: f64,
        speed_of_sound: f64,
    ) -> Array1<f64> {
        // Endfire: direction along array axis (typically [1, 0, 0])
        Self::compute_real(
            &SteeringVectorMethod::PlaneWave,
            [1.0, 0.0, 0.0], // x-direction
            frequency,
            sensor_positions,
            speed_of_sound,
        )
        .expect("endfire steering computation must succeed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

    // ─── SteeringVector exact value tests ─────────────────────────────────────

    /// Broadside plane wave (direction=[0,0,1]) along sensors aligned on x-axis.
    ///
    /// Sensors at [0,0,0] and [d,0,0]; direction = [0,0,1] (z-axis).
    /// Phase delay at each sensor: k·(pos·dir) = k·0 = 0 for both.
    /// Steering: [exp(j·0), exp(j·0)] = [1+0j, 1+0j].
    #[test]
    fn steering_plane_wave_broadside_all_ones() {
        let f = 1000.0_f64;
        let c = SOUND_SPEED_WATER_SIM;
        let sensors: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let a = SteeringVector::compute(
            &SteeringVectorMethod::PlaneWave,
            [0.0, 0.0, 1.0],
            f,
            sensors,
            c,
        )
        .expect("broadside steering should succeed");

        assert_eq!(a.len(), 2, "steering vector length must equal n_sensors");
        for (i, &val) in a.iter().enumerate() {
            assert!(
                (val.re - 1.0).abs() < 1e-14 && val.im.abs() < 1e-14,
                "steering[{i}] = {val:?} (expected 1+0j for broadside)"
            );
        }
    }

    /// Endfire plane wave (direction=[1,0,0]) gives quarter-wave negative phase at d=λ/4.
    ///
    /// Phase convention: delays[i] = −k·(pos·dir) (standard receive beamforming sign).
    /// Sensors at [0,0,0] and [λ/4, 0, 0]; direction = [1,0,0].
    /// k = 2πf/c; phase[0]=0, phase[1] = −k·λ/4 = −π/2.
    /// a = [exp(j·0), exp(−j·π/2)] = [1+0j, 0−j].
    #[test]
    fn steering_plane_wave_endfire_quarter_wave_gives_negative_imaginary_unit() {
        let f = 1000.0_f64;
        let c = SOUND_SPEED_WATER_SIM;
        let lambda = c / f; // 1.5 m
        let sensors: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [lambda / 4.0, 0.0, 0.0]];
        let a = SteeringVector::compute(
            &SteeringVectorMethod::PlaneWave,
            [1.0, 0.0, 0.0],
            f,
            sensors,
            c,
        )
        .expect("endfire steering should succeed");

        assert_eq!(a.len(), 2);
        // sensor 0: exp(j·0) = 1+0j
        assert!(
            (a[0].re - 1.0).abs() < 1e-13,
            "a[0].re={}, expected 1.0",
            a[0].re
        );
        assert!(a[0].im.abs() < 1e-13, "a[0].im={}, expected 0.0", a[0].im);
        // sensor 1: exp(−j·π/2) = 0−j (receive convention: phase = −k·dot)
        assert!(a[1].re.abs() < 1e-13, "a[1].re={}, expected 0.0", a[1].re);
        assert!(
            (a[1].im + 1.0).abs() < 1e-13,
            "a[1].im={}, expected −1.0",
            a[1].im
        );
    }

    /// Endfire plane wave at full wavelength spacing gives no phase shift (2π cycle).
    ///
    /// d = λ = c/f; phase[1] = k·λ = 2π → exp(j·2π) = 1+0j.
    #[test]
    fn steering_plane_wave_full_wavelength_spacing_is_in_phase() {
        let f = 1000.0_f64;
        let c = SOUND_SPEED_WATER_SIM;
        let lambda = c / f;
        let sensors: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [lambda, 0.0, 0.0]];
        let a = SteeringVector::compute(
            &SteeringVectorMethod::PlaneWave,
            [1.0, 0.0, 0.0],
            f,
            sensors,
            c,
        )
        .expect("full-wavelength steering should succeed");

        // exp(j·2π) = 1+0j
        assert!(
            (a[1].re - 1.0).abs() < 1e-12,
            "a[1].re={} (expected 1.0 at full wavelength)",
            a[1].re
        );
        assert!(
            a[1].im.abs() < 1e-12,
            "a[1].im={} (expected 0.0 at full wavelength)",
            a[1].im
        );
    }

    /// SphericalWave at distance = λ gives amplitude 1/λ at phase 2π.
    ///
    /// sensor at origin, source at [λ, 0, 0]; distance=λ.
    /// phase = k·λ = 2π → exp(j·2π) = 1 → a[0] = 1/λ + 0j.
    #[test]
    fn steering_spherical_wave_one_wavelength_exact_amplitude() {
        let f = 1000.0_f64;
        let c = SOUND_SPEED_WATER_SIM;
        let lambda = c / f; // 1.5 m
        let sensors: &[[f64; 3]] = &[[0.0, 0.0, 0.0]];
        let source = [lambda, 0.0, 0.0];
        let a = SteeringVector::compute(
            &SteeringVectorMethod::SphericalWave {
                source_position: source,
            },
            [0.0, 0.0, 1.0],
            f,
            sensors,
            c,
        )
        .expect("spherical wave steering should succeed");

        // a[0] = exp(j·2π) / λ = 1/λ + 0j
        let expected_re = 1.0 / lambda;
        assert!(
            (a[0].re - expected_re).abs() < 1e-12,
            "a[0].re={} (expected 1/λ={expected_re})",
            a[0].re
        );
        assert!(a[0].im.abs() < 1e-12, "a[0].im={} (expected 0.0)", a[0].im);
    }

    /// `SteeringVector::broadside` returns all-ones real vector for x-axis sensors.
    ///
    /// direction=[0,0,1]: all sensors have zero projection onto z → phase=0 → Re[exp(0)]=1.
    #[test]
    fn steering_broadside_returns_all_ones_for_x_aligned_sensors() {
        let f = 1000.0_f64;
        let c = SOUND_SPEED_WATER_SIM;
        let sensors: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let a = SteeringVector::broadside(sensors, f, c);
        for (i, &val) in a.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-13,
                "broadside[{i}] = {val} (expected 1.0)"
            );
        }
    }

    /// `SteeringVector::compute` rejects non-unit direction vectors.
    #[test]
    fn steering_rejects_non_unit_direction() {
        let sensors: &[[f64; 3]] = &[[0.0, 0.0, 0.0]];
        let result = SteeringVector::compute(
            &SteeringVectorMethod::PlaneWave,
            [1.0, 1.0, 0.0], // norm = √2, not unit
            1000.0,
            sensors,
            SOUND_SPEED_WATER_SIM,
        );
        assert!(result.is_err(), "non-unit direction must be rejected");
    }
}
