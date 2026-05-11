use super::FieldCoupler;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use crate::core::constants::thermodynamic::SPECIFIC_HEAT_WATER;
use crate::core::error::KwaversResult;
use crate::domain::field::indices::{LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
use ndarray::Array3;

impl FieldCoupler {
    /// Apply weak coupling (single pass)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn apply_weak_coupling(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        self.couple_acoustic_to_optical(fields, dt)?;
        self.couple_optical_to_thermal(fields, dt)?;
        self.couple_acoustic_to_thermal(fields, dt)?;
        Ok(())
    }

    /// Apply strong coupling (iterative)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn apply_strong_coupling(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        let mut previous_fields = fields.to_vec();

        for iteration in 0..self.max_iterations {
            self.apply_weak_coupling(fields, dt)?;

            if self.check_convergence(&previous_fields, fields) {
                break;
            }

            previous_fields = fields.to_vec();

            if iteration > 0 {
                self.apply_relaxation(&previous_fields, fields);
            }
        }

        Ok(())
    }

    /// Apply adaptive coupling
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_adaptive_coupling(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        let gradients = self.calculate_field_gradients(fields);
        let coupling_strength = self.adjust_coupling_strength(&gradients);
        self.apply_coupling_with_strength(fields, dt, coupling_strength)
    }

    /// Couple acoustic field to optical field (photoelastic effect).
    fn couple_acoustic_to_optical(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        let pressure = fields[PRESSURE_IDX].clone();
        let intensity = &mut fields[LIGHT_IDX];

        for ((i, j, k), &p) in pressure.indexed_iter() {
            let delta_n = 1e-12 * p;
            let modulation = (self.coupling_strength * delta_n).mul_add(dt, 1.0);
            intensity[[i, j, k]] *= modulation;
        }

        Ok(())
    }

    /// Couple optical field to thermal field (absorption heating).
    fn couple_optical_to_thermal(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        let intensity = fields[LIGHT_IDX].clone();
        let temperature = &mut fields[TEMPERATURE_IDX];

        let absorption_coefficient = 10.0; // 10 m⁻¹ (typical for tissue)

        for ((i, j, k), &i_val) in intensity.indexed_iter() {
            let heat_source = absorption_coefficient * i_val;
            let delta_t = heat_source * dt / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
            temperature[[i, j, k]] += delta_t;
        }

        Ok(())
    }

    /// Couple acoustic field to thermal field (absorption heating).
    fn couple_acoustic_to_thermal(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        let pressure = fields[PRESSURE_IDX].clone();
        let temperature = &mut fields[TEMPERATURE_IDX];

        let absorption_coefficient = 0.5; // 0.5 Np/m (typical for tissue)

        for ((i, j, k), &p) in pressure.indexed_iter() {
            let intensity = p * p / (DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE);
            let heat_source = absorption_coefficient * intensity;
            let delta_t = heat_source * dt / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
            temperature[[i, j, k]] += delta_t;
        }

        Ok(())
    }

    /// Check for convergence using relative tolerance.
    ///
    /// Computes the maximum relative change between iterations:
    ///   ε_rel = max |current − previous| / (‖current‖_∞ + 1e-15)
    pub(super) fn check_convergence(
        &self,
        previous: &[Array3<f64>],
        current: &[Array3<f64>],
    ) -> bool {
        for (prev_field, curr_field) in previous.iter().zip(current.iter()) {
            let field_norm = curr_field.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

            let max_rel_diff = prev_field
                .iter()
                .zip(curr_field.iter())
                .map(|(p, c)| (p - c).abs() / (field_norm + 1e-15))
                .fold(0.0_f64, f64::max);

            if max_rel_diff > self.tolerance {
                return false;
            }
        }
        true
    }

    /// Apply relaxation (omega = 0.5) for iterative stability.
    fn apply_relaxation(&self, previous: &[Array3<f64>], current: &mut [Array3<f64>]) {
        let omega = 0.5;
        for (prev_field, curr_field) in previous.iter().zip(current.iter_mut()) {
            for ((i, j, k), &prev_val) in prev_field.indexed_iter() {
                let curr_val = curr_field[[i, j, k]];
                curr_field[[i, j, k]] = omega * curr_val + (1.0 - omega) * prev_val;
            }
        }
    }

    /// Calculate maximum gradient magnitude per field.
    fn calculate_field_gradients(&self, fields: &[Array3<f64>]) -> Vec<f64> {
        fields
            .iter()
            .map(|field| {
                let mut max_gradient: f64 = 0.0;
                for i in 1..field.shape()[0] - 1 {
                    for j in 1..field.shape()[1] - 1 {
                        for k in 1..field.shape()[2] - 1 {
                            let grad_x = field[[i + 1, j, k]] - field[[i - 1, j, k]];
                            let grad_y = field[[i, j + 1, k]] - field[[i, j - 1, k]];
                            let grad_z = field[[i, j, k + 1]] - field[[i, j, k - 1]];
                            let gradient = grad_z
                                .mul_add(grad_z, grad_x.mul_add(grad_x, grad_y * grad_y))
                                .sqrt();
                            max_gradient = max_gradient.max(gradient);
                        }
                    }
                }
                max_gradient
            })
            .collect()
    }

    /// Adjust coupling strength based on gradient magnitudes.
    fn adjust_coupling_strength(&self, gradients: &[f64]) -> f64 {
        let max_gradient = gradients.iter().fold(0.0, |max, &g| g.max(max));
        if max_gradient > 1.0 {
            0.1
        } else if max_gradient > 0.1 {
            0.5
        } else {
            1.0
        }
    }

    /// Apply coupling with a specified strength override.
    fn apply_coupling_with_strength(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
        strength: f64,
    ) -> KwaversResult<()> {
        let temp_coupler = Self {
            strategy: self.strategy,
            coupling_strength: strength,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        };
        temp_coupler.apply_weak_coupling(fields, dt)
    }
}
