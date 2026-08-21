use super::MultiphysicsFieldCoupler;
use kwavers_core::constants::fundamental::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, OPTICAL_ABSORPTION_TISSUE_NIR,
    SOUND_SPEED_TISSUE,
};
use kwavers_core::constants::thermodynamic::SPECIFIC_HEAT_WATER;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::indices::{LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
use leto::Array3;

impl MultiphysicsFieldCoupler {
    /// Apply weak coupling (single pass)
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(super) fn apply_weak_coupling(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        validate_coupled_field_set(fields)?;
        self.couple_acoustic_to_optical(fields, dt)?;
        Self::couple_optical_to_thermal(fields, dt)?;
        Self::couple_acoustic_to_thermal(fields, dt)?;
        Ok(())
    }

    /// Apply strong coupling (iterative)
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(super) fn apply_strong_coupling(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
    ) -> KwaversResult<()> {
        validate_coupled_field_set(fields)?;
        let mut previous_fields = fields.to_vec();

        for iteration in 0..self.max_iterations {
            self.apply_weak_coupling(fields, dt)?;

            if self.check_convergence(&previous_fields, fields) {
                break;
            }

            copy_fields_into(&mut previous_fields, fields);

            if iteration > 0 {
                Self::apply_relaxation(&previous_fields, fields);
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
        validate_coupled_field_set(fields)?;
        let gradients = Self::calculate_field_gradients(fields);
        let coupling_strength = Self::adjust_coupling_strength(&gradients);
        self.apply_coupling_with_strength(fields, dt, coupling_strength)
    }

    /// Couple acoustic field to optical field (photoelastic effect).
    ///
    /// Refractive index modulation: Δn = dn/dp · p, where dn/dp ≈ 1.5×10⁻¹⁰ Pa⁻¹
    /// for water (Schmid et al. 2012, "Photoacoustic sound generation in water
    /// droplets", Appl. Phys. Lett. 100:014105). The field-coupler contract
    /// accepts collocated fields but no medium-property provider, so this path
    /// uses the documented nominal coefficient. [`AcousticOpticalSolver`]
    /// carries a caller-supplied coefficient for medium-specific coupling.
    fn couple_acoustic_to_optical(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        let (pressure, intensity) = read_write_fields::<PRESSURE_IDX, LIGHT_IDX>(fields)?;

        // The generic field-coupler API has no medium-property input.
        const DN_DP: f64 = 1e-12; // documented nominal water coefficient

        for ([i, j, k], &p) in pressure.indexed_iter() {
            let delta_n = DN_DP * p;
            let modulation = (self.coupling_strength * delta_n).mul_add(dt, 1.0);
            intensity[[i, j, k]] *= modulation;
        }

        Ok(())
    }

    /// Couple optical field to thermal field (absorption heating).
    ///
    /// Heat source: Q = μ_a · I, where μ_a is the optical absorption coefficient
    /// and I is the optical intensity (fluence rate). Uses
    /// [`OPTICAL_ABSORPTION_TISSUE_NIR`] (10 m⁻¹, typical for soft tissue in the
    /// NIR window, Jacques 2013). The generic field-coupler API has no
    /// per-voxel optical-property input, so it applies this nominal tissue
    /// coefficient.
    fn couple_optical_to_thermal(fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        let (intensity, temperature) = read_write_fields::<LIGHT_IDX, TEMPERATURE_IDX>(fields)?;

        for ([i, j, k], &i_val) in intensity.indexed_iter() {
            let heat_source = OPTICAL_ABSORPTION_TISSUE_NIR * i_val;
            let delta_t = heat_source * dt / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
            temperature[[i, j, k]] += delta_t;
        }

        Ok(())
    }

    /// Couple acoustic field to thermal field (absorption heating).
    ///
    /// Acoustic intensity: I = p² / (2ρc) (Morton & Ter Haar 1998).
    /// Heat source: Q = α · I, where α is the acoustic absorption coefficient.
    /// Uses [`ACOUSTIC_ABSORPTION_TISSUE`] (0.5 dB/(cm·MHz), Duck 1990) as a
    /// generic tissue default. The generic field-coupler API has no
    /// frequency-dependent medium-property input, so it applies this nominal
    /// tissue coefficient.
    fn couple_acoustic_to_thermal(fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        let (pressure, temperature) = read_write_fields::<PRESSURE_IDX, TEMPERATURE_IDX>(fields)?;

        const TWO: f64 = 2.0;
        let impedance = DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE;

        for ([i, j, k], &p) in pressure.indexed_iter() {
            let intensity = p * p / (TWO * impedance);
            let heat_source = ACOUSTIC_ABSORPTION_TISSUE * intensity;
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
    fn apply_relaxation(previous: &[Array3<f64>], current: &mut [Array3<f64>]) {
        let omega = 0.5;
        for (prev_field, curr_field) in previous.iter().zip(current.iter_mut()) {
            for ([i, j, k], &prev_val) in prev_field.indexed_iter() {
                let curr_val = curr_field[[i, j, k]];
                curr_field[[i, j, k]] = omega * curr_val + (1.0 - omega) * prev_val;
            }
        }
    }

    /// Calculate maximum gradient magnitude per field.
    fn calculate_field_gradients(fields: &[Array3<f64>]) -> Vec<f64> {
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
    fn adjust_coupling_strength(gradients: &[f64]) -> f64 {
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

fn validate_coupled_field_set(fields: &[Array3<f64>]) -> KwaversResult<()> {
    validate_field_index::<PRESSURE_IDX>(fields.len())?;
    validate_field_index::<TEMPERATURE_IDX>(fields.len())?;
    validate_field_index::<LIGHT_IDX>(fields.len())?;
    validate_coupled_shapes::<PRESSURE_IDX, TEMPERATURE_IDX>(
        &fields[PRESSURE_IDX],
        &fields[TEMPERATURE_IDX],
    )?;
    validate_coupled_shapes::<PRESSURE_IDX, LIGHT_IDX>(&fields[PRESSURE_IDX], &fields[LIGHT_IDX])
}

/// Copy the current iteration state into existing snapshot buffers.
///
/// Strong coupling needs one previous-state volume per field for convergence
/// testing. Reusing those volumes avoids one `Vec<Array3<_>>` allocation and
/// one owned array allocation per non-converged iteration.
fn copy_fields_into(target: &mut [Array3<f64>], source: &[Array3<f64>]) {
    for (target_field, source_field) in target.iter_mut().zip(source.iter()) {
        target_field.assign(source_field);
    }
}

/// Borrow one read-only field and one mutable field without cloning volumes.
///
/// `READ` and `WRITE` are structural field-index parameters. The compiler
/// specializes the split path for each coupling edge, while the implementation
/// keeps one authoritative disjoint-borrow and shape-validation contract.
fn read_write_fields<const READ: usize, const WRITE: usize>(
    fields: &mut [Array3<f64>],
) -> KwaversResult<(&Array3<f64>, &mut Array3<f64>)> {
    validate_field_index::<READ>(fields.len())?;
    validate_field_index::<WRITE>(fields.len())?;
    if READ == WRITE {
        return Err(KwaversError::InvalidInput(format!(
            "MultiphysicsFieldCoupler requires distinct read/write indices, got {READ}"
        )));
    }

    let (read, write) = if READ < WRITE {
        let (left, right) = fields.split_at_mut(WRITE);
        (&left[READ], &mut right[0])
    } else {
        let (left, right) = fields.split_at_mut(READ);
        (&right[0], &mut left[WRITE])
    };

    validate_coupled_shapes::<READ, WRITE>(read, write)?;
    Ok((read, write))
}

fn validate_field_index<const INDEX: usize>(len: usize) -> KwaversResult<()> {
    if INDEX >= len {
        return Err(KwaversError::InvalidInput(format!(
            "MultiphysicsFieldCoupler requires field index {INDEX}, but only {len} fields were provided"
        )));
    }
    Ok(())
}

fn validate_coupled_shapes<const READ: usize, const WRITE: usize>(
    read: &Array3<f64>,
    write: &Array3<f64>,
) -> KwaversResult<()> {
    if read.shape() != write.shape() {
        return Err(KwaversError::DimensionMismatch(format!(
            "MultiphysicsFieldCoupler edge {READ}->{WRITE} requires matching shapes, got read {:?} and write {:?}",
            read.shape(),
            write.shape()
        )));
    }
    Ok(())
}
