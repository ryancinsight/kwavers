use super::material::FieldCouplingGains;
use super::MultiphysicsFieldCoupler;
use aequitas::systems::si::quantities::{Intensity, Pressure};
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
        gains: &impl Fn([usize; 3]) -> FieldCouplingGains,
        strength: f64,
    ) -> KwaversResult<()> {
        Self::couple_acoustic_to_optical(fields, gains, strength)?;
        Self::couple_optical_to_thermal(fields, gains)?;
        Self::couple_acoustic_to_thermal(fields, gains)?;
        Ok(())
    }

    /// Apply strong coupling (iterative)
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(super) fn apply_strong_coupling(
        &self,
        fields: &mut [Array3<f64>],
        gains: &impl Fn([usize; 3]) -> FieldCouplingGains,
    ) -> KwaversResult<()> {
        let mut previous_fields = fields.to_vec();

        for iteration in 0..self.max_iterations {
            self.apply_weak_coupling(fields, gains, self.coupling_strength)?;

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
        gains: &impl Fn([usize; 3]) -> FieldCouplingGains,
    ) -> KwaversResult<()> {
        let gradients = Self::calculate_field_gradients(fields);
        let coupling_strength = Self::adjust_coupling_strength(&gradients);
        self.apply_weak_coupling(fields, gains, coupling_strength)
    }

    /// Scale optical intensity by the pressure-driven refractive-index response.
    fn couple_acoustic_to_optical(
        fields: &mut [Array3<f64>],
        gains: &impl Fn([usize; 3]) -> FieldCouplingGains,
        strength: f64,
    ) -> KwaversResult<()> {
        let (pressure, intensity) = read_write_fields::<PRESSURE_IDX, LIGHT_IDX>(fields)?;
        for (cell, &p) in pressure.indexed_iter() {
            let response = gains(cell).photoelastic * Pressure::from_base(p);
            intensity[cell] *= strength.mul_add(response.into_base(), 1.0);
        }
        Ok(())
    }

    /// Apply optical absorption heating over the step duration.
    fn couple_optical_to_thermal(
        fields: &mut [Array3<f64>],
        gains: &impl Fn([usize; 3]) -> FieldCouplingGains,
    ) -> KwaversResult<()> {
        let (intensity, temperature) = read_write_fields::<LIGHT_IDX, TEMPERATURE_IDX>(fields)?;
        for (cell, &value) in intensity.indexed_iter() {
            temperature[cell] += (gains(cell).optical * Intensity::from_base(value)).into_base();
        }
        Ok(())
    }

    /// Apply acoustic intensity absorption with peak pressure p and I = p²/(2ρc).
    fn couple_acoustic_to_thermal(
        fields: &mut [Array3<f64>],
        gains: &impl Fn([usize; 3]) -> FieldCouplingGains,
    ) -> KwaversResult<()> {
        let (pressure, temperature) = read_write_fields::<PRESSURE_IDX, TEMPERATURE_IDX>(fields)?;
        for (cell, &value) in pressure.indexed_iter() {
            let p = Pressure::from_base(value);
            temperature[cell] += (gains(cell).acoustic * (p * p)).into_base();
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
                let [nx, ny, nz] = field.shape();
                for i in 1..nx.saturating_sub(1) {
                    for j in 1..ny.saturating_sub(1) {
                        for k in 1..nz.saturating_sub(1) {
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
}

pub(super) fn validate_coupled_field_set(fields: &[Array3<f64>]) -> KwaversResult<()> {
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
