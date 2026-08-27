//! Host medium preparation for GPU PSTD execution.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_medium::{core::MIN_PHYSICAL_SOUND_SPEED, Medium};

const POWER_LAW_SENTINEL_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug)]
pub(super) enum PreparedAbsorptionCoefficients {
    Uniform(f64),
    Spatial(Vec<f64>),
}

impl PreparedAbsorptionCoefficients {
    pub(super) fn at(&self, index: usize) -> f64 {
        match self {
            Self::Uniform(coefficient) => *coefficient,
            Self::Spatial(coefficients) => coefficients[index],
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            Self::Uniform(_) => 0,
            Self::Spatial(coefficients) => coefficients
                .capacity()
                .saturating_mul(std::mem::size_of::<f64>()),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedAbsorption {
    pub(super) exponent: f64,
    pub(super) coefficients: PreparedAbsorptionCoefficients,
}

struct PowerLawResolver {
    fallback_exponent: f64,
    exponent: Option<f64>,
    coefficients: Option<Vec<f64>>,
    uniform_coefficient: Option<f64>,
}

impl PowerLawResolver {
    fn new(alpha_coeff_db: f64, alpha_power: f64, total: usize) -> KwaversResult<Self> {
        if !alpha_coeff_db.is_finite() || alpha_coeff_db < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD alpha_coeff_db must be finite and non-negative; got {alpha_coeff_db}"
            )));
        }
        if alpha_coeff_db > 0.0 {
            validate_active_power_law_exponent(alpha_power)?;
            return Ok(Self {
                fallback_exponent: alpha_power,
                exponent: Some(alpha_power),
                coefficients: None,
                uniform_coefficient: Some(alpha_coeff_db),
            });
        }
        Ok(Self {
            fallback_exponent: alpha_power,
            exponent: None,
            coefficients: Some(Vec::with_capacity(total)),
            uniform_coefficient: None,
        })
    }

    fn uses_medium(&self) -> bool {
        self.coefficients.is_some()
    }

    fn push_medium_pair(
        &mut self,
        index: usize,
        coefficient: f64,
        medium_power: f64,
    ) -> KwaversResult<()> {
        let coefficients = self
            .coefficients
            .as_mut()
            .expect("invariant: medium pairs are pushed only for delegated absorption");
        if !coefficient.is_finite() || coefficient < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD medium alpha coefficient must be finite and non-negative at flat index {index}; got {coefficient}"
            )));
        }
        coefficients.push(coefficient);
        if coefficient == 0.0 {
            return Ok(());
        }
        let effective_power = if medium_power.abs() > POWER_LAW_SENTINEL_TOLERANCE
            && (medium_power - 1.0).abs() > POWER_LAW_SENTINEL_TOLERANCE
        {
            medium_power
        } else {
            self.fallback_exponent
        };
        validate_active_power_law_exponent(effective_power)?;
        if self
            .exponent
            .is_some_and(|resolved| resolved != effective_power)
        {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "Hephaestus PSTD requires one active absorption exponent; medium flat index {index} uses {effective_power}"
            )));
        }
        self.exponent = Some(effective_power);
        Ok(())
    }

    fn finish(self) -> Option<PreparedAbsorption> {
        self.exponent.map(|exponent| PreparedAbsorption {
            exponent,
            coefficients: self.uniform_coefficient.map_or_else(
                || {
                    PreparedAbsorptionCoefficients::Spatial(
                        self.coefficients
                            .expect("invariant: delegated absorption retains coefficients"),
                    )
                },
                PreparedAbsorptionCoefficients::Uniform,
            ),
        })
    }
}

/// Provider-ready host snapshot of the medium fields consumed by GPU PSTD.
///
/// Construction traverses the medium once, converts uploaded fields to the
/// GPU's native `f32` representation, and resolves the power-law coefficient /
/// exponent ownership contract before CPML preparation or device acquisition.
/// The snapshot can therefore be retained by batch-oriented solver adapters
/// without retaining a borrowed [`Medium`].
#[derive(Clone, Debug)]
pub struct PstdMediumSnapshot {
    pub(super) c0_flat: Vec<f32>,
    pub(super) rho0_flat: Vec<f32>,
    pub(super) bon_a_flat: Vec<f32>,
    pub(super) c_ref: f64,
    pub(super) absorption: Option<PreparedAbsorption>,
    alpha_coeff_db: f64,
    alpha_power: f64,
}

impl PstdMediumSnapshot {
    /// Capture and validate the medium fields required by GPU PSTD.
    ///
    /// A positive `alpha_coeff_db` and `alpha_power` form one uniform
    /// absorption pair that overrides the medium. Zero coefficient delegates
    /// both values to the medium; all active voxels must then resolve to one
    /// exponent because the GPU solver owns one fractional-Laplacian symbol.
    ///
    /// # Errors
    ///
    /// Returns an error when the grid exceeds GPU addressing limits, an
    /// absorption coefficient is invalid, an enabled exponent is singular, or
    /// active medium voxels use heterogeneous exponents.
    pub fn from_medium<M: Medium + ?Sized>(
        grid: &Grid,
        medium: &M,
        alpha_coeff_db: f64,
        alpha_power: f64,
    ) -> KwaversResult<Self> {
        let total = super::validate_pstd_grid_shape(grid).map_err(KwaversError::InvalidInput)?;
        let mut c0_flat = Vec::with_capacity(total);
        let mut rho0_flat = Vec::with_capacity(total);
        let mut bon_a_flat = Vec::with_capacity(total);
        let mut absorption = PowerLawResolver::new(alpha_coeff_db, alpha_power, total)?;
        let mut c_ref = MIN_PHYSICAL_SOUND_SPEED;

        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                for iz in 0..grid.nz {
                    let flat = c0_flat.len();
                    let sound_speed = medium.sound_speed(ix, iy, iz);
                    c_ref = c_ref.max(sound_speed);
                    c0_flat.push(sound_speed as f32);
                    rho0_flat.push(medium.density(ix, iy, iz) as f32);
                    bon_a_flat.push((medium.nonlinearity(ix, iy, iz) / 2.0) as f32);
                    if absorption.uses_medium() {
                        let (x, y, z) = grid.indices_to_coordinates(ix, iy, iz);
                        absorption.push_medium_pair(
                            flat,
                            medium.alpha_coefficient(x, y, z, grid),
                            medium.alpha_power(x, y, z, grid),
                        )?;
                    }
                }
            }
        }

        Ok(Self {
            c0_flat,
            rho0_flat,
            bon_a_flat,
            c_ref,
            absorption: absorption.finish(),
            alpha_coeff_db,
            alpha_power,
        })
    }

    /// Return the heap bytes retained by the packed medium fields.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.c0_flat
            .capacity()
            .saturating_add(self.rho0_flat.capacity())
            .saturating_add(self.bon_a_flat.capacity())
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(
                self.absorption
                    .as_ref()
                    .map_or(0, |absorption| absorption.coefficients.resident_bytes()),
            )
    }

    pub(super) fn validate_absorption_config(
        &self,
        alpha_coeff_db: f64,
        alpha_power: f64,
    ) -> KwaversResult<()> {
        if self.alpha_coeff_db != alpha_coeff_db || self.alpha_power != alpha_power {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD medium snapshot was prepared for absorption pair ({}, {}) but the run requested ({alpha_coeff_db}, {alpha_power})",
                self.alpha_coeff_db, self.alpha_power
            )));
        }
        Ok(())
    }
}

fn validate_active_power_law_exponent(alpha_power: f64) -> KwaversResult<()> {
    if !alpha_power.is_finite() || (alpha_power - 1.0).abs() < POWER_LAW_SENTINEL_TOLERANCE {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD alpha_power must be finite and must not equal 1.0 for enabled fractional absorption; got {alpha_power}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct PowerLawAbsorption {
    exponent: f64,
    explicit_coefficient: Option<f64>,
}

#[cfg(test)]
fn resolve_power_law_absorption(
    alpha_coeff_db: f64,
    alpha_power: f64,
    medium_pairs: impl IntoIterator<Item = (f64, f64)>,
) -> KwaversResult<Option<PowerLawAbsorption>> {
    let pairs = medium_pairs.into_iter();
    let mut resolver = PowerLawResolver::new(alpha_coeff_db, alpha_power, pairs.size_hint().0)?;
    if resolver.uses_medium() {
        for (index, (coefficient, medium_power)) in pairs.enumerate() {
            resolver.push_medium_pair(index, coefficient, medium_power)?;
        }
    }
    Ok(resolver.finish().map(|absorption| PowerLawAbsorption {
        exponent: absorption.exponent,
        explicit_coefficient: match absorption.coefficients {
            PreparedAbsorptionCoefficients::Uniform(coefficient) => Some(coefficient),
            PreparedAbsorptionCoefficients::Spatial(_) => None,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::{resolve_power_law_absorption, PowerLawAbsorption};

    #[test]
    fn zero_explicit_and_medium_absorption_is_lossless() {
        assert_eq!(
            resolve_power_law_absorption(0.0, 1.0, [(0.0, 1.0)])
                .expect("inactive medium coefficients do not enable absorption"),
            None
        );
    }

    #[test]
    fn explicit_absorption_pair_overrides_medium_pair() {
        assert_eq!(
            resolve_power_law_absorption(0.75, 1.5, [(0.4, 1.2)])
                .expect("valid explicit absorption pair"),
            Some(PowerLawAbsorption {
                exponent: 1.5,
                explicit_coefficient: Some(0.75),
            })
        );
    }

    #[test]
    fn medium_absorption_resolves_one_uniform_active_exponent() {
        assert_eq!(
            resolve_power_law_absorption(0.0, 1.5, [(0.2, 1.3), (0.0, 1.7), (0.4, 1.3)])
                .expect("uniform active medium exponent"),
            Some(PowerLawAbsorption {
                exponent: 1.3,
                explicit_coefficient: None,
            })
        );
    }

    #[test]
    fn medium_absorption_sentinel_uses_configuration_fallback() {
        assert_eq!(
            resolve_power_law_absorption(0.0, 1.5, [(0.2, 1.0)])
                .expect("sentinel exponent delegates to configuration"),
            Some(PowerLawAbsorption {
                exponent: 1.5,
                explicit_coefficient: None,
            })
        );
    }

    #[test]
    fn heterogeneous_medium_absorption_exponents_are_rejected() {
        let error = resolve_power_law_absorption(0.0, 1.5, [(0.2, 1.3), (0.4, 1.7)])
            .expect_err("one GPU spectral symbol cannot represent multiple exponents");
        assert_eq!(
            error.to_string(),
            "Feature not available: Hephaestus PSTD requires one active absorption exponent; medium flat index 1 uses 1.7"
        );
    }

    #[test]
    fn enabled_absorption_rejects_singular_power_law_exponent() {
        let error = resolve_power_law_absorption(0.0022, 1.0, [])
            .expect_err("enabled fractional absorption cannot use y=1");
        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD alpha_power must be finite and must not equal 1.0 for enabled fractional absorption; got 1"
        );
    }
}
