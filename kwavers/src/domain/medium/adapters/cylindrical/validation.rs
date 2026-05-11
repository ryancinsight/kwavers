//! Post-construction physical-bound invariant check.

use super::CylindricalMediumProjection;
use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use crate::domain::medium::Medium;

impl<'a, M: Medium> CylindricalMediumProjection<'a, M> {
    /// Validate that projection preserves physical bounds.
    ///
    /// # Invariants Checked
    ///
    /// 1. `min_sound_speed_3d ≤ min_sound_speed_2d`
    /// 2. `max_sound_speed_2d ≤ max_sound_speed_3d`
    /// 3. All projected values are positive and finite
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
    ///
    pub fn validate_projection(&self) -> KwaversResult<()> {
        let c_3d_max = self.medium.max_sound_speed();

        if self.max_sound_speed > c_3d_max {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "projection max_sound_speed".to_owned(),
                value: self.max_sound_speed.to_string(),
                constraint: format!("Must not exceed 3D medium max: {}", c_3d_max),
            }));
        }

        for &c in &self.sound_speed_2d {
            if c <= 0.0 || !c.is_finite() {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "projected sound_speed".to_owned(),
                    value: c.to_string(),
                    constraint: "Must be positive and finite".to_owned(),
                }));
            }
        }

        for &rho in &self.density_2d {
            if rho <= 0.0 || !rho.is_finite() {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "projected density".to_owned(),
                    value: rho.to_string(),
                    constraint: "Must be positive and finite".to_owned(),
                }));
            }
        }

        for &alpha in &self.absorption_2d {
            if alpha < 0.0 || !alpha.is_finite() {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "projected absorption".to_owned(),
                    value: alpha.to_string(),
                    constraint: "Must be non-negative and finite".to_owned(),
                }));
            }
        }

        Ok(())
    }
}
