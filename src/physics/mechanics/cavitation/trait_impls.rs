// src/physics/mechanics/cavitation/trait_impls.rs
use super::model::CavitationModel;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::CavitationModelBehavior;
use ndarray::{Array3, Zip};
use log::debug;

impl CavitationModelBehavior for CavitationModel {
    fn update_cavitation(
        &mut self,
        p_update: &mut Array3<f64>,
        p: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) -> Array3<f64> {
        // This is the logic moved from src/physics/mechanics/cavitation/core.rs
        // It calls inherent helper methods like self.calculate_second_derivative, etc.
        // which are assumed to be defined in other submodules (dynamics.rs, effects.rs)
        // as pub(super) or public methods of CavitationModel.

        const LOCAL_MAX_RADIUS: f64 = 2.0e-4;
        const LOCAL_MIN_RADIUS: f64 = 1.0e-10;
        const LOCAL_MAX_VELOCITY: f64 = 1.0e3;
        const LOCAL_MAX_ACCELERATION: f64 = 1.0e12;
        const LOCAL_MAX_PRESSURE: f64 = 5.0e7;

        let mut has_extreme_pressure = false;
        let max_p_abs = p.iter().fold(0.0, |max_abs: f64, &val| max_abs.max(val.abs()));
        if max_p_abs > LOCAL_MAX_PRESSURE {
            has_extreme_pressure = true;
            debug!("Extreme pressure detected: {:.2e} Pa, using adaptive time step for cavitation", max_p_abs);
        }

        let actual_dt = if has_extreme_pressure {
            dt                 * (LOCAL_MAX_PRESSURE / max_p_abs.max(1.0)).clamp(0.01, 0.5)
        } else {
            dt
        };

        // These methods are assumed to be inherent methods of CavitationModel,
        // possibly defined in dynamics.rs or effects.rs and made pub(super) or public.
        self.calculate_second_derivative(p, grid, medium, frequency, actual_dt);

        for val in self.d2r_dt2.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            } else {
                *val = val.clamp(-LOCAL_MAX_ACCELERATION, LOCAL_MAX_ACCELERATION);
            }
        }

        self.update_bubble_dynamics(actual_dt);

        Zip::from(&mut self.radius)
            .and(&mut self.velocity)
            .for_each(|r, v| {
                *r = r.clamp(LOCAL_MIN_RADIUS, LOCAL_MAX_RADIUS);
                *v = v.clamp(-LOCAL_MAX_VELOCITY, LOCAL_MAX_VELOCITY);

                if (*r == LOCAL_MIN_RADIUS && *v < 0.0) || (*r == LOCAL_MAX_RADIUS && *v > 0.0) {
                    *v = 0.0;
                }
            });

        self.calculate_acoustic_effects(p_update, p, grid, medium, has_extreme_pressure)
    }

    fn radius(&self) -> &Array3<f64> {
        // Call the inherent accessor method from model.rs
        self.radius()
    }

    fn velocity(&self) -> &Array3<f64> {
        // Call the inherent accessor method from model.rs
        self.velocity()
    }

    fn temperature(&self) -> &Array3<f64> {
        // Call the inherent accessor method from model.rs
        self.temperature()
    }

    fn set_radius(&mut self, new_radius: &Array3<f64>) {
        // This assumes CavitationModel has a public field `radius` or an inherent `set_radius` method.
        // Based on model.rs, `radius` is pub(crate).
        // For direct field access from another module (trait_impls is a sibling module to model),
        // this won't work if they are in different crates, but within the same crate, pub(crate) is fine.
        // If CavitationModel's fields were private, it would need an inherent `pub(crate) fn set_radius(&mut self, ...)`
        if self.radius.shape() == new_radius.shape() {
            self.radius.assign(new_radius);
        } else {
            // Handle shape mismatch, perhaps log an error or panic,
            // though ideally shapes should always match.
            debug!("Shape mismatch in set_radius for CavitationModel. Current: {:?}, New: {:?}", self.radius.shape(), new_radius.shape());
            // For now, let's attempt assign if they are broadcastable, or just panic/error.
            // Ndarray's assign will panic if shapes are not compatible.
            self.radius.assign(new_radius);
        }
    }
}
