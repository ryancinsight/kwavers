use std::collections::HashMap;

use leto::Array3;

use crate::acoustics::bubble_dynamics::adaptive_integration::integrate_bubble_dynamics_adaptive;

use super::model::{BubbleField, BubbleFieldSolver};

impl BubbleField {
    /// Advance all bubbles by one time step.
    ///
    /// Each bubble is integrated with local acoustic pressure plus secondary
    /// Bjerknes pressure from all other bubbles, evaluated explicitly at t^n.
    pub fn update(
        &mut self,
        pressure_field: &Array3<f64>,
        dp_dt_field: &Array3<f64>,
        dt: f64,
        t: f64,
    ) {
        let secondary_pressures: HashMap<(usize, usize, usize), f64> = if self.bubbles.len() > 1 {
            self.compute_secondary_pressures()
        } else {
            HashMap::new()
        };

        for (&pos, state) in &mut self.bubbles {
            let (i, j, k) = pos;
            let p_acoustic = pressure_field[[i, j, k]];
            let dp_dt = dp_dt_field[[i, j, k]];
            let p_secondary = secondary_pressures.get(&pos).copied().unwrap_or(0.0);
            let p_effective = p_acoustic + p_secondary;

            let integrate = match &self.solver {
                BubbleFieldSolver::KellerMiksis(model) => {
                    integrate_bubble_dynamics_adaptive(model, state, p_effective, dp_dt, dt, t)
                }
                BubbleFieldSolver::KellerHerring(model) => {
                    integrate_bubble_dynamics_adaptive(model, state, p_effective, dp_dt, dt, t)
                }
                BubbleFieldSolver::RayleighPlesset(model) => {
                    integrate_bubble_dynamics_adaptive(model, state, p_effective, dp_dt, dt, t)
                }
            };

            if let Err(e) = integrate {
                // One bubble's adaptive-step failure must not abort the whole
                // field; log at a meaningful boundary and continue with the
                // remaining bubbles.
                log::warn!("bubble dynamics integration failed at ({i}, {j}, {k}): {e:?}");
            }
        }

        self.record_history(t);
    }

    /// Record time history of bubble states.
    fn record_history(&mut self, t: f64) {
        self.time_history.push(t);

        if self.radius_history.is_empty() {
            for _ in 0..self.bubbles.len() {
                self.radius_history.push(Vec::new());
                self.temperature_history.push(Vec::new());
            }
        }

        for (idx, (_, state)) in self.bubbles.iter().enumerate() {
            if idx < self.radius_history.len() {
                self.radius_history[idx].push(state.radius);
                self.temperature_history[idx].push(state.temperature);
            }
        }
    }
}
