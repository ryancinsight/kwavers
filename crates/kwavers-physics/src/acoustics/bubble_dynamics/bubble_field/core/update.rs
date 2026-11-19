use std::collections::HashMap;

use leto::Array3;

use crate::acoustics::bubble_dynamics::{
    acoustic_pressure::InstantaneousPressure,
    adaptive_integration::integrate_bubble_dynamics_adaptive,
};
use aequitas::systems::si::quantities::PressureRate;

use super::model::{BubbleField, BubbleFieldSolver};

impl BubbleField {
    /// Advance all bubbles by one time step.
    ///
    /// Each bubble is integrated with local acoustic pressure plus secondary
    /// Bjerknes pressure from all other bubbles, evaluated explicitly at t^n.
    ///
    /// The pressures passed here are INSTANTANEOUS field values
    /// (`p_∞(t) − p0` and its time derivative), as the Keller–Miksis /
    /// Keller–Herring models require (see the drive contract on
    /// `AdaptiveBubbleModel::calculate_acceleration`). A wave-solver field
    /// is a broadband time series, not a single-frequency tone, so there is
    /// no amplitude to extract — the instantaneous form is the only one
    /// that can represent it, and the secondary Bjerknes pressures are
    /// instantaneous by construction.
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
            // Dense field arrays stay scalar numerical storage (ADR 070);
            // each sampled value is wrapped at the model boundary.
            let p_acoustic = InstantaneousPressure::from_pascals(pressure_field[[i, j, k]]);
            let dp_dt = PressureRate::from_base(dp_dt_field[[i, j, k]]);
            let p_secondary = InstantaneousPressure::from_pascals(
                secondary_pressures.get(&pos).copied().unwrap_or(0.0),
            );
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
