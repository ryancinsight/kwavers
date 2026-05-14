use std::collections::HashMap;

use super::model::BubbleField;

impl BubbleField {
    /// Compute the secondary Bjerknes pressure contribution at every bubble
    /// position due to the instantaneous radiation of all other bubbles.
    ///
    /// ```text
    /// p_secondary_j = Σ_{i≠j}  − ρ_L [R_i² R̈_i + 2 R_i Ṙ_i²] / d_ij
    /// ```
    ///
    /// Pairs with `d_ij = 0` and pairs where `R_i / d_ij < coupling_threshold`
    /// are skipped.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub(super) fn compute_secondary_pressures(&self) -> HashMap<(usize, usize, usize), f64> {
        let positions: Vec<(usize, usize, usize)> = self.bubbles.keys().copied().collect();
        let mut corrections: HashMap<(usize, usize, usize), f64> =
            positions.iter().map(|&pos| (pos, 0.0)).collect();

        let (dx, dy, dz) = self.grid_spacing;

        for (idx_j, &pos_j) in positions.iter().enumerate() {
            for (idx_i, &pos_i) in positions.iter().enumerate() {
                if idx_i == idx_j {
                    continue;
                }

                let state_i = &self.bubbles[&pos_i];
                let delta_x = (pos_i.0 as f64 - pos_j.0 as f64) * dx;
                let delta_y = (pos_i.1 as f64 - pos_j.1 as f64) * dy;
                let delta_z = (pos_i.2 as f64 - pos_j.2 as f64) * dz;
                let d_ij = delta_z
                    .mul_add(delta_z, delta_x.mul_add(delta_x, delta_y * delta_y))
                    .sqrt();

                if d_ij == 0.0 || state_i.radius / d_ij < self.coupling_threshold {
                    continue;
                }

                let r = state_i.radius;
                let rdot = state_i.wall_velocity;
                let rddot = state_i.wall_acceleration;
                let p_ij = -self.rho_liquid * (r * r).mul_add(rddot, 2.0 * r * rdot * rdot) / d_ij;

                *corrections.get_mut(&pos_j).unwrap() += p_ij;
            }
        }

        corrections
    }
}
