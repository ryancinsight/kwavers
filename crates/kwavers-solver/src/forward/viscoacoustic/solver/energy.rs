use crate::forward::viscoacoustic::solver::Coeff;
use crate::forward::viscoacoustic::solver::ViscoacousticMemorySolver;

impl ViscoacousticMemorySolver {
    /// Acoustic energy `Σ [p²/(2M_∞) + ρ|v|²/2] ΔV` \`J`. Conserved (to leapfrog
    /// round-off) for the lossless medium; decays monotonically with relaxation.
    #[must_use]
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal solver state violates
    /// the precondition required by this operation.
    pub fn energy(&self) -> f64 {
        // PE = Σ p²/(2 M_∞(x));  KE = Σ ρ(x)|v|²/2 = Σ |v|²/(2/ρ) = Σ |v|²·inv_rho⁻¹/2.
        // The uniform specialization evaluates the same terms in the same
        // order as the grid fold over a constant field, so the two agree
        // bitwise.
        let pe = match &self.m_inf {
            Coeff::Uniform(mi) => self
                .p
                .iter()
                .fold(0.0_f64, |acc, &p| acc + p * p / (2.0 * mi)),
            Coeff::Field(m_inf) => {
                leto_ops::zip_fold(&self.p.view(), &m_inf.view(), 0.0_f64, |acc, &p, &mi| {
                    acc + p * p / (2.0 * mi)
                })
                .expect("invariant: pressure and modulus fields share grid shape")
            }
        };
        // Inactive-axis velocity components are exactly zero for all time, so
        // their kinetic contribution is the zero term and is not traversed.
        let vx_iter = self.vx.iter().copied().chain(std::iter::repeat(0.0_f64));
        let vy_iter = self.vy.iter().copied().chain(std::iter::repeat(0.0_f64));
        let vz_iter = self.vz.iter().copied().chain(std::iter::repeat(0.0_f64));
        let ke = match &self.inv_rho {
            // The uniform branch is bounded by the grid-shaped pressure field:
            // exactly the voxel count the grid branch's `inv_rho` iteration
            // would supply, in the same order.
            Coeff::Uniform(ir) => vx_iter
                .zip(vy_iter)
                .zip(vz_iter)
                .zip(self.p.iter())
                .fold(0.0_f64, |acc, (((vx, vy), vz), _)| {
                    acc + (vx * vx + vy * vy + vz * vz) / (2.0 * ir)
                }),
            Coeff::Field(inv_rho) => vx_iter
                .zip(vy_iter)
                .zip(vz_iter)
                .zip(inv_rho.iter())
                .fold(0.0_f64, |acc, (((vx, vy), vz), &ir)| {
                    acc + (vx * vx + vy * vy + vz * vz) / (2.0 * ir)
                }),
        };
        (pe + ke) * self.cell_volume
    }
}
