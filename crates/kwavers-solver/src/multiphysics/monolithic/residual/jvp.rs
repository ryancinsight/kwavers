use super::super::coupler::MonolithicCoupler;
use super::super::residual_metric::norm;
use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;
use leto::Array3;

impl MonolithicCoupler {
    /// Jacobian-vector product: `J·v ≈ [F(u + εv) − F(u)] / ε`.
    ///
    /// The perturbation state lives in solver-owned scratch storage. The
    /// returned vector reuses the `F(u + εv)` allocation and converts it in
    /// place into the finite-difference quotient, avoiding an additional
    /// full-state scaled-difference allocation.
    ///
    /// # Errors
    /// - Propagates any error returned while evaluating `F(u)` or `F(u+εv)`.
    pub(in crate::multiphysics::monolithic) fn jacobian_vector_product(
        &mut self,
        v: &Array3<f64>,
        u: &Array3<f64>,
        u_prev: &Array3<f64>,
        dt: f64,
        dims: (usize, usize, usize),
        field_order: &[UnifiedFieldType],
    ) -> KwaversResult<Array3<f64>> {
        let eps = 1e-8 * (1.0 + norm(u));
        let f_u = self.compute_residual(u, u_prev, dt, dims, field_order)?;

        let mut u_plus = self
            .jvp_state_scratch
            .take()
            .filter(|scratch| scratch.shape() == u.shape())
            .unwrap_or_else(|| Array3::zeros(u.shape()));
        u_plus.assign(u);
        for (candidate, direction) in u_plus.iter_mut().zip(v.iter()) {
            {
                *candidate += eps * direction;
            };
        }

        let mut f_u_plus = match self.compute_residual(&u_plus, u_prev, dt, dims, field_order) {
            Ok(residual) => residual,
            Err(error) => {
                self.jvp_state_scratch = Some(u_plus);
                return Err(error);
            }
        };
        self.jvp_state_scratch = Some(u_plus);

        let inv_eps = 1.0 / eps;
        for (jv, base) in f_u_plus.iter_mut().zip(f_u.iter()) {
            *jv = (*jv - base) * inv_eps;
        }
        Ok(f_u_plus)
    }
}
