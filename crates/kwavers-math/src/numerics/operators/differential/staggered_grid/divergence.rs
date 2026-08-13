//! Yee-staggered divergence: the **adjoint** of the forward difference.
//!
//! # Why this is not `apply_backward_*`
//!
//! Both compute `(f[i] − f[i−1])/Δ` in the interior. They differ at the low
//! face, where `f[−1]` does not exist, and that difference is the whole point:
//!
//! | | closure at `i = 0` | contract |
//! |---|---|---|
//! | [`StaggeredGridOperator::apply_backward_x_into`] | `(f[1] − f[0])/Δx` — one-sided | a *general* first derivative, consistent (first order) at the boundary for an arbitrary field |
//! | [`StaggeredGridOperator::apply_divergence_x_into`] | `f[0]/Δx` — outside value taken as zero | the *divergence of a face-centred velocity* under a rigid wall |
//!
//! The one-sided closure is the right answer when differentiating a general
//! field. It is the wrong answer for a Yee divergence, and the failure is not a
//! small boundary inaccuracy — it destroys energy conservation over the whole
//! domain.
//!
//! # The adjointness that makes the leapfrog conservative
//!
//! The velocity update applies the forward difference `G`, and the pressure
//! update applies the divergence `D`. A leapfrog built from them conserves
//! discrete energy exactly when
//!
//! ```text
//!   D = −Gᵀ
//! ```
//!
//! i.e. `⟨Gp, u⟩ = −⟨p, Du⟩` for every `p` and every `u` whose far face vanishes.
//! That identity is what makes the update symplectic; without it the scheme has
//! no conserved quantity and the boundary row pumps energy in. Writing it out
//! for `nx = 3` with `u₂ = 0`:
//!
//! ```text
//!   ⟨Gp, u⟩ = [u₀(p₁−p₀) + u₁(p₂−p₁)]/Δx
//!   Du      = [u₀/Δx, (u₁−u₀)/Δx, −u₁/Δx]            (zero-flux closure)
//!   −⟨p, Du⟩ = [u₀(p₁−p₀) + u₁(p₂−p₁)]/Δx            ✓
//! ```
//!
//! With the one-sided closure the first row becomes `(u₁−u₀)/Δx` instead of
//! `u₀/Δx` and the identity fails, which is precisely the defect this module
//! exists to fix (KW-SOL-081): a lossless standing wave grew its energy by
//! nearly five orders of magnitude over two thousand steps.
//!
//! # Boundary condition
//!
//! Taking the out-of-domain face velocity as zero is a **rigid wall**: no flux
//! crosses the low face. It pairs with the velocity update's zeroing of the far
//! face, so the domain is closed on both sides. An absorbing boundary replaces
//! this closure with its own; a PML operates inside the domain and leaves it
//! intact.

use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

use super::operator::StaggeredGridOperator;

impl StaggeredGridOperator {
    /// Divergence contribution along X: `(u[i] − u[i−1])/Δx`, with `u[−1] = 0`.
    ///
    /// `dst` must have shape `(nx, ny, nz)`, matching the pressure grid.
    ///
    /// # Errors
    /// Propagates the insufficient-grid-points error from the underlying
    /// difference.
    pub fn apply_divergence_x_into(
        &self,
        field: ArrayView3<'_, f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.apply_backward_x_into(field, dst)?;
        // Replace the one-sided closure with the zero-flux one. Rewriting the
        // single low-face plane costs O(ny·nz) against the O(nx·ny·nz) pass and
        // keeps one implementation of the interior stencil.
        let [_, ny, nz] = dst.shape();
        // Divide rather than multiply by a reciprocal, matching the interior
        // stencil in `backward.rs`; the two differ by an ulp and the plane is
        // O(ny·nz) against the pass's O(nx·ny·nz), so consistency is free.
        let dx = self.dx;
        for k in 0..nz {
            for j in 0..ny {
                dst[[0, j, k]] = field[[0, j, k]] / dx;
            }
        }
        Ok(())
    }

    /// Divergence contribution along Y: `(u[j] − u[j−1])/Δy`, with `u[−1] = 0`.
    ///
    /// # Errors
    /// Propagates the insufficient-grid-points error from the underlying
    /// difference.
    pub fn apply_divergence_y_into(
        &self,
        field: ArrayView3<'_, f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.apply_backward_y_into(field, dst)?;
        let [nx, _, nz] = dst.shape();
        let dy = self.dy;
        for k in 0..nz {
            for i in 0..nx {
                dst[[i, 0, k]] = field[[i, 0, k]] / dy;
            }
        }
        Ok(())
    }

    /// Divergence contribution along Z: `(u[k] − u[k−1])/Δz`, with `u[−1] = 0`.
    ///
    /// # Errors
    /// Propagates the insufficient-grid-points error from the underlying
    /// difference.
    pub fn apply_divergence_z_into(
        &self,
        field: ArrayView3<'_, f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.apply_backward_z_into(field, dst)?;
        let [nx, ny, _] = dst.shape();
        let dz = self.dz;
        for j in 0..ny {
            for i in 0..nx {
                dst[[i, j, 0]] = field[[i, j, 0]] / dz;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
