//! Thin leto-backed holder for the FDTD spatial-order 2nd/4th/6th central-difference
//! operators.
//!
//! After ADR 0018 the construct wraps a single
//! [`let_ops::FiniteDifference3D<f64>`] rather than the three
//! previously-kwavers-owned `CentralDifference{2,4,6}` structs. The kernel-side
//! stencils now live in leto-ops; the wrapping here preserves the historic
//! `CentralDifferenceOperator::new(order, dx, dy, dz)` signature so velocity/
//! pressure updater callers do not change.

use leto::{Array3, ArrayView3};
use leto_ops::{FiniteDifference3D, FiniteDifference3DScheme};

use kwavers_core::error::{KwaversError, KwaversResult};

#[derive(Debug, Clone)]
pub(crate) struct CentralDifferenceOperator(FiniteDifference3D<f64>);

impl CentralDifferenceOperator {
    /// New.
    ///
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the
    ///   `spatial_order` argument is not in `{2, 4, 6}` or if the
    ///   supplied grid spacings are not strictly positive.
    pub(crate) fn new(order: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        let scheme = match order {
            2 => FiniteDifference3DScheme::CentralSecondOrder,
            4 => FiniteDifference3DScheme::CentralFourthOrder,
            6 => FiniteDifference3DScheme::CentralSixthOrder,
            other => {
                return Err(KwaversError::InvalidInput(format!(
                    "spatial_order must be 2, 4, or 6, got {other}"
                )));
            }
        };
        let op = FiniteDifference3D::<f64>::new(scheme, dx, dy, dz)
            .map_err(|e| KwaversError::InvalidInput(format!("{e}")))?;
        Ok(Self(op))
    }

    /// Apply X-derivative in-place into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation; passes through to the leto-side kernel.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] when the
    ///   differentiated axis has fewer than the minimum required
    ///   points for the chosen central scheme.
    pub(crate) fn apply_x_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.0
            .apply_x_into(field, dst)
            .map_err(|e| KwaversError::InvalidInput(format!("{e}")))
    }

    /// Apply Y-derivative in-place into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation; passes through to the leto-side kernel.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] when the
    ///   differentiated axis has fewer than the minimum required
    ///   points for the chosen central scheme.
    pub(crate) fn apply_y_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.0
            .apply_y_into(field, dst)
            .map_err(|e| KwaversError::InvalidInput(format!("{e}")))
    }

    /// Apply Z-derivative in-place into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation; passes through to the leto-side kernel.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] when the
    ///   differentiated axis has fewer than the minimum required
    ///   points for the chosen central scheme.
    pub(crate) fn apply_z_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.0
            .apply_z_into(field, dst)
            .map_err(|e| KwaversError::InvalidInput(format!("{e}")))
    }
}
