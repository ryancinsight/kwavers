//! The Newton Jacobian as an Athena linear operator.
//!
//! Athena drives a Krylov solve through [`LinearOperator`], which sees only
//! `y = A x` over flat backend vectors. The monolithic Newton system's `A` is
//! the Jacobian, and kwavers never forms it: it is applied by finite difference
//! on the coupled residual
//! ([`MonolithicCoupler::jacobian_vector_product`]). This module is the seam
//! between the two — the flat Krylov vector and the stacked field geometry the
//! residual evaluates on are the same `n` values in the same order, so the
//! bridge is a copy into and out of a reused staging buffer, never a reshape of
//! the physics.

use super::super::coupler::MonolithicCoupler;
use athena_core::{KrylovBackend, LinearOperator};
use athena_leto::{LetoBackend, LetoBackendError};
use kwavers_core::error::KwaversError;
use kwavers_field::UnifiedFieldType;
use leto::Array3;
use std::cell::RefCell;

/// The CPU backend the monolithic Newton solve runs on.
type Backend = LetoBackend<f64>;

/// One Newton step's Jacobian, applied by finite difference on the residual.
///
/// The operator borrows the linearisation point rather than owning it: it lives
/// for exactly one Newton iteration, during which `state` and `previous_state`
/// are fixed.
pub(in crate::multiphysics::monolithic) struct JacobianOperator<'a> {
    coupler: &'a MonolithicCoupler,
    state: &'a Array3<f64>,
    previous_state: &'a Array3<f64>,
    dt: f64,
    dims: (usize, usize, usize),
    field_order: &'a [UnifiedFieldType],
    /// Staging buffer holding the flat Krylov direction in stacked-field shape.
    ///
    /// Reused across every application in the Newton step, so the Krylov basis
    /// costs no allocation per operator application.
    direction: RefCell<Array3<f64>>,
    /// The typed residual failure raised by the most recent application.
    ///
    /// [`LinearOperator::apply`] may only report the backend's own error type,
    /// which cannot carry a [`KwaversError`]. Dropping the physics failure and
    /// reporting a generic backend error would lose the reason the residual
    /// could not be evaluated, so the typed error is kept here and reclaimed by
    /// [`Self::take_failure`] once the solve returns.
    failure: RefCell<Option<KwaversError>>,
}

impl<'a> JacobianOperator<'a> {
    /// Bind the Jacobian at one Newton linearisation point.
    pub(in crate::multiphysics::monolithic) fn new(
        coupler: &'a MonolithicCoupler,
        state: &'a Array3<f64>,
        previous_state: &'a Array3<f64>,
        dt: f64,
        dims: (usize, usize, usize),
        field_order: &'a [UnifiedFieldType],
    ) -> Self {
        Self {
            coupler,
            state,
            previous_state,
            dt,
            dims,
            field_order,
            direction: RefCell::new(Array3::zeros(state.shape())),
            failure: RefCell::new(None),
        }
    }

    /// Reclaim the typed residual failure, if the solve hit one.
    ///
    /// Athena reports the solve's own outcome; this reports why the physics
    /// underneath it could not be evaluated. A caller checks it before
    /// accepting the returned iterate.
    pub(in crate::multiphysics::monolithic) fn take_failure(&self) -> Option<KwaversError> {
        self.failure.borrow_mut().take()
    }
}

impl LinearOperator<Backend> for JacobianOperator<'_> {
    fn dimension(&self) -> usize {
        let [rows, columns, planes] = self.state.shape();
        rows * columns * planes
    }

    fn apply(
        &self,
        _backend: &Backend,
        input: <Backend as KrylovBackend>::View<'_>,
        mut output: <Backend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let input = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;

        // The residual evaluator reaches the coupler's own scratch, never this
        // operator, so nothing inside the borrow can re-enter and conflict.
        let mut direction = self.direction.borrow_mut();
        let staged = direction
            .as_slice_mut()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if staged.len() != input.len() {
            return Err(LetoBackendError::LengthMismatch {
                left: input.len(),
                right: staged.len(),
            });
        }
        staged.copy_from_slice(input);

        let product = self
            .coupler
            .jacobian_vector_product(
                &direction,
                self.state,
                self.previous_state,
                self.dt,
                self.dims,
                self.field_order,
            )
            .map_err(|error| {
                let message = error.to_string();
                *self.failure.borrow_mut() = Some(error);
                LetoBackendError::Leto(leto::LetoError::InvalidInput(format!(
                    "monolithic Jacobian-vector product failed: {message}"
                )))
            })?;

        let image = product
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if image.len() != output.len() {
            return Err(LetoBackendError::LengthMismatch {
                left: image.len(),
                right: output.len(),
            });
        }
        output.copy_from_slice(image);
        Ok(())
    }
}
