//! Multi-physics solver implementation
//!
//! This module implements a unified solver for coupled acoustic-optical-thermal simulations.

use kwavers_core::error::KwaversResult;
use kwavers_domain::field::indices::{LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX, TOTAL_FIELDS};
use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use crate::multiphysics::field_coupling::{
    FieldCouplingStrategy, MultiphysicsFieldCoupler,
};
use ndarray::Array3;
use std::sync::Arc;

/// Unified multi-physics solver
#[derive(Debug)]
pub struct CoupledMultiPhysicsSolver {
    /// Field coupler for multi-physics interactions
    field_coupler: MultiphysicsFieldCoupler,
    /// Computational grid
    _grid: Grid,
    /// Medium properties
    _medium: Arc<dyn Medium + Send + Sync>,
    /// Multi-physics field array
    fields: Vec<Array3<f64>>,
}

impl CoupledMultiPhysicsSolver {
    /// Create a new multi-physics solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        grid: Grid,
        medium: Arc<dyn Medium + Send + Sync>,
        coupling_strategy: FieldCouplingStrategy,
    ) -> KwaversResult<Self> {
        // Initialize field coupler
        let field_coupler = MultiphysicsFieldCoupler::new(coupling_strategy);

        // Initialize multi-physics fields
        let mut fields = Vec::with_capacity(TOTAL_FIELDS);
        for _ in 0..TOTAL_FIELDS {
            fields.push(Array3::zeros(grid.dimensions()));
        }

        Ok(Self {
            field_coupler,
            _grid: grid,
            _medium: medium,
            fields,
        })
    }

    /// Step the multi-physics simulation
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn step(&mut self, dt: f64) -> KwaversResult<()> {
        // Apply field coupling
        self.field_coupler.couple_fields(&mut self.fields, dt)?;

        Ok(())
    }

    /// Get reference to multi-physics fields
    pub fn fields(&self) -> &[Array3<f64>] {
        &self.fields
    }

    /// Get mutable reference to multi-physics fields
    pub fn fields_mut(&mut self) -> &mut [Array3<f64>] {
        &mut self.fields
    }

    /// Get acoustic pressure field
    pub fn acoustic_pressure(&self) -> &Array3<f64> {
        &self.fields[PRESSURE_IDX]
    }

    /// Get optical intensity field
    pub fn optical_intensity(&self) -> &Array3<f64> {
        &self.fields[LIGHT_IDX]
    }

    /// Get temperature field
    pub fn temperature(&self) -> &Array3<f64> {
        &self.fields[TEMPERATURE_IDX]
    }
}
