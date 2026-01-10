//! Multi-physics solver implementation
//!
//! This module implements a unified solver for coupled acoustic-optical-thermal simulations.

use crate::domain::core::error::KwaversResult;
use crate::domain::field::indices::*;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::solver::multiphysics::field_coupling::{CouplingStrategy, FieldCoupler};
use ndarray::Array3;
use std::sync::Arc;

/// Unified multi-physics solver
#[derive(Debug)]
pub struct MultiPhysicsSolver {
    /// Field coupler for multi-physics interactions
    field_coupler: FieldCoupler,
    /// Computational grid
    _grid: Grid,
    /// Medium properties
    _medium: Arc<dyn Medium + Send + Sync>,
    /// Multi-physics field array
    fields: Vec<Array3<f64>>,
}

impl MultiPhysicsSolver {
    /// Create a new multi-physics solver
    pub fn new(
        grid: Grid,
        medium: Arc<dyn Medium + Send + Sync>,
        coupling_strategy: CouplingStrategy,
    ) -> KwaversResult<Self> {
        // Initialize field coupler
        let field_coupler = FieldCoupler::new(coupling_strategy);

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
