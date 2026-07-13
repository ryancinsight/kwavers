use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_math::numerics::operators::NumericsTrilinearInterpolator;
use leto::ArrayView3;
use std::collections::HashMap;

use super::residual::max_abs_difference;
use super::{
    CoupledPhysicsSolver, MultiPhysicsConservationEnforcer, SimulationMultiPhysicsInterface,
    SimulationPhysicsDomain,
};

/// Field coupling manager for conservative interpolation between domains
#[derive(Debug)]
pub struct MultiPhysicsFieldCoupler {
    /// Interpolation operators for each domain pair
    interpolators:
        HashMap<(SimulationPhysicsDomain, SimulationPhysicsDomain), NumericsTrilinearInterpolator>,
    /// Coupling interface definitions
    interfaces: HashMap<
        (SimulationPhysicsDomain, SimulationPhysicsDomain),
        SimulationMultiPhysicsInterface,
    >,
    /// Conservation enforcement
    conservation: MultiPhysicsConservationEnforcer,
}

impl MultiPhysicsFieldCoupler {
    /// Create a new field coupler
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            interpolators: HashMap::new(),
            interfaces: HashMap::new(),
            conservation: MultiPhysicsConservationEnforcer::new(),
        }
    }

    /// Add coupling between two physics domains
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn add_coupling(
        &mut self,
        source_domain: SimulationPhysicsDomain,
        target_domain: SimulationPhysicsDomain,
        source_grid: &Grid,
        target_grid: &Grid,
    ) -> KwaversResult<()> {
        let key = (source_domain, target_domain);

        // Create interpolator for this domain pair
        let interpolator =
            NumericsTrilinearInterpolator::new(target_grid.dx, target_grid.dy, target_grid.dz);

        // Conservative interpolation via AABB overlap quadrature (Farhat et al. 1998)
        // and Schwarz alternating coupling are implemented in MultiPhysicsConservationEnforcer below.
        let interface = SimulationMultiPhysicsInterface::new(source_grid, target_grid)?;

        self.interpolators.insert(key, interpolator);
        self.interfaces.insert(key, interface);

        Ok(())
    }

    /// Transfer field conservatively between domains
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn transfer_field(
        &mut self,
        source_domain: SimulationPhysicsDomain,
        target_domain: SimulationPhysicsDomain,
        field_name: &str,
        source_solver: &dyn CoupledPhysicsSolver,
        target_solver: &mut dyn CoupledPhysicsSolver,
        relaxation: f64,
    ) -> KwaversResult<f64> {
        let _key = (source_domain, target_domain);

        // Get source field
        let source_field = source_solver.get_field(field_name)?;

        // Apply conservative interpolation
        let interpolated = self.conservation.conservative_interpolate(
            &source_field,
            source_solver.grid(),
            target_solver.grid(),
        )?;

        // Apply relaxation for stability
        let current_target = target_solver.get_field(field_name)?.to_contiguous();
        let relaxed_field = &(&interpolated * relaxation) + &(&current_target * (1.0 - relaxation));

        // Update target field
        target_solver.set_field(field_name, relaxed_field.view())?;

        let residual = max_abs_difference(relaxed_field.view(), current_target.view())?;
        Ok(residual)
    }

    /// Transfer a pre-snapshotted field array between domains.
    ///
    /// Identical to `transfer_field` but accepts a raw `ArrayView3<f64>` for the
    /// source field rather than extracting it from a solver. Used by monolithic
    /// coupling to implement Jacobi-style simultaneous updates (all source fields
    /// are taken from the same snapshot u^k rather than the evolving state).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn transfer_field_array(
        &mut self,
        source_domain: SimulationPhysicsDomain,
        target_domain: SimulationPhysicsDomain,
        field_name: &str,
        source_snapshot: ArrayView3<f64>,
        target_solver: &mut dyn CoupledPhysicsSolver,
        relaxation: f64,
    ) -> KwaversResult<f64> {
        // Apply conservative interpolation from snapshot to target grid.
        // Reconstruct a temporary Grid from the stored source dims if the interface
        // exists; otherwise fall back to using the snapshot dimensions directly via
        // the target grid (valid when source and target grids are the same size).
        let interpolated = if let Some(iface) = self.interfaces.get(&(source_domain, target_domain))
        {
            let (nx, ny, nz) = iface.source_dims;
            let (dx, dy, dz) = iface.source_spacing;
            // Build ephemeral source grid to satisfy conservative_interpolate's validation
            let src_grid = Grid::new(nx, ny, nz, dx, dy, dz).map_err(|e| {
                KwaversError::InvalidInput(format!("Monolithic coupling: invalid source grid: {e}"))
            })?;
            self.conservation.conservative_interpolate(
                &source_snapshot,
                &src_grid,
                target_solver.grid(),
            )?
        } else {
            // No registered coupling: assume same grid (identity transfer)
            source_snapshot.to_contiguous()
        };

        let current_target = target_solver.get_field(field_name)?.to_contiguous();
        let relaxed_field = &(&interpolated * relaxation) + &(&current_target * (1.0 - relaxation));
        target_solver.set_field(field_name, relaxed_field.view())?;

        let residual = max_abs_difference(relaxed_field.view(), current_target.view())?;
        Ok(residual)
    }
}

impl Default for MultiPhysicsFieldCoupler {
    fn default() -> Self {
        Self::new()
    }
}
