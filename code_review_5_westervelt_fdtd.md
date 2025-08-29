# Code Review: src/physics/mechanics/acoustic_wave/westervelt_fdtd.rs

## Issues and Recommendations

1. **Issue:** Unnecessary clone of entire pressure array for Laplacian calculation
   - **Rationale:** Line 98 clones the entire pressure array to avoid borrow checker issues. This allocates O(n³) memory and copies all data, severely impacting performance.
   - **Suggestion:**
   ```rust
   fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
       // Use raw pointers to avoid borrow checker issues without cloning
       let pressure_ptr = self.pressure.as_ptr();
       let laplacian_ptr = self.laplacian.as_mut_ptr();
       let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
       
       unsafe {
           // Safe because we only read from pressure and write to different array
           match self.config.spatial_order {
               2 => self.laplacian_order2_unsafe(pressure_ptr, laplacian_ptr, nx, ny, nz, grid),
               4 => self.laplacian_order4_unsafe(pressure_ptr, laplacian_ptr, nx, ny, nz, grid),
               _ => unreachable!(),
           }
       }
       Ok(())
   }
   
   // Alternative: Split into separate arrays
   fn calculate_laplacian_into(
       pressure: &Array3<f64>,
       laplacian: &mut Array3<f64>,
       config: &WesterveltFdtdConfig,
       grid: &Grid,
   ) -> KwaversResult<()> {
       // Now no borrow issues since arrays are separate parameters
   }
   ```
   - **Critique:** The clone is a critical performance bug that makes the solver unusable for large grids. Using unsafe code with proper safety documentation or restructuring to avoid self-borrowing are both valid solutions. The unsafe approach is common in numerical codes (see BLAS implementations).

2. **Issue:** Grid dimensions accessed through public fields
   - **Rationale:** The code accesses `grid.nx`, `grid.ny`, `grid.nz`, `grid.dx` directly, violating encapsulation if Grid fields become private.
   - **Suggestion:**
   ```rust
   fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
       let (nx, ny, nz) = grid.dim();  // Use getter method
       let (dx, dy, dz) = grid.spacing_meters();  // Use getter for spacing
       
       let dx2_inv = 1.0 / (dx * dx);
       let dy2_inv = 1.0 / (dy * dy);
       let dz2_inv = 1.0 / (dz * dz);
       // ...
   }
   ```
   - **Critique:** Using getter methods maintains encapsulation and allows Grid implementation to evolve without breaking dependent code.

3. **Issue:** Triple-nested loops without parallelization
   - **Rationale:** The Laplacian calculation uses serial triple-nested loops, missing significant parallelization opportunity for multi-core CPUs.
   - **Suggestion:**
   ```rust
   use rayon::prelude::*;
   use ndarray::parallel::prelude::*;
   
   fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
       let (nx, ny, nz) = grid.dim();
       let dx2_inv = 1.0 / (grid.dx_meters().powi(2));
       let dy2_inv = 1.0 / (grid.dy_meters().powi(2));
       let dz2_inv = 1.0 / (grid.dz_meters().powi(2));
       
       // Parallel iteration over z-slices
       self.laplacian
           .axis_iter_mut(ndarray::Axis(2))
           .into_par_iter()
           .enumerate()
           .for_each(|(k, mut slice)| {
               if k > 0 && k < nz - 1 {
                   for j in 1..ny - 1 {
                       for i in 1..nx - 1 {
                           slice[[i, j]] = self.compute_laplacian_point(
                               i, j, k, dx2_inv, dy2_inv, dz2_inv
                           );
                       }
                   }
               }
           });
       Ok(())
   }
   ```
   - **Critique:** Parallelizing over the outermost feasible dimension provides good speedup with minimal synchronization overhead. For large grids, this can provide 4-8x speedup on modern CPUs.

4. **Issue:** No boundary condition handling in Laplacian
   - **Rationale:** The Laplacian calculation skips boundary points entirely, effectively applying zero Neumann conditions without documentation.
   - **Suggestion:**
   ```rust
   enum BoundaryCondition {
       Dirichlet(f64),
       Neumann(f64),
       Absorbing,
       Periodic,
   }
   
   impl WesterveltFdtd {
       fn apply_boundary_conditions(&mut self, bc: &BoundaryCondition) {
           let (nx, ny, nz) = self.pressure.dim();
           
           match bc {
               BoundaryCondition::Neumann(grad) => {
                   // Set boundary values to enforce gradient
                   for j in 0..ny {
                       for k in 0..nz {
                           self.pressure[[0, j, k]] = self.pressure[[1, j, k]] - grad * self.dx;
                           self.pressure[[nx-1, j, k]] = self.pressure[[nx-2, j, k]] + grad * self.dx;
                       }
                   }
                   // Similar for other boundaries
               }
               BoundaryCondition::Periodic => {
                   // Wrap boundaries
                   for j in 0..ny {
                       for k in 0..nz {
                           self.pressure[[0, j, k]] = self.pressure[[nx-2, j, k]];
                           self.pressure[[nx-1, j, k]] = self.pressure[[1, j, k]];
                       }
                   }
               }
               // ...
           }
       }
   }
   ```
   - **Critique:** Explicit boundary condition handling is essential for physical accuracy. The current implicit approach leads to reflections that corrupt the solution.

5. **Issue:** Missing validation for spatial order configuration
   - **Rationale:** The code accepts any spatial_order but only implements 2 and 4, with no error handling for other values.
   - **Suggestion:**
   ```rust
   #[derive(Debug, Clone, Copy)]
   pub enum SpatialOrder {
       Second,
       Fourth,
       Sixth,
   }
   
   impl SpatialOrder {
       fn minimum_grid_points(&self) -> usize {
           match self {
               Self::Second => 3,
               Self::Fourth => 5,
               Self::Sixth => 7,
           }
       }
   }
   
   pub struct WesterveltFdtdConfig {
       pub spatial_order: SpatialOrder,
       // ...
   }
   
   impl WesterveltFdtd {
       pub fn new(config: WesterveltFdtdConfig, grid: &Grid) -> KwaversResult<Self> {
           // Validate grid size for spatial order
           let min_points = config.spatial_order.minimum_grid_points();
           if grid.nx() < min_points || grid.ny() < min_points || grid.nz() < min_points {
               return Err(KwaversError::InvalidInput(format!(
                   "Grid too small for {:?} order: need at least {} points per dimension",
                   config.spatial_order, min_points
               )));
           }
           // ...
       }
   }
   ```
   - **Critique:** Type-safe enums prevent runtime errors and make the API self-documenting. Validation at construction ensures invariants are maintained.

6. **Issue:** No absorption term storage optimization
   - **Rationale:** The `pressure_prev2` field is `Option<Array3<f64>>` but is always needed when absorption is enabled, leading to unnecessary Option checks.
   - **Suggestion:**
   ```rust
   pub struct WesterveltFdtd {
       config: WesterveltFdtdConfig,
       pressure: Array3<f64>,
       pressure_prev: Array3<f64>,
       // Only allocate if absorption is enabled
       absorption_state: Option<AbsorptionState>,
       laplacian: Array3<f64>,
   }
   
   struct AbsorptionState {
       pressure_prev2: Array3<f64>,
       // Could add more absorption-specific fields
       attenuation_map: Option<Array3<f64>>,
   }
   
   impl WesterveltFdtd {
       pub fn new(config: WesterveltFdtdConfig, grid: &Grid) -> Self {
           let shape = grid.dim();
           let absorption_state = if config.enable_absorption {
               Some(AbsorptionState {
                   pressure_prev2: Array3::zeros(shape),
                   attenuation_map: None,
               })
           } else {
               None
           };
           // ...
       }
   }
   ```
   - **Critique:** Grouping related fields improves code organization and makes the memory cost of features explicit. This pattern scales better as more optional features are added.

7. **Issue:** Constants defined inline reduce readability
   - **Rationale:** Fourth-order stencil coefficients are defined as local constants in the loop, making them hard to verify and reuse.
   - **Suggestion:**
   ```rust
   /// Fourth-order finite difference coefficients
   mod stencil {
       /// Second derivative, 4th order accuracy
       pub mod d2 {
           pub const C0: f64 = -1.0 / 12.0;
           pub const C1: f64 = 4.0 / 3.0;
           pub const C2: f64 = -5.0 / 2.0;
           
           /// Verify coefficients sum to zero (conservation)
           #[cfg(test)]
           fn verify_coefficients() {
               let sum = 2.0 * C0 + 2.0 * C1 + C2;
               assert!((sum - 0.0).abs() < 1e-14);
           }
       }
       
       /// Sixth-order coefficients
       pub mod d2_order6 {
           pub const C0: f64 = 1.0 / 90.0;
           pub const C1: f64 = -3.0 / 20.0;
           pub const C2: f64 = 3.0 / 2.0;
           pub const C3: f64 = -49.0 / 18.0;
       }
   }
   
   // Usage:
   use self::stencil::d2;
   let d2_dx2 = (d2::C0 * pressure[[i - 2, j, k]] + /*...*/) * dx2_inv;
   ```
   - **Critique:** Centralizing coefficients improves maintainability and enables verification. The module structure allows easy extension to higher orders. Reference: LeVeque (2007) "Finite Difference Methods".