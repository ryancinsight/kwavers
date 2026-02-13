use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::source::{GridSource, Source, SourceField, SourceMode};
use ndarray::{Array2, Array3, Zip};

#[derive(Debug)]
pub struct SourceHandler {
    source: GridSource,
    p_indices: Vec<(usize, usize, usize, f64)>,
    u_indices: Vec<(usize, usize, usize, f64)>,
    p_scale_rho: Vec<f64>,
    p_scale_p: Vec<f64>,
    /// Effective propagation dimensionality for the FDTD pressure source.
    /// A plane wave (filling 2D, propagating in 1D) has source_propagation_dim = 1.
    /// A line source (filling 1D, propagating in 2D) has source_propagation_dim = 2.
    /// A point source (0D, propagating in 3D) has source_propagation_dim = 3.
    source_propagation_dim: f64,
}

impl SourceHandler {
    pub fn new(source: GridSource, grid: &Grid) -> KwaversResult<Self> {
        let shape = (grid.nx, grid.ny, grid.nz);

        if source.p_signal.is_some() && source.p_mask.is_none() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Pressure source signal provided without pressure source mask"
                        .to_string(),
                },
            ));
        }

        if source.u_signal.is_some() && source.u_mask.is_none() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Velocity source signal provided without velocity source mask"
                        .to_string(),
                },
            ));
        }

        let mut p_indices = Vec::new();
        let mut source_propagation_dim = 0.0;
        if let Some(mask) = &source.p_mask {
            if mask.dim() != shape {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Pressure source mask shape mismatch: expected {:?}, got {:?}",
                            shape,
                            mask.dim()
                        ),
                    },
                ));
            }
            for ((i, j, k), &val) in mask.indexed_iter() {
                if val != 0.0 {
                    p_indices.push((i, j, k, val));
                }
            }

            if !p_indices.is_empty() {
                // Compute effective propagation dimensionality.
                //
                // For k-Wave compatibility in FDTD direct-pressure schemes:
                // k-Wave's split-density adds source to N density components and
                // the wave equation propagates each dimension independently. In
                // FDTD with a single pressure field, the divergence operator handles
                // all dimensions at once. The correct source scaling depends on the
                // source geometry:
                //
                // - Plane source (fills 2D, e.g. all x,y at one z): propagates in
                //   1D -> source_propagation_dim = 1
                // - Line source (fills 1D): propagates in 2D -> source_propagation_dim = 2
                // - Point source (0D): propagates in 3D -> source_propagation_dim = 3
                //
                // We detect the source dimensionality by checking how many unique
                // coordinate indices the source occupies in each axis.
                let (nx, ny, nz) = shape;

                // Count unique indices per axis
                let mut x_set = std::collections::HashSet::new();
                let mut y_set = std::collections::HashSet::new();
                let mut z_set = std::collections::HashSet::new();
                for &(i, j, k, _) in &p_indices {
                    x_set.insert(i);
                    y_set.insert(j);
                    z_set.insert(k);
                }

                // A "filled" dimension means the source spans (nearly) all grid points
                // in that dimension. An "unfilled" dimension contributes to propagation.
                let mut dim_count = 0usize;
                if nx > 1 {
                    dim_count += 1;
                }
                if ny > 1 {
                    dim_count += 1;
                }
                if nz > 1 {
                    dim_count += 1;
                }

                // Count filled dimensions (source spans > 50% of grid in that axis)
                let mut source_fill_dims = 0;
                if nx > 1 && x_set.len() > nx / 2 {
                    source_fill_dims += 1;
                }
                if ny > 1 && y_set.len() > ny / 2 {
                    source_fill_dims += 1;
                }
                if nz > 1 && z_set.len() > nz / 2 {
                    source_fill_dims += 1;
                }

                // Propagation dimensions = total active dimensions - source fill dimensions
                source_propagation_dim = (dim_count - source_fill_dims).max(1) as f64;
            }
        }

        let mut u_indices = Vec::new();
        if let Some(mask) = &source.u_mask {
            if mask.dim() != shape {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Velocity source mask shape mismatch: expected {:?}, got {:?}",
                            shape,
                            mask.dim()
                        ),
                    },
                ));
            }
            for ((i, j, k), &val) in mask.indexed_iter() {
                if val != 0.0 {
                    u_indices.push((i, j, k, val));
                }
            }
        }

        if source.p_mask.is_some() {
            let signal = source.p_signal.as_ref().ok_or_else(|| {
                KwaversError::Validation(ValidationError::ConstraintViolation {
                    message: "Pressure source mask provided without pressure source signal"
                        .to_string(),
                })
            })?;

            if p_indices.is_empty() {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Pressure source mask contains no active source points"
                            .to_string(),
                    },
                ));
            }

            let num_sources_signal = signal.shape()[0];
            if num_sources_signal != 1 && num_sources_signal != p_indices.len() {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Pressure source signal shape mismatch: expected [1|{}, time], got [{}, time]",
                            p_indices.len(),
                            num_sources_signal
                        ),
                    },
                ));
            }
        }

        if source.u_mask.is_some() {
            let signal = source.u_signal.as_ref().ok_or_else(|| {
                KwaversError::Validation(ValidationError::ConstraintViolation {
                    message: "Velocity source mask provided without velocity source signal"
                        .to_string(),
                })
            })?;

            if u_indices.is_empty() {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Velocity source mask contains no active source points"
                            .to_string(),
                    },
                ));
            }

            if signal.shape()[0] != 3 {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Velocity source signal shape mismatch: expected [3, sources, time], got [{}, sources, time]",
                            signal.shape()[0]
                        ),
                    },
                ));
            }

            let num_sources_signal = signal.shape()[1];
            if num_sources_signal != 1 && num_sources_signal != u_indices.len() {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Velocity source signal shape mismatch: expected [3, 1|{}, time], got [3, {}, time]",
                            u_indices.len(),
                            num_sources_signal
                        ),
                    },
                ));
            }
        }

        Ok(Self {
            source,
            p_indices,
            u_indices,
            p_scale_rho: Vec::new(),
            p_scale_p: Vec::new(),
            source_propagation_dim,
        })
    }

    /// Precompute k-Wave compatible pressure source scaling for mass (rho) and pressure updates.
    ///
    /// This mirrors `scale_source_terms` in k-Wave for additive/dirichlet modes on a uniform grid.
    pub fn prepare_pressure_source_scaling(&mut self, grid: &Grid, c0: &Array3<f64>, dt: f64) {
        if self.p_indices.is_empty() {
            self.p_scale_rho.clear();
            self.p_scale_p.clear();
            return;
        }

        let mut dim_count = 0usize;
        if grid.nx > 1 {
            dim_count += 1;
        }
        if grid.ny > 1 {
            dim_count += 1;
        }
        if grid.nz > 1 {
            dim_count += 1;
        }
        let n_dim = if dim_count == 0 {
            1.0
        } else {
            dim_count as f64
        };

        // k-Wave uses dx for uniform grids; assume uniform spacing here.
        let dx = grid.dx;

        self.p_scale_rho = Vec::with_capacity(self.p_indices.len());
        self.p_scale_p = Vec::with_capacity(self.p_indices.len());

        for &(i, j, k, _weight) in &self.p_indices {
            let c0_val = c0[[i, j, k]];
            let (scale_rho, scale_p) = match self.source.p_mode {
                SourceMode::Dirichlet => {
                    // k-Wave: source.p scaled by 1/(N * c0^2); for direct pressure use 1.0
                    let rho_scale = 1.0 / (n_dim * c0_val * c0_val);
                    (rho_scale, 1.0)
                }
                SourceMode::Additive | SourceMode::AdditiveNoCorrection => {
                    // k-Wave additive mass source scaling.
                    //
                    // k-Wave adds source.p (pre-scaled by 2·dt/(N·c₀·dx)) to EACH of
                    // the N split-density components (rhox, rhoy, rhoz).
                    //
                    // rho_scale: Per-component density scaling (with 1/N split).
                    //   Used by PSTD which adds to all N density components.
                    //
                    // p_scale: Propagation-dimension-adjusted pressure scaling.
                    //   Used by FDTD which adds to a single pressure field.
                    //   k-Wave's split-density naturally distributes the source
                    //   energy across N propagation dimensions. In FDTD's single
                    //   pressure equation, we scale by 1/N_prop where N_prop is
                    //   the number of propagation dimensions (determined by source
                    //   geometry: plane→1, line→2, point→3).
                    //   p_scale = 2·dt·c₀ / (N_prop · dx).
                    let rho_scale = (2.0 * dt) / (n_dim * c0_val * dx);
                    let n_prop = self.source_propagation_dim.max(1.0);
                    let p_scale = (2.0 * dt * c0_val) / (n_prop * dx);
                    (rho_scale, p_scale)
                }
            };

            self.p_scale_rho.push(scale_rho);
            self.p_scale_p.push(scale_p);
        }
    }

    pub fn add_source(
        &mut self,
        source: std::sync::Arc<dyn Source>,
        grid: &Grid,
        nt: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let mask = source.create_mask(grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        if mask.dim() != shape {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Source mask shape mismatch: expected {:?}, got {:?}",
                        shape,
                        mask.dim()
                    ),
                },
            ));
        }

        match source.source_type() {
            SourceField::Pressure => {
                if self.source.p_mask.is_some() || self.source.p_signal.is_some() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Multiple pressure sources are not supported in SourceHandler"
                                .to_string(),
                        },
                    ));
                }

                let mut indices = Vec::new();
                for ((i, j, k), &val) in mask.indexed_iter() {
                    if val != 0.0 {
                        indices.push((i, j, k, val));
                    }
                }
                if indices.is_empty() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Source mask contains no active source points".to_string(),
                        },
                    ));
                }

                let mut signal = Array2::zeros((1, nt));
                for step in 0..nt {
                    let t = step as f64 * dt;
                    signal[[0, step]] = source.amplitude(t);
                }

                self.source.p_mask = Some(mask);
                self.source.p_signal = Some(signal);
                self.p_indices = indices;

                Ok(())
            }
            SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ => {
                if self.source.u_mask.is_some() || self.source.u_signal.is_some() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Multiple velocity sources are not supported in SourceHandler"
                                .to_string(),
                        },
                    ));
                }

                let mut indices = Vec::new();
                for ((i, j, k), &val) in mask.indexed_iter() {
                    if val != 0.0 {
                        indices.push((i, j, k, val));
                    }
                }
                if indices.is_empty() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Source mask contains no active source points".to_string(),
                        },
                    ));
                }

                let mut signal = ndarray::Array3::zeros((3, 1, nt));
                let comp = match source.source_type() {
                    SourceField::VelocityX => 0,
                    SourceField::VelocityY => 1,
                    SourceField::VelocityZ => 2,
                    SourceField::Pressure => {
                        return Err(KwaversError::Validation(
                            ValidationError::ConstraintViolation {
                                message: "Pressure source cannot be used as velocity source in this context. Use add_source with Pressure source type in the pressure source branch.".to_string(),
                            },
                        ));
                    }
                };
                for step in 0..nt {
                    let t = step as f64 * dt;
                    signal[[comp, 0, step]] = source.amplitude(t);
                }

                self.source.u_mask = Some(mask);
                self.source.u_signal = Some(signal);
                self.u_indices = indices;

                Ok(())
            }
        }
    }

    /// Apply initial conditions to fields
    pub fn apply_initial_conditions(
        &self,
        p: &mut Array3<f64>,
        rho: &mut Array3<f64>,
        c0: &Array3<f64>,
        ux: &mut Array3<f64>,
        uy: &mut Array3<f64>,
        uz: &mut Array3<f64>,
    ) {
        if let Some(p0) = &self.source.p0 {
            p.assign(p0);
            Zip::from(rho)
                .and(p0)
                .and(c0)
                .for_each(|rho_cell, &p_cell, &c0_cell| {
                    *rho_cell = p_cell / (c0_cell * c0_cell);
                });
        }

        if let Some((ux0, uy0, uz0)) = &self.source.u0 {
            ux.assign(ux0);
            uy.assign(uy0);
            uz.assign(uz0);
        }
    }

    /// Inject mass source (pressure source) into density field
    ///
    /// Adds source.p / c0^2 to rho
    pub fn inject_mass_source(&self, time_index: usize, rho: &mut Array3<f64>, c0: &Array3<f64>) {
        if let Some(signal) = &self.source.p_signal {
            if time_index < signal.shape()[1] {
                let mode = self.source.p_mode;
                let is_scalar_signal = signal.shape()[0] == 1 && self.p_indices.len() > 1;

                for (idx, &(i, j, k, weight)) in self.p_indices.iter().enumerate() {
                    let val = if is_scalar_signal {
                        signal[[0, time_index]]
                    } else {
                        signal[[idx, time_index]]
                    };

                    let scale = self
                        .p_scale_rho
                        .get(idx)
                        .copied()
                        .unwrap_or_else(|| 1.0 / (c0[[i, j, k]] * c0[[i, j, k]]));
                    let rho_val = weight * val * scale;

                    match mode {
                        SourceMode::Additive | SourceMode::AdditiveNoCorrection => {
                            rho[[i, j, k]] += rho_val;
                        }
                        SourceMode::Dirichlet => {
                            rho[[i, j, k]] = rho_val;
                        }
                    }
                }
            }
        }
    }

    /// Inject pressure source directly into pressure field (for FDTD).
    ///
    /// Uses the explicitly-set source mode (`p_mode`). k-Wave's additive mode
    /// applies mass-source scaling `2·dt/(N·c₀·dx)` to the source values before
    /// adding them to the density field. This scaling is pre-computed in
    /// `prepare_pressure_source_scaling()` and stored in `p_scale_p`.
    pub fn inject_pressure_source(&self, time_index: usize, p: &mut Array3<f64>) {
        if let Some(signal) = &self.source.p_signal {
            if time_index < signal.shape()[1] {
                let mode = self.source.p_mode;
                let is_scalar_signal = signal.shape()[0] == 1 && self.p_indices.len() > 1;

                for (idx, &(i, j, k, weight)) in self.p_indices.iter().enumerate() {
                    let val = if is_scalar_signal {
                        signal[[0, time_index]]
                    } else {
                        signal[[idx, time_index]]
                    };
                    let scale = self.p_scale_p.get(idx).copied().unwrap_or(1.0);
                    let p_val = weight * val * scale;

                    match mode {
                        SourceMode::Additive | SourceMode::AdditiveNoCorrection => {
                            p[[i, j, k]] += p_val;
                        }
                        SourceMode::Dirichlet => {
                            p[[i, j, k]] = p_val;
                        }
                    }
                }
            }
        }
    }

    /// Add pressure source into a destination array (for k-space source correction workflows)
    pub fn add_pressure_source_into(&self, time_index: usize, dest: &mut Array3<f64>) {
        if let Some(signal) = &self.source.p_signal {
            if time_index < signal.shape()[1] {
                let mode = self.source.p_mode;
                let is_scalar_signal = signal.shape()[0] == 1 && self.p_indices.len() > 1;

                for (idx, &(i, j, k, weight)) in self.p_indices.iter().enumerate() {
                    let val = if is_scalar_signal {
                        signal[[0, time_index]]
                    } else {
                        signal[[idx, time_index]]
                    };
                    let scale = self.p_scale_p.get(idx).copied().unwrap_or(1.0);
                    let p_val = weight * val * scale;

                    match mode {
                        SourceMode::Additive | SourceMode::AdditiveNoCorrection => {
                            dest[[i, j, k]] += p_val;
                        }
                        SourceMode::Dirichlet => {
                            dest[[i, j, k]] = p_val;
                        }
                    }
                }
            }
        }
    }

    /// Add pressure source into a density/mass destination array (rho)
    pub fn add_pressure_source_into_density(&self, time_index: usize, dest: &mut Array3<f64>) {
        if let Some(signal) = &self.source.p_signal {
            if time_index < signal.shape()[1] {
                let mode = self.source.p_mode;
                let is_scalar_signal = signal.shape()[0] == 1 && self.p_indices.len() > 1;

                for (idx, &(i, j, k, weight)) in self.p_indices.iter().enumerate() {
                    let val = if is_scalar_signal {
                        signal[[0, time_index]]
                    } else {
                        signal[[idx, time_index]]
                    };
                    let scale = self.p_scale_rho.get(idx).copied().unwrap_or(0.0);
                    let rho_val = weight * val * scale;

                    match mode {
                        SourceMode::Additive | SourceMode::AdditiveNoCorrection => {
                            dest[[i, j, k]] += rho_val;
                        }
                        SourceMode::Dirichlet => {
                            dest[[i, j, k]] = rho_val;
                        }
                    }
                }
            }
        }
    }

    /// Enforce Dirichlet pressure source after field updates.
    /// Only enforces when the source mode is explicitly Dirichlet.
    pub fn enforce_pressure_dirichlet(&self, time_index: usize, p: &mut Array3<f64>) {
        if self.source.p_mode != SourceMode::Dirichlet {
            return;
        }

        if let Some(signal) = &self.source.p_signal {
            if time_index < signal.shape()[1] {
                let is_scalar_signal = signal.shape()[0] == 1 && self.p_indices.len() > 1;

                for (idx, &(i, j, k, weight)) in self.p_indices.iter().enumerate() {
                    let val = if is_scalar_signal {
                        signal[[0, time_index]]
                    } else {
                        signal[[idx, time_index]]
                    };
                    let scale = self.p_scale_p.get(idx).copied().unwrap_or(1.0);
                    p[[i, j, k]] = weight * val * scale;
                }
            }
        }
    }

    /// Inject force source (velocity source) into velocity fields
    pub fn inject_force_source(
        &self,
        time_index: usize,
        ux: &mut Array3<f64>,
        uy: &mut Array3<f64>,
        uz: &mut Array3<f64>,
    ) {
        if let Some(signal) = &self.source.u_signal {
            if time_index < signal.shape()[2] {
                let mode = self.source.u_mode;
                let is_scalar_signal = signal.shape()[1] == 1 && self.u_indices.len() > 1;

                for (idx, &(i, j, k, weight)) in self.u_indices.iter().enumerate() {
                    let sig_idx = if is_scalar_signal { 0 } else { idx };
                    let val_x = weight * signal[[0, sig_idx, time_index]];
                    let val_y = weight * signal[[1, sig_idx, time_index]];
                    let val_z = weight * signal[[2, sig_idx, time_index]];

                    match mode {
                        SourceMode::Additive | SourceMode::AdditiveNoCorrection => {
                            ux[[i, j, k]] += val_x;
                            uy[[i, j, k]] += val_y;
                            uz[[i, j, k]] += val_z;
                        }
                        SourceMode::Dirichlet => {
                            ux[[i, j, k]] = val_x;
                            uy[[i, j, k]] = val_y;
                            uz[[i, j, k]] = val_z;
                        }
                    }
                }
            }
        }
    }

    pub fn has_pressure_source(&self) -> bool {
        self.source.p_signal.is_some() && !self.p_indices.is_empty()
    }

    pub fn has_velocity_source(&self) -> bool {
        self.source.u_signal.is_some() && !self.u_indices.is_empty()
    }

    pub fn pressure_mode(&self) -> SourceMode {
        self.source.p_mode
    }

    pub fn velocity_mode(&self) -> SourceMode {
        self.source.u_mode
    }
}
