use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};
use leto::Array3;

mod inject;
mod registration;
mod scaling;

/// Collect active pressure-source voxels in MATLAB / Fortran order.
#[inline]
pub(super) fn collect_pressure_indices_fortran(
    mask: &Array3<f64>,
) -> Vec<(usize, usize, usize, f64)> {
    let [nx, ny, nz] = mask.shape();
    let mut indices = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let val = mask[[i, j, k]];
                if val != 0.0 {
                    indices.push((i, j, k, val));
                }
            }
        }
    }
    indices
}

#[derive(Debug)]
pub struct SourceHandler {
    pub(super) source: GridSource,
    pub(super) p_indices: Vec<(usize, usize, usize, f64)>,
    pub(super) u_indices: Vec<(usize, usize, usize, f64)>,
    pub(super) p_scale_rho: Vec<f64>,
    pub(super) p_scale_p: Vec<f64>,
    /// Effective propagation dimensionality for the FDTD pressure source.
    pub(super) source_propagation_dim: f64,
    /// Per-voxel k-space source correction for velocity sources in additive mode.
    ///
    /// ## Theorem (K-Wave Source Kappa, Treeby & Cox 2010)
    /// For additive velocity sources, k-Wave applies the half-step leapfrog phase
    /// correction `ifftshift(cos(c_ref·|k|·dt/2))` evaluated at each source voxel's
    /// physical position. This compensates for the staggered-time discretisation and
    /// ensures the injected source spectrum matches the analytic solution to spectral
    /// accuracy.
    ///
    /// After ifftshift, the value at physical position (i,j,k) is:
    ///   κ(i,j,k) = cos(c_ref·|k_fft[(i+Nx/2)%Nx, (j+Ny/2)%Ny, (k+Nz/2)%Nz]|·dt/2)
    ///
    /// Empty when no correction has been set (no k-space filtering applied).
    pub(super) u_kappa: Vec<f64>,
    /// Per-source-point additive velocity source scale factor `2·c₀·Δt/Δα`
    /// for axis α ∈ {x, y, z} (Cox et al. IEEE IUS 2018; k-wave-python
    /// `kspace_solver.py:533`). Applied multiplicatively in
    /// [`Self::inject_force_source`] for `Additive` and
    /// `AdditiveNoCorrection` modes; ignored for `Dirichlet`. Empty when
    /// no velocity source is registered or [`Self::prepare_velocity_source_scaling`]
    /// has not yet been called.
    pub(super) u_scale_x: Vec<f64>,
    pub(super) u_scale_y: Vec<f64>,
    pub(super) u_scale_z: Vec<f64>,
}

impl SourceHandler {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(source: GridSource, grid: &Grid) -> KwaversResult<Self> {
        let shape = [grid.nx, grid.ny, grid.nz];

        if source.p_signal.is_some() && source.p_mask.is_none() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Pressure source signal provided without pressure source mask"
                        .to_owned(),
                },
            ));
        }

        if source.u_signal.is_some() && source.u_mask.is_none() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Velocity source signal provided without velocity source mask"
                        .to_owned(),
                },
            ));
        }

        let mut p_indices = Vec::new();
        let mut source_propagation_dim = 0.0;
        if let Some(mask) = &source.p_mask {
            if mask.shape() != shape {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Pressure source mask shape mismatch: expected {:?}, got {:?}",
                            shape,
                            mask.shape()
                        ),
                    },
                ));
            }
            p_indices = collect_pressure_indices_fortran(mask);

            if !p_indices.is_empty() {
                let [nx, ny, nz] = shape;

                let mut x_set = std::collections::HashSet::new();
                let mut y_set = std::collections::HashSet::new();
                let mut z_set = std::collections::HashSet::new();
                for &(i, j, k, _) in &p_indices {
                    x_set.insert(i);
                    y_set.insert(j);
                    z_set.insert(k);
                }

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

                let mut source_fill_dims = 0;
                if nx > 1 && (x_set.len()) > nx / 2 {
                    source_fill_dims += 1;
                }
                if ny > 1 && (y_set.len()) > ny / 2 {
                    source_fill_dims += 1;
                }
                if nz > 1 && (z_set.len()) > nz / 2 {
                    source_fill_dims += 1;
                }

                source_propagation_dim = (dim_count - source_fill_dims).max(1) as f64;
            }
        }

        let mut u_indices = Vec::new();
        if let Some(mask) = &source.u_mask {
            if mask.shape() != shape {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Velocity source mask shape mismatch: expected {:?}, got {:?}",
                            shape,
                            mask.shape()
                        ),
                    },
                ));
            }
            for ([i, j, k], &val) in mask.indexed_iter() {
                if val != 0.0 {
                    u_indices.push((i, j, k, val));
                }
            }
        }

        if source.p_mask.is_some() {
            let signal = source.p_signal.as_ref().ok_or_else(|| {
                KwaversError::Validation(ValidationError::ConstraintViolation {
                    message: "Pressure source mask provided without pressure source signal"
                        .to_owned(),
                })
            })?;

            if p_indices.is_empty() {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Pressure source mask contains no active source points".to_owned(),
                    },
                ));
            }

            let num_sources_signal = signal.shape()[0];
            if num_sources_signal != 1 && num_sources_signal != (p_indices.len()) {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Pressure source signal shape mismatch: expected [1|{}, time], got [{}, time]",
                            (p_indices.len()),
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
                        .to_owned(),
                })
            })?;

            if u_indices.is_empty() {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Velocity source mask contains no active source points".to_owned(),
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
            if num_sources_signal != 1 && num_sources_signal != (u_indices.len()) {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Velocity source signal shape mismatch: expected [3, 1|{}, time], got [3, {}, time]",
                            (u_indices.len()),
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
            u_kappa: Vec::new(),
            u_scale_x: Vec::new(),
            u_scale_y: Vec::new(),
            u_scale_z: Vec::new(),
        })
    }

    #[must_use]
    pub fn has_initial_pressure(&self) -> bool {
        self.source.p0.is_some()
    }

    #[must_use]
    pub fn has_initial_velocity(&self) -> bool {
        self.source.u0.is_some()
    }

    #[must_use]
    pub fn has_pressure_source(&self) -> bool {
        self.source.p_signal.is_some() && !self.p_indices.is_empty()
    }

    #[must_use]
    pub fn has_velocity_source(&self) -> bool {
        self.source.u_signal.is_some() && !self.u_indices.is_empty()
    }

    #[must_use]
    pub fn pressure_mode(&self) -> SourceMode {
        self.source.p_mode
    }

    #[must_use]
    pub fn velocity_mode(&self) -> SourceMode {
        self.source.u_mode
    }
}
