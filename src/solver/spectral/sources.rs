use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use ndarray::{Array2, Array3, Zip};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SourceMode {
    #[default]
    Additive,
    Dirichlet, // Enforce value (hard source)
}

/// Container for k-space source definitions
#[derive(Default, Clone, Debug)]
pub struct SpectralSource {
    // Initial conditions
    pub p0: Option<Array3<f64>>,
    pub u0: Option<(Array3<f64>, Array3<f64>, Array3<f64>)>,

    // Time varying pressure source
    // If p_mask is defined, p_signal must be defined
    pub p_mask: Option<Array3<f64>>,
    pub p_signal: Option<Array2<f64>>, // [num_sources, time_steps]
    pub p_mode: SourceMode,

    // Time varying velocity source
    pub u_mask: Option<Array3<f64>>,
    pub u_signal: Option<Array3<f64>>, // [3, num_sources, time_steps]
    pub u_mode: SourceMode,
}

#[derive(Debug)]
pub struct SourceHandler {
    source: SpectralSource,
    p_indices: Vec<(usize, usize, usize)>,
    u_indices: Vec<(usize, usize, usize)>,
}

impl SourceHandler {
    pub fn new(source: SpectralSource, grid: &Grid) -> KwaversResult<Self> {
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
                    p_indices.push((i, j, k));
                }
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
                    u_indices.push((i, j, k));
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
        })
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

                for (idx, &(i, j, k)) in self.p_indices.iter().enumerate() {
                    let val = if is_scalar_signal {
                        signal[[0, time_index]]
                    } else {
                        signal[[idx, time_index]]
                    };

                    // Scale by c0^2 to convert pressure to density perturbation
                    // p = c0^2 * rho => rho = p / c0^2
                    let c0_val = c0[[i, j, k]];
                    let rho_val = val / (c0_val * c0_val);

                    match mode {
                        SourceMode::Additive => {
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

                for (idx, &(i, j, k)) in self.u_indices.iter().enumerate() {
                    let sig_idx = if is_scalar_signal { 0 } else { idx };
                    let val_x = signal[[0, sig_idx, time_index]];
                    let val_y = signal[[1, sig_idx, time_index]];
                    let val_z = signal[[2, sig_idx, time_index]];

                    match mode {
                        SourceMode::Additive => {
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
