use crate::grid::Grid;
use ndarray::{Array2, Array3, Zip};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SourceMode {
    #[default]
    Additive,
    Dirichlet, // Enforce value (hard source)
}

/// Container for k-Wave style source definitions
#[derive(Default, Clone, Debug)]
pub struct KWaveSource {
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
    source: KWaveSource,
    p_indices: Vec<(usize, usize, usize)>,
    u_indices: Vec<(usize, usize, usize)>,
}

impl SourceHandler {
    pub fn new(source: KWaveSource, _grid: &Grid) -> Self {
        let mut p_indices = Vec::new();
        if let Some(mask) = &source.p_mask {
            assert_eq!(
                mask.shape(),
                &[_grid.nx, _grid.ny, _grid.nz],
                "Pressure source mask dimensions must match grid"
            );
            for ((i, j, k), &val) in mask.indexed_iter() {
                if val != 0.0 {
                    p_indices.push((i, j, k));
                }
            }
        }

        let mut u_indices = Vec::new();
        if let Some(mask) = &source.u_mask {
            assert_eq!(
                mask.shape(),
                &[_grid.nx, _grid.ny, _grid.nz],
                "Velocity source mask dimensions must match grid"
            );
            for ((i, j, k), &val) in mask.indexed_iter() {
                if val != 0.0 {
                    u_indices.push((i, j, k));
                }
            }
        }

        Self {
            source,
            p_indices,
            u_indices,
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
}
