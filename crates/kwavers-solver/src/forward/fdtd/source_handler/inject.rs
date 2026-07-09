use super::SourceHandler;
use kwavers_source::SourceMode;
use leto::Array3 as LetoArray3;
use leto::Array3 as NdArray3;

pub(crate) trait Array3Read {
    fn at(&self, i: usize, j: usize, k: usize) -> f64;
}

pub(crate) trait Array3Write: Array3Read {
    fn add_or_set(&mut self, i: usize, j: usize, k: usize, value: f64, mode: SourceMode);
}

impl Array3Read for LetoArray3<f64> {
    fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        self[[i, j, k]]
    }
}

impl Array3Read for NdArray3<f64> {
    fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        self[[i, j, k]]
    }
}

impl Array3Write for LetoArray3<f64> {
    fn add_or_set(&mut self, i: usize, j: usize, k: usize, value: f64, mode: SourceMode) {
        match mode {
            SourceMode::Additive | SourceMode::AdditiveNoCorrection => self[[i, j, k]] += value,
            SourceMode::Dirichlet => self[[i, j, k]] = value,
        }
    }
}

impl Array3Write for NdArray3<f64> {
    fn add_or_set(&mut self, i: usize, j: usize, k: usize, value: f64, mode: SourceMode) {
        match mode {
            SourceMode::Additive | SourceMode::AdditiveNoCorrection => self[[i, j, k]] += value,
            SourceMode::Dirichlet => self[[i, j, k]] = value,
        }
    }
}

impl SourceHandler {
    /// Apply initial conditions to fields.
    pub(crate) fn apply_initial_conditions(
        &self,
        p: &mut LetoArray3<f64>,
        rho: &mut impl Array3Write,
        c0: &impl Array3Read,
        ux: &mut LetoArray3<f64>,
        uy: &mut LetoArray3<f64>,
        uz: &mut LetoArray3<f64>,
    ) {
        if let Some(p0) = &self.source.p0 {
            for (dst, src) in p
                .as_slice_mut()
                .expect("FDTD pressure field must be contiguous")
                .iter_mut()
                .zip(p0.iter())
            {
                *dst = *src;
            }
            let [nx, ny, nz] = p.shape();
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let p_cell = p0[[i, j, k]];
                        let c0_cell = c0.at(i, j, k);
                        rho.add_or_set(
                            i,
                            j,
                            k,
                            p_cell / (c0_cell * c0_cell),
                            SourceMode::Dirichlet,
                        );
                    }
                }
            }
        }

        if let Some((ux0, uy0, uz0)) = &self.source.u0 {
            for (dst, src) in ux
                .as_slice_mut()
                .expect("FDTD velocity-x field must be contiguous")
                .iter_mut()
                .zip(ux0.iter())
            {
                *dst = *src;
            }
            for (dst, src) in uy
                .as_slice_mut()
                .expect("FDTD velocity-y field must be contiguous")
                .iter_mut()
                .zip(uy0.iter())
            {
                *dst = *src;
            }
            for (dst, src) in uz
                .as_slice_mut()
                .expect("FDTD velocity-z field must be contiguous")
                .iter_mut()
                .zip(uz0.iter())
            {
                *dst = *src;
            }
        }
    }

    /// Inject pressure source directly into pressure field (for FDTD).
    ///
    /// Uses the explicitly-set source mode (`p_mode`). k-Wave's additive mode
    /// applies mass-source scaling `2·dt/(N·c₀·dx)` to the source values before
    /// adding them to the density field. This scaling is pre-computed in
    /// `prepare_pressure_source_scaling()` and stored in `p_scale_p`.
    pub fn inject_pressure_source(&self, time_index: usize, p: &mut LetoArray3<f64>) {
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

    /// Add pressure source into a destination array (for k-space source correction workflows).
    pub fn add_pressure_source_into(&self, time_index: usize, dest: &mut LetoArray3<f64>) {
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

    /// Add pressure source into a density/mass destination array (rho).
    pub(crate) fn add_pressure_source_into_density(
        &self,
        time_index: usize,
        dest: &mut impl Array3Write,
    ) {
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

                    dest.add_or_set(i, j, k, rho_val, mode);
                }
            }
        }
    }

    /// Enforce Dirichlet pressure source after field updates.
    pub fn enforce_pressure_dirichlet(&self, time_index: usize, p: &mut LetoArray3<f64>) {
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

    /// Inject force source (velocity source) into velocity fields.
    ///
    /// For `Additive` mode, applies the per-voxel k-space source correction
    /// `κ(i,j,k) = ifftshift(cos(c_ref·|k|·dt/2))[i,j,k]` precomputed by
    /// `set_velocity_source_kappa()`. This matches k-Wave's additive velocity
    /// source injection (Treeby & Cox 2010). `AdditiveNoCorrection` bypasses the
    /// k-space filter for direct injection.
    pub fn inject_force_source(
        &self,
        time_index: usize,
        ux: &mut LetoArray3<f64>,
        uy: &mut LetoArray3<f64>,
        uz: &mut LetoArray3<f64>,
    ) {
        if let Some(signal) = &self.source.u_signal {
            if time_index < signal.shape()[2] {
                let mode = self.source.u_mode;
                let is_scalar_signal = signal.shape()[1] == 1 && self.u_indices.len() > 1;
                let has_kappa = !self.u_kappa.is_empty();
                // Per-source-point per-axis k-Wave additive scale `2·c₀·Δt/Δα`.
                // Vectors are populated by `prepare_velocity_source_scaling`;
                // when empty (e.g. unit tests that bypass solver setup) we
                // fall back to scale = 1 so existing test fixtures continue
                // to behave as before. Dirichlet always uses scale = 1
                // because the signal IS the velocity value.
                let has_scale = !self.u_scale_x.is_empty();

                for (idx, &(i, j, k, weight)) in self.u_indices.iter().enumerate() {
                    let sig_idx = if is_scalar_signal { 0 } else { idx };
                    let kappa = if has_kappa && mode == SourceMode::Additive {
                        self.u_kappa[idx]
                    } else {
                        1.0
                    };
                    let (sx, sy, sz) = if has_scale {
                        (
                            self.u_scale_x[idx],
                            self.u_scale_y[idx],
                            self.u_scale_z[idx],
                        )
                    } else {
                        (1.0, 1.0, 1.0)
                    };
                    let val_x = sx * kappa * weight * signal[[0, sig_idx, time_index]];
                    let val_y = sy * kappa * weight * signal[[1, sig_idx, time_index]];
                    let val_z = sz * kappa * weight * signal[[2, sig_idx, time_index]];

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
}
