use super::DomainPMLBoundary;
use crate::Boundary;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use log::trace;
use ndarray::{Array3, ArrayViewMut3};

impl Boundary for DomainPMLBoundary {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn apply_acoustic(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<()> {
        trace!("Applying spatial acoustic PML at step {}", time_step);
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;

        if 2 * t >= nx {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "PML thickness {} incompatible with grid nx={}; require 2*thickness < nx",
                        t, nx
                    ),
                },
            ));
        }
        if ny > 1 && 2 * t >= ny {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid ny={}; require 2*thickness < ny for y-PML",
                    t, ny
                ),
            }));
        }
        if nz > 1 && 2 * t >= nz {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid nz={}; require 2*thickness < nz for z-PML",
                    t, nz
                ),
            }));
        }

        let apply_y = ny > 1;
        let apply_z = nz > 1;

        let exp_x = Self::precompute_exp_factors(&self.acoustic_damping_x);
        let exp_y_full = if apply_y {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_y),
                ny,
                t,
            )
        } else {
            vec![1.0; ny]
        };
        let exp_z_full = if apply_z {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_z),
                nz,
                t,
            )
        } else {
            vec![1.0; nz]
        };

        for i in 0..t {
            let fx = exp_x[i];
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    field[[i, j, k]] *= fx.min(fy).min(fz);
                }
            }
            let ri = nx - 1 - i;
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    field[[ri, j, k]] *= fx.min(fy).min(fz);
                }
            }
        }

        let x_start = t;
        let x_end = nx - t;

        if apply_y && x_end > x_start {
            let exp_y = Self::precompute_exp_factors(&self.acoustic_damping_y);
            for j in 0..t {
                let fy = exp_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        field[[i, j, k]] *= fy.min(fz);
                    }
                }
                let rj = ny - 1 - j;
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        field[[i, rj, k]] *= fy.min(fz);
                    }
                }
            }

            let y_start = t;
            let y_end = ny - t;

            if apply_z && y_end > y_start {
                let exp_z = Self::precompute_exp_factors(&self.acoustic_damping_z);
                for k in 0..t {
                    let fz = exp_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, k]] *= fz;
                        }
                    }
                    let rk = nz - 1 - k;
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, rk]] *= fz;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<num_complex::Complex<f64>>,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<()> {
        trace!(
            "Applying frequency domain acoustic PML at step {}",
            time_step
        );
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;

        if 2 * t >= nx {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "PML thickness {} incompatible with grid nx={}; require 2*thickness < nx",
                        t, nx
                    ),
                },
            ));
        }
        if ny > 1 && 2 * t >= ny {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid ny={}; require 2*thickness < ny for y-PML",
                    t, ny
                ),
            }));
        }
        if nz > 1 && 2 * t >= nz {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid nz={}; require 2*thickness < nz for z-PML",
                    t, nz
                ),
            }));
        }

        let apply_y = ny > 1;
        let apply_z = nz > 1;

        let exp_x = Self::precompute_exp_factors(&self.acoustic_damping_x);
        let exp_y_full = if apply_y {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_y),
                ny,
                t,
            )
        } else {
            vec![1.0; ny]
        };
        let exp_z_full = if apply_z {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_z),
                nz,
                t,
            )
        } else {
            vec![1.0; nz]
        };

        for i in 0..t {
            let fx = exp_x[i];
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    let decay = fx.min(fy).min(fz);
                    field[[i, j, k]].re *= decay;
                    field[[i, j, k]].im *= decay;
                }
            }
            let ri = nx - 1 - i;
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    let decay = fx.min(fy).min(fz);
                    field[[ri, j, k]].re *= decay;
                    field[[ri, j, k]].im *= decay;
                }
            }
        }

        let x_start = t;
        let x_end = nx - t;

        if apply_y && x_end > x_start {
            let exp_y = Self::precompute_exp_factors(&self.acoustic_damping_y);
            for j in 0..t {
                let fy = exp_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        let decay = fy.min(fz);
                        field[[i, j, k]].re *= decay;
                        field[[i, j, k]].im *= decay;
                    }
                }
                let rj = ny - 1 - j;
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        let decay = fy.min(fz);
                        field[[i, rj, k]].re *= decay;
                        field[[i, rj, k]].im *= decay;
                    }
                }
            }

            let y_start = t;
            let y_end = ny - t;

            if apply_z && y_end > y_start {
                let exp_z = Self::precompute_exp_factors(&self.acoustic_damping_z);
                for k in 0..t {
                    let fz = exp_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, k]].re *= fz;
                            field[[i, j, k]].im *= fz;
                        }
                    }
                    let rk = nz - 1 - k;
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, rk]].re *= fz;
                            field[[i, j, rk]].im *= fz;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_light(&mut self, mut field: ArrayViewMut3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying light PML at step {}", time_step);
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;
        let apply_y = ny > 1 && 2 * t < ny;
        let apply_z = nz > 1 && 2 * t < nz;

        for i in 0..t.min(nx.saturating_sub(1)) {
            let d_x = self.light_damping_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    let d_y = if apply_y {
                        self.get_damping(j, &self.light_damping_y, ny)
                    } else {
                        0.0
                    };
                    let d_z = if apply_z {
                        self.get_damping(k, &self.light_damping_z, nz)
                    } else {
                        0.0
                    };
                    Self::apply_damping(&mut field[[i, j, k]], d_x + d_y + d_z);
                }
            }
            let ri = nx - 1 - i;
            let d_x_r = self.light_damping_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    let d_y = if apply_y {
                        self.get_damping(j, &self.light_damping_y, ny)
                    } else {
                        0.0
                    };
                    let d_z = if apply_z {
                        self.get_damping(k, &self.light_damping_z, nz)
                    } else {
                        0.0
                    };
                    Self::apply_damping(&mut field[[ri, j, k]], d_x_r + d_y + d_z);
                }
            }
        }

        let x_start = t;
        let x_end = nx - t;
        if x_end > x_start {
            for j in 0..t {
                let d_y = self.light_damping_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let d_z = self.get_damping(k, &self.light_damping_z, nz);
                        Self::apply_damping(&mut field[[i, j, k]], d_y + d_z);
                    }
                }
                let rj = ny - 1 - j;
                let d_y_r = self.light_damping_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let d_z = self.get_damping(k, &self.light_damping_z, nz);
                        Self::apply_damping(&mut field[[i, rj, k]], d_y_r + d_z);
                    }
                }
            }

            let y_start = t;
            let y_end = ny - t;
            if y_end > y_start {
                for k in 0..t {
                    let d_z = self.light_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, k]], d_z);
                        }
                    }
                    let rk = nz - 1 - k;
                    let d_z_r = self.light_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, rk]], d_z_r);
                        }
                    }
                }
            }
        }
    }
}
