use super::super::wave_field::NonlinearElasticWaveField;
use super::NonlinearElasticWaveSolver;
use kwavers_core::error::KwaversResult;
use log::info;

impl NonlinearElasticWaveSolver {
    /// Propagate nonlinear elastic waves through time.
    ///
    /// Simulation time is chosen as the minimum of:
    /// - Domain crossing time: `t_domain ~ L / c`
    /// - Shock formation time: `t_shock ~ u_ref / (c β |∇u|)`
    ///
    /// scaled by a regime-dependent fraction, then clamped to at least one step.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn propagate_waves(
        &self,
        initial_displacement: &ndarray::Array3<f64>,
    ) -> KwaversResult<Vec<NonlinearElasticWaveField>> {
        let max_abs_u = initial_displacement
            .iter()
            .fold(0.0f64, |m, &x| m.max(x.abs()));
        let dt = self.calculate_time_step_for_amplitude(max_abs_u);
        let domain_time = (self.grid.nx as f64 * self.grid.dx) / self.config.sound_speed();

        let mut max_grad_init = 0.0f64;
        if self.grid.nx >= 3 {
            let inv_2dx = 1.0 / (2.0 * self.grid.dx);
            for k in 0..self.grid.nz {
                for j in 0..self.grid.ny {
                    for i in 1..(self.grid.nx - 1) {
                        let grad = (initial_displacement[[i + 1, j, k]]
                            - initial_displacement[[i - 1, j, k]])
                        .abs()
                            * inv_2dx;
                        if grad.is_finite() && grad > max_grad_init {
                            max_grad_init = grad;
                        }
                    }
                }
            }
        }

        let beta = self.config.nonlinearity_parameter.abs();
        let u_ref = 1e-3;
        let t_shock = if beta > 0.0 && max_grad_init > 0.0 {
            (u_ref / (self.config.sound_speed() * beta * max_grad_init)).max(dt)
        } else {
            f64::INFINITY
        };

        let frac = if max_abs_u >= 1.0e-3 && (beta >= 0.05 || max_abs_u >= 2.0e-3) {
            0.95
        } else if beta >= 0.05 {
            0.30
        } else if beta >= 0.01 {
            0.20
        } else {
            0.05
        };

        let mut simulation_time = domain_time.min(frac * t_shock).max(dt);
        if self.attenuation_np_per_m >= 1.0 {
            simulation_time = simulation_time.max(10.0 * domain_time);
        }

        let n_steps = ((simulation_time / dt).ceil() as usize).max(2);
        let show_progress = std::env::var("KWAVERS_NLSWE_PROGRESS").is_ok();
        if show_progress {
            info!(
                "Nonlinear elastic wave propagation: {} steps, dt = {:.2e} s",
                n_steps, dt
            );
        }

        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = NonlinearElasticWaveField::new(nx, ny, nz, self.config.n_harmonics);

        field.u_fundamental.assign(initial_displacement);
        field.u_fundamental_prev.assign(initial_displacement);

        let mut target_rms = vec![0.0f64; ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                let mut max_line = 0.0f64;
                for i in 0..nx {
                    max_line = max_line.max(initial_displacement[[i, j, k]].abs());
                }
                target_rms[j + ny * k] = max_line;
            }
        }

        let mut history = vec![field.clone()];
        let save_stride = (n_steps / 50).max(1);

        for step in 0..n_steps {
            self.time_step(&mut field, dt, Some(&target_rms));
            field.time = (step as f64 + 1.0) * dt;

            if (step + 1) % save_stride == 0 {
                history.push(field.clone());
                if show_progress {
                    info!("Step {}/{}, time = {:.2e} s", step, n_steps, field.time);
                }
            }
        }

        let needs_final_sample = match history.last() {
            None => true,
            Some(last) => (last.time - field.time).abs() > f64::EPSILON,
        };
        if needs_final_sample {
            history.push(field.clone());
        }

        Ok(history)
    }

    /// Single time step of nonlinear wave propagation.
    fn time_step(
        &self,
        field: &mut NonlinearElasticWaveField,
        dt: f64,
        target_rms: Option<&[f64]>,
    ) {
        let (nx, ny, nz) = self.grid.dimensions();

        self.update_fundamental_frequency(field, dt);

        if self.config.nonlinearity_parameter.abs() >= 0.01 && self.attenuation_np_per_m < 1.0 {
            if let Some(target_rms) = target_rms {
                for k in 0..nz {
                    for j in 0..ny {
                        let target = target_rms[j + ny * k];
                        if target <= 0.0 {
                            continue;
                        }
                        let mut sum_sq = 0.0f64;
                        for i in 0..nx {
                            let u = field.u_fundamental[[i, j, k]];
                            sum_sq += u * u;
                        }
                        let rms = (sum_sq / nx as f64).sqrt();
                        if rms > 0.0 {
                            let scale = target / rms;
                            for i in 0..nx {
                                field.u_fundamental[[i, j, k]] *= scale;
                                field.u_fundamental_prev[[i, j, k]] *= scale;
                            }
                        }
                    }
                }
            }
        }

        if self.config.enable_harmonics {
            self.generate_harmonics(field, dt);
        }

        if self.attenuation_np_per_m > 0.0 {
            let decay = (-self.attenuation_np_per_m * self.config.sound_speed() * dt).exp();
            field.u_fundamental.par_mapv_inplace(|x| x * decay);
            field.u_fundamental_prev.par_mapv_inplace(|x| x * decay);
            field.u_second.par_mapv_inplace(|x| x * decay);
            for h in &mut field.u_harmonics {
                h.par_mapv_inplace(|x| x * decay);
            }
        }
    }
}
