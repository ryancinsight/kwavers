use super::{pstd_source_gain, pstd_source_time_shift_samples, PSTDSolver};
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::{CPMLBoundary, PMLBoundary};
use crate::domain::field::wave::WaveFields;
use crate::domain::grid::Grid;
use crate::domain::medium::MaterialFields;
use crate::domain::medium::Medium;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::GridSource;
use crate::math::fft::shift_operators::generate_shift_1d;
use crate::solver::fdtd::SourceHandler;
use crate::solver::forward::pstd::config::{
    BoundaryConfig, CompatibilityMode, KSpaceMethod, PSTDConfig,
};
use crate::solver::forward::pstd::implementation::k_space::{PSTDKSGrid, PSTDKSOperators};
use crate::solver::forward::pstd::numerics::operators::initialize_spectral_operators;
use crate::solver::forward::pstd::numerics::spectral_correction::CorrectionMethod;
use crate::solver::forward::pstd::physics::absorption::initialize_absorption_operators;
use crate::solver::forward::pstd::propagator::axisymmetric::AsContext;
use crate::solver::forward::pstd::utils::compute_k_magnitude;
use crate::solver::geometry::Geometry;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;
use std::sync::Arc;

impl PSTDSolver {
    pub fn new(
        mut config: PSTDConfig,
        grid: Grid,
        medium: &dyn Medium,
        source: GridSource,
    ) -> KwaversResult<Self> {
        let grid = Arc::new(grid);
        if config.compatibility_mode == CompatibilityMode::Reference {
            config.spectral_correction.method = CorrectionMethod::Treeby2010;
        }

        let (k_ops, kappa, k_max, c_ref) = initialize_spectral_operators(&config, &grid, medium)?;
        let mut k_mag = compute_k_magnitude(&k_ops.kx, &k_ops.ky, &k_ops.kz);

        let _x_max: f64 = 0.0;
        let boundary: Option<Box<dyn crate::domain::boundary::Boundary>> = match &config.boundary {
            BoundaryConfig::PML(pml_config) => {
                Some(Box::new(PMLBoundary::new(pml_config.clone())?))
            }
            BoundaryConfig::CPML(cpml_config) => Some(Box::new(CPMLBoundary::new_with_time_step(
                cpml_config.clone(),
                &grid,
                c_ref,
                Some(config.dt),
            )?)),
            BoundaryConfig::None => None,
        };

        let absorption =
            initialize_absorption_operators(&config, &grid, medium, &k_mag, k_max, c_ref)?;
        k_mag.mapv_inplace(|k| (0.5 * c_ref * config.dt * k).cos());
        let source_kappa = k_mag;
        let field_arrays =
            crate::solver::forward::pstd::data::initialize_field_arrays(&grid, medium)?;

        let dk_x = 2.0 * PI / (grid.nx as f64 * grid.dx);
        let dk_y = 2.0 * PI / (grid.ny as f64 * grid.dy);
        let dk_z = 2.0 * PI / (grid.nz as f64 * grid.dz);
        let (ddx_k_shift_pos, ddx_k_shift_neg) = generate_shift_1d(grid.nx, dk_x, grid.dx);
        let (ddy_k_shift_pos, ddy_k_shift_neg) = generate_shift_1d(grid.ny, dk_y, grid.dy);
        let (ddz_k_shift_pos, ddz_k_shift_neg) = generate_shift_1d(grid.nz, dk_z, grid.dz);

        let shape = (grid.nx, grid.ny, grid.nz);
        let (rho0, c0, bon) = if medium.is_homogeneous() {
            (
                Array3::from_elem(shape, medium.density(0, 0, 0)),
                Array3::from_elem(shape, medium.sound_speed(0, 0, 0)),
                Array3::from_elem(shape, medium.nonlinearity(0, 0, 0)),
            )
        } else {
            let mut rho0 = Array3::zeros(shape);
            let mut c0 = Array3::zeros(shape);
            let mut bon = Array3::zeros(shape);

            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                        rho0[[i, j, k]] = crate::domain::medium::density_at(medium, x, y, z, &grid);
                        c0[[i, j, k]] =
                            crate::domain::medium::sound_speed_at(medium, x, y, z, &grid);
                        bon[[i, j, k]] =
                            crate::domain::medium::nonlinearity_at(medium, x, y, z, &grid);
                    }
                }
            }

            (rho0, c0, bon)
        };

        let sensor_recorder =
            SensorRecorder::new(config.sensor_mask.as_ref(), shape, config.nt + 1)?;
        let mut source_handler = SourceHandler::new(source, &grid)?;
        if source_handler.has_velocity_source() {
            source_handler.set_velocity_source_kappa(&source_kappa, grid.nx, grid.ny, grid.nz);
        }

        let config_dt = config.dt;

        let mut solver = Self {
            config,
            grid: grid.clone(),
            sensor_recorder,
            source_handler,
            dynamic_sources: Vec::new(),
            source_injection_modes: Vec::new(),
            velocity_source_grad_masks: Vec::new(),
            time_step_index: 0,
            fft: crate::math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz),
            kappa,
            source_kappa,
            filter: k_ops.filter,
            k_max,
            c_ref,
            mass_source_scale: {
                let n_dim = [grid.nx > 1, grid.ny > 1, grid.nz > 1]
                    .iter()
                    .filter(|&&d| d)
                    .count()
                    .max(1) as f64;
                let dx_min = grid.dx.min(grid.dy).min(grid.dz);
                2.0 * config_dt / (n_dim * c_ref * dx_min)
            },
            source_time_shift_samples: pstd_source_time_shift_samples(),
            source_gain: pstd_source_gain(),
            boundary,
            fields: WaveFields {
                p: field_arrays.p,
                ux: field_arrays.ux,
                uy: field_arrays.uy,
                uz: field_arrays.uz,
            },
            rhox: Array3::zeros(shape),
            rhoy: Array3::zeros(shape),
            rhoz: Array3::zeros(shape),
            p_k: field_arrays.p_k,
            ux_k: Array3::zeros(shape),
            uy_k: Array3::zeros(shape),
            uz_k: Array3::zeros(shape),
            grad_k: Array3::zeros(shape),
            materials: MaterialFields { rho0, c0 },
            bon,
            absorption,
            kspace_operators: None,
            ddx_k_shift_pos,
            ddy_k_shift_pos,
            ddz_k_shift_pos,
            ddx_k_shift_neg,
            ddy_k_shift_neg,
            ddz_k_shift_neg,
            dpx: Array3::zeros(shape),
            dpy: Array3::zeros(shape),
            dpz: Array3::zeros(shape),
            div_u: Array3::zeros(shape),
            as_ctx: None,
        };

        if solver.config.kspace_method == KSpaceMethod::FullKSpace {
            let k_grid = PSTDKSGrid::from_spatial_grid(&grid)?;
            solver.kspace_operators = Some(PSTDKSOperators::new(k_grid));
        }

        if solver.config.geometry == Geometry::CylindricalAS {
            if grid.ny != 1 {
                return Err(KwaversError::InvalidInput(
                    "CylindricalAS requires ny = 1".into(),
                ));
            }
            solver.as_ctx = Some(AsContext::new(
                grid.nx,
                grid.nz,
                grid.dx,
                grid.dz,
                c_ref,
                config_dt,
                solver.ddx_k_shift_pos.clone(),
                solver.ddx_k_shift_neg.clone(),
            )?);
        }

        solver.source_handler.prepare_pressure_source_scaling(
            &grid,
            &solver.materials.c0,
            solver.config.dt,
        );

        solver.source_handler.apply_initial_conditions(
            &mut solver.fields.p,
            &mut solver.div_u,
            &solver.materials.c0,
            &mut solver.fields.ux,
            &mut solver.fields.uy,
            &mut solver.fields.uz,
        );

        Zip::from(&mut solver.rhox)
            .and(&mut solver.rhoy)
            .and(&mut solver.rhoz)
            .and(&solver.div_u)
            .for_each(|rx, ry, rz, &rho| {
                let split = rho / 3.0;
                *rx = split;
                *ry = split;
                *rz = split;
            });

        if solver.source_handler.has_initial_pressure()
            && !solver.source_handler.has_initial_velocity()
        {
            solver.initialize_ivp_velocity()?;
        }

        Ok(solver)
    }

    /// Initialize velocity fields at t = −dt/2 for exact IVP staggered leapfrog start.
    ///
    /// The exact traveling-wave solution gives, for each Fourier mode:
    ///   ux_sgx_k[i,j,k] = ddx_k_shift_pos[i] / |k| · sin(c_ref·dt·|k|/2) / (ρ₀·c_ref) · p0_k[i,j,k]
    fn initialize_ivp_velocity(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let rho0_ref = self.materials.rho0.mean().unwrap_or(1000.0);
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        if !dt.is_finite() || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "dt must be finite and positive, got {dt}"
            )));
        }
        if !rho0_ref.is_finite() || rho0_ref <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "rho0_ref must be finite and positive, got {rho0_ref}"
            )));
        }
        let scale_prefactor = dt / (2.0 * rho0_ref);
        Zip::from(&mut self.div_u)
            .and(&self.source_kappa)
            .for_each(|scale, &kap| {
                let theta = kap.clamp(-1.0, 1.0).acos();
                *scale = if theta < 1e-30 {
                    scale_prefactor
                } else {
                    scale_prefactor * theta.sin() / theta
                };
            });

        self.fft.forward_into(&self.fields.p, &mut self.p_k);

        {
            let ddx = self.ddx_k_shift_pos.view();
            let sin_s = self.div_u.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(i, _j, _k), gk, &ss, &p| {
                    *gk = ddx[i] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_k, &mut self.fields.ux, &mut self.ux_k);

        if has_y {
            let ddy = self.ddy_k_shift_pos.view();
            let sin_s = self.div_u.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, j, _k), gk, &ss, &p| {
                    *gk = ddy[j] * ss * p;
                });
            self.fft
                .inverse_into(&self.grad_k, &mut self.fields.uy, &mut self.uy_k);
        } else {
            self.fields.uy.fill(0.0);
        }

        if has_z {
            let ddz = self.ddz_k_shift_pos.view();
            let sin_s = self.div_u.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, _j, k_idx), gk, &ss, &p| {
                    *gk = ddz[k_idx] * ss * p;
                });
            self.fft
                .inverse_into(&self.grad_k, &mut self.fields.uz, &mut self.uz_k);
        } else {
            self.fields.uz.fill(0.0);
        }

        Ok(())
    }
}
