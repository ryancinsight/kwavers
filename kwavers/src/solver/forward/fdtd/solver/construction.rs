//! `GenericFdtdSolver::new` constructor.
//!
//! Validates spatial order, builds central-difference and staggered-grid
//! operators, precomputes per-cell material properties (`ρ₀`, `c₀`,
//! `ρ₀·c₀²`), prepares the source handler, optionally constructs k-space
//! correction operators, applies initial pressure/velocity conditions,
//! and pre-allocates Westervelt and divergence/gradient scratch buffers.

use log::info;
use ndarray::{Array3, Zip};

use super::central_diff::CentralDifferenceOperator;
use super::{FdtdMetrics, GenericFdtdSolver};
use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use crate::domain::field::wave::WaveFields;
use crate::domain::grid::Grid;
use crate::domain::medium::{MaterialFields, Medium};
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::grid_source::GridSource;
use crate::math::numerics::operators::StaggeredGridOperator;
use crate::physics::acoustics::mechanics::acoustic_wave::SpatialOrder;

use super::super::config::{FdtdConfig, KSpaceCorrectionMode};
use super::super::kspace_correction::KSpaceFdtdOperators;
use super::super::source_handler::SourceHandler;

impl GenericFdtdSolver<Array3<f64>> {
    /// Create a new FDTD solver
    pub fn new(
        config: FdtdConfig,
        grid: &Grid,
        medium: &dyn Medium,
        source: GridSource,
    ) -> KwaversResult<Self> {
        info!("Initializing FDTD solver with config: {:?}", config);

        // Validate spatial order by converting to enum
        let spatial_order = SpatialOrder::from_usize(config.spatial_order)?;

        let central_operator =
            CentralDifferenceOperator::new(config.spatial_order, grid.dx, grid.dy, grid.dz)?;
        let staggered_operator = StaggeredGridOperator::new(grid.dx, grid.dy, grid.dz)?;

        let source_handler = SourceHandler::new(source, grid)?;
        let sensor_recorder = SensorRecorder::new(
            config.sensor_mask.as_ref(),
            (grid.nx, grid.ny, grid.nz),
            config.nt + 1,
        )?;

        // Initialize fields
        let shape = (grid.nx, grid.ny, grid.nz);
        let mut fields = WaveFields::new(shape);
        let mut materials = MaterialFields::new(shape);

        // Pre-compute material properties
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    materials.rho0[[i, j, k]] =
                        crate::domain::medium::density_at(medium, x, y, z, grid);
                    materials.c0[[i, j, k]] =
                        crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                }
            }
        }

        // Pre-compute rho * c^2 element-wise
        let mut rho_c_squared = Array3::<f64>::zeros(shape);
        Zip::from(&mut rho_c_squared)
            .and(&materials.rho0)
            .and(&materials.c0)
            .for_each(|rho_c_sq, &rho, &c| {
                *rho_c_sq = rho * c * c;
            });

        // Precompute k-Wave compatible pressure and velocity source scaling
        let mut source_handler = source_handler;
        source_handler.prepare_pressure_source_scaling(grid, &materials.c0, config.dt);
        source_handler.prepare_velocity_source_scaling(grid, &materials.c0, config.dt);

        // Initialize k-space correction operators when requested.
        // c_ref = mean sound speed over all grid cells (same convention as PSTD).
        let mut kspace_ops = if config.kspace_correction == KSpaceCorrectionMode::Spectral {
            let c_sum: f64 = materials.c0.iter().sum();
            let c_ref = c_sum / materials.c0.len() as f64;
            Some(KSpaceFdtdOperators::new(
                grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz, c_ref, config.dt,
            ))
        } else {
            None
        };

        // Apply initial conditions (p0, u0) — mirrors PSTD solver behaviour
        let mut rho_init = Array3::zeros(shape);
        source_handler.apply_initial_conditions(
            &mut fields.p,
            &mut rho_init,
            &materials.c0,
            &mut fields.ux,
            &mut fields.uy,
            &mut fields.uz,
        );
        // Note: FDTD uses a single rho field so no split needed (cf. PSTD rhox/rhoy/rhoz)

        if source_handler.has_initial_pressure()
            && !source_handler.has_initial_velocity()
            && matches!(config.kspace_correction, KSpaceCorrectionMode::Spectral)
        {
            let rho0_ref = materials.rho0.mean().unwrap_or(1000.0);
            let Some(kspace_ops) = kspace_ops.as_mut() else {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "kspace_correction".to_string(),
                    value: "spectral".to_string(),
                    constraint:
                        "spectral k-space correction requires precomputed k-space operators"
                            .to_string(),
                }));
            };
            kspace_ops.initialize_ivp_velocity(
                &fields.p,
                config.dt,
                rho0_ref,
                &mut fields.ux,
                &mut fields.uy,
                &mut fields.uz,
            )?;
        }

        // Precompute nonlinear medium property arrays (only when nonlinear mode is on)
        let (p_prev, p_prev2, nl_scratch, nl_coeff) = if config.enable_nonlinear {
            let mut beta = Array3::<f64>::zeros(shape);
            let mut c4 = Array3::<f64>::zeros(shape);
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                        let bn = crate::domain::medium::nonlinearity_at(medium, x, y, z, grid);
                        let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                        // β = 1 + B/(2A) where B/A is returned by nonlinearity_at
                        beta[[i, j, k]] = 1.0 + bn * 0.5;
                        c4[[i, j, k]] = c.powi(4);
                    }
                }
            }
            // Precompute β/(ρ₀·c₀⁴) once; used every step in the hot nonlinear kernel.
            // Reduces per-element inner-loop reads from 5 to 3, cutting memory traffic ~40%.
            // beta and c4 are intermediate; only nl_coeff is retained in the struct.
            let nl = ndarray::Zip::from(&beta)
                .and(&materials.rho0)
                .and(&c4)
                .map_collect(|&b, &rho, &c| b / (rho * c));
            (
                Some(Array3::<f64>::zeros(shape)),
                Some(Array3::<f64>::zeros(shape)),
                Some(Array3::<f64>::zeros(shape)),
                Some(nl),
            )
        } else {
            (None, None, None, None)
        };

        // Pre-allocate staggered pressure-gradient scratch buffers.
        // Shape (nx−1, ny, nz) for dp_dx, etc. — allocated once, reused every step.
        // Only created when `staggered_grid = true` and the dimension has ≥ 2 points.
        let (dp_dx_scratch, dp_dy_scratch, dp_dz_scratch) = if config.staggered_grid {
            (
                if grid.nx > 1 {
                    Some(Array3::<f64>::zeros((grid.nx - 1, grid.ny, grid.nz)))
                } else {
                    None
                },
                if grid.ny > 1 {
                    Some(Array3::<f64>::zeros((grid.nx, grid.ny - 1, grid.nz)))
                } else {
                    None
                },
                if grid.nz > 1 {
                    Some(Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz - 1)))
                } else {
                    None
                },
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            config,
            grid: grid.clone(),
            central_operator,
            staggered_operator,
            metrics: FdtdMetrics::new(),
            cpml_boundary: None,
            spatial_order,
            gpu_accelerator: None,
            source_handler,
            dynamic_sources: Vec::new(),
            source_injection_modes: Vec::new(),
            sensor_recorder,
            time_step_index: 0,
            fields,
            materials,
            rho_c_squared,
            p_prev,
            p_prev2,
            nl_scratch,
            nl_coeff,
            kspace_ops,
            // Pre-allocate divergence scratch buffers — one-time cost; zero per-step alloc
            dvx_scratch: Array3::<f64>::zeros(shape),
            dvy_scratch: Array3::<f64>::zeros(shape),
            divergence_scratch: Array3::<f64>::zeros(shape),
            dp_dx_scratch,
            dp_dy_scratch,
            dp_dz_scratch,
        })
    }
}
