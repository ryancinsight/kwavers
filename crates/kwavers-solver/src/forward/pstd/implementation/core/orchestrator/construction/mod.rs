use kwavers_core::constants::numerical::TWO_PI;
mod ivp_velocity;

use super::{pstd_source_gain, pstd_source_time_shift_samples, PSTDSolver};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::boundary::{CPMLBoundary, DomainPMLBoundary};
use kwavers_domain::field::wave::WaveFields;
use kwavers_grid::Grid;
use kwavers_domain::medium::MaterialFields;
use kwavers_domain::medium::Medium;
use kwavers_domain::sensor::recorder::simple::SensorRecorder;
use kwavers_domain::source::GridSource;
use kwavers_math::fft::shift_operators::generate_shift_1d;
use crate::fdtd::SourceHandler;
use crate::forward::pstd::config::{
    BoundaryConfig, CompatibilityMode, KSpaceMethod, PSTDConfig,
};
use crate::forward::pstd::implementation::k_space::{PSTDKSGrid, PSTDKSOperators};
use crate::forward::pstd::numerics::operators::initialize_spectral_operators;
use crate::forward::pstd::numerics::spectral_correction::SpectralCorrectionMethod;
use crate::forward::pstd::physics::absorption::initialize_absorption_operators;
use crate::forward::pstd::propagator::axisymmetric::AsContext;
use crate::forward::pstd::utils::compute_k_magnitude;
use crate::geometry::SolverGeometry;
use ndarray::{s, Array3, Zip};
use std::sync::Arc;

impl PSTDSolver {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        mut config: PSTDConfig,
        grid: Grid,
        medium: &dyn Medium,
        source: GridSource,
    ) -> KwaversResult<Self> {
        let grid = Arc::new(grid);
        let nz_c = grid.nz / 2 + 1; // half-spectrum length for r2c z-axis
        if config.compatibility_mode == CompatibilityMode::Reference {
            config.spectral_correction.method = SpectralCorrectionMethod::Treeby2010;
        }

        let (mut k_ops, kappa, k_max, c_ref) =
            initialize_spectral_operators(&config, &grid, medium)?;
        let mut k_mag = compute_k_magnitude(&k_ops.kx, &k_ops.ky, &k_ops.kz);

        let _x_max: f64 = 0.0;
        let boundary: Option<Box<dyn kwavers_domain::boundary::Boundary>> = match &config.boundary {
            BoundaryConfig::PML(pml_config) => {
                Some(Box::new(DomainPMLBoundary::new(pml_config.clone())?))
            }
            BoundaryConfig::CPML(cpml_config) => Some(Box::new(CPMLBoundary::new_with_time_step(
                cpml_config.clone(),
                &grid,
                c_ref,
                Some(config.dt),
            )?)),
            BoundaryConfig::None => None,
        };
        // Precompute split-field PML exp factors from the boundary's sigma profiles.
        // Populated once here; zero per-step allocation; `None` for non-CPML boundaries.
        let pml_exp = boundary.as_ref().and_then(|b| b.pml_exp_factors_owned());

        let mut absorption =
            initialize_absorption_operators(&config, &grid, medium, &k_mag, k_max, c_ref)?;
        // Truncate spectral nabla operators to the r2c half-spectrum (z-axis: nz_c = nz/2+1).
        // nabla1 = |k|^(y-2) and nabla2 = |k|^(y-1) are symmetric under kz sign flip,
        // so the first nz_c z-values are the complete independent set for real-input fields.
        if let Some(ref mut abs) = absorption {
            abs.nabla1 = abs.nabla1.slice(s![.., .., ..nz_c]).to_owned();
            abs.nabla2 = abs.nabla2.slice(s![.., .., ..nz_c]).to_owned();
        }
        // Capture the raw |k| half-spectrum (r2c z-axis: nz_c = nz/2+1) BEFORE the
        // kappa cosine transform overwrites k_mag. Needed to build the broadband
        // residual-gas absorption spectral shape ĝ(c·|k|) on demand.
        let k_mag_half = k_mag.slice(s![.., .., ..nz_c]).to_owned();
        k_mag.par_mapv_inplace(|k| (0.5 * c_ref * config.dt * k).cos());
        // source_kappa still has shape (nx, ny, nz) here — set_velocity_source_kappa
        // needs the full array to perform ifftshift indexing (kk = (k+nz/2)%nz).
        let source_kappa = k_mag;

        // Truncate kappa and source_kappa to the r2c half-spectrum AFTER source_handler
        // has extracted per-source-point values from the full array.
        // kappa = sinc(c_ref·|k|·dt) is symmetric; source_kappa = cos(c_ref·|k|·dt/2) is
        // symmetric. Both are evaluated only for kz in [0, nz_c) during every solver step.
        // Truncating saves nx·ny·(nz/2-1)·8 bytes per array at construction time.
        // NOTE: must happen AFTER set_velocity_source_kappa() call below.
        //
        // Also truncate the anti-aliasing filter to the r2c half-spectrum.
        // The filter is real-valued and symmetric (depends on |k|), so values at
        // kz ∈ [0, nz/2] (indices [0, nz_c)) are the complete independent set.
        if let Some(ref mut f) = k_ops.filter {
            *f = f.slice(s![.., .., ..nz_c]).to_owned();
        }
        let field_arrays =
            crate::forward::pstd::data::initialize_field_arrays(&grid, medium)?;

        let dk_x = TWO_PI / (grid.nx as f64 * grid.dx);
        let dk_y = TWO_PI / (grid.ny as f64 * grid.dy);
        let dk_z = TWO_PI / (grid.nz as f64 * grid.dz);
        let (ddx_k_shift_pos, ddx_k_shift_neg) = generate_shift_1d(grid.nx, dk_x, grid.dx);
        let (ddy_k_shift_pos, ddy_k_shift_neg) = generate_shift_1d(grid.ny, dk_y, grid.dy);
        // Z-axis: generate full-length shifts, then truncate to nz_c = nz/2+1.
        // generate_shift_1d produces i·k·exp(±i·k·ds/2) for k in rfftfreq order
        // (indices [0, nz_c) cover non-negative kz exactly), so prefix truncation is exact.
        let (ddz_full_pos, ddz_full_neg) = generate_shift_1d(grid.nz, dk_z, grid.dz);
        let ddz_k_shift_pos = ddz_full_pos.slice(s![..nz_c]).to_owned();
        let ddz_k_shift_neg = ddz_full_neg.slice(s![..nz_c]).to_owned();

        let shape = (grid.nx, grid.ny, grid.nz);
        // `bon` is populated only when nonlinearity is active; `None` otherwise.
        // Saves N×8 bytes for the common linear case.
        let (rho0, c0, bon) = if medium.is_homogeneous() {
            let bon = config
                .nonlinearity
                .then(|| Array3::from_elem(shape, medium.nonlinearity(0, 0, 0)));
            (
                Array3::from_elem(shape, medium.density(0, 0, 0)),
                Array3::from_elem(shape, medium.sound_speed(0, 0, 0)),
                bon,
            )
        } else {
            let mut rho0 = Array3::zeros(shape);
            let mut c0 = Array3::zeros(shape);
            let mut bon = config.nonlinearity.then(|| Array3::<f64>::zeros(shape));

            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                        rho0[[i, j, k]] = kwavers_domain::medium::density_at(medium, x, y, z, &grid);
                        c0[[i, j, k]] =
                            kwavers_domain::medium::sound_speed_at(medium, x, y, z, &grid);
                        if let Some(ref mut b) = bon {
                            b[[i, j, k]] =
                                kwavers_domain::medium::nonlinearity_at(medium, x, y, z, &grid);
                        }
                    }
                }
            }

            (rho0, c0, bon)
        };

        let sensor_recorder =
            SensorRecorder::new(config.sensor_mask.as_ref(), shape, config.nt + 1)?;
        let mut source_handler = SourceHandler::new(source, &grid)?;
        if source_handler.has_velocity_source() {
            // set_velocity_source_kappa uses ifftshift indexing (kk = (k+nz/2)%nz),
            // requiring the full (nx,ny,nz) source_kappa. Must be called BEFORE truncation.
            source_handler.set_velocity_source_kappa(&source_kappa, grid.nx, grid.ny, grid.nz);
        }

        // Truncate kappa and source_kappa to the r2c half-spectrum after source_handler
        // has extracted its per-source-point values. Every subsequent solver use operates
        // on the half-spectrum only: nz_c = nz/2+1 rather than nz.
        // Saves: nx·ny·(nz/2-1)·8 bytes per array
        //   (≈1 MB/array at 64³, 8 MB at 128³, 66 MB at 256³).
        let kappa = kappa.slice(s![.., .., ..nz_c]).to_owned();
        let source_kappa = source_kappa.slice(s![.., .., ..nz_c]).to_owned();

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
            fft: kwavers_math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz),
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
            pml_exp,
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
            // Half-spectrum k-space buffers: r2c z-axis reduces nz → nz_c = nz/2+1.
            // ux_k is the shared velocity k-space scratch; uy_k and uz_k are eliminated
            // (Opt-8) — all three axes are processed sequentially, one buffer suffices.
            ux_k: Array3::zeros((grid.nx, grid.ny, nz_c)),
            grad_k: Array3::zeros((grid.nx, grid.ny, nz_c)),
            materials: MaterialFields { rho0, c0 },
            bon,
            absorption,
            k_mag_half,
            residual_gas_absorption: None,
            kspace_operators: None,
            ddx_k_shift_pos,
            ddy_k_shift_pos,
            ddz_k_shift_pos,
            ddx_k_shift_neg,
            ddy_k_shift_neg,
            ddz_k_shift_neg,
            dpx: Array3::zeros(shape),
            dpy: Array3::zeros(shape),
            // dpz eliminated (Opt-12): dpx is reused for all three velocity gradient axes
            // and for the absorption L1 accumulator — axes are sequential, no overlap.
            div_u: Array3::zeros(shape),
            div_ux: Array3::zeros(shape),
            div_uy: Array3::zeros(shape),
            div_uz: Array3::zeros(shape),
            as_ctx: None,
            alpha_np_m: None, // allocated on demand by populate_alpha_np_m_at_frequency / set_alpha_np_m
            dirichlet_pml_bypass_x: Vec::new(),
            pml_bypass_plane_scratch: Array3::zeros((0, grid.ny, grid.nz)),
        };

        if solver.config.kspace_method == KSpaceMethod::FullKSpace {
            let k_grid = PSTDKSGrid::from_spatial_grid(&grid)?;
            solver.kspace_operators = Some(PSTDKSOperators::new(k_grid));
        }

        if solver.config.geometry == SolverGeometry::CylindricalAS {
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
        solver.source_handler.prepare_velocity_source_scaling(
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

        // Split the total density perturbation (rho = p0/c²) equally among the active
        // spatial dimensions.  For NZ=1 the z-divergence of uz is identically zero
        // (k_z = 0 under periodic DFT) and the z-directional PML profile is neutral
        // (sigma_z = 0; kernels.rs:52 early-returns set_neutral when n ≤ 1).  If rhoz
        // were initialised to p0/(3c²) it would never evolve, permanently contributing
        // a static pressure of p0/3 to the EOS (p = c²·(ρx+ρy+ρz)) and inflating the
        // RMS at every sensor after the acoustic wave has passed.  k-Wave's 2-D solver
        // uses only ρx and ρy (two-component split), so its pressure returns to zero;
        // the three-component kwavers split produces a spurious DC background instead.
        //
        // General rule: split among active dimensions only.
        //   3-D (nz > 1): ρx = ρy = ρz = ρ/3
        //   2-D (ny > 1, nz = 1): ρx = ρy = ρ/2, ρz = 0
        //   1-D (ny = 1, nz = 1): ρx = ρ,    ρy = ρz = 0
        let has_y_dim = solver.grid.ny > 1;
        let has_z_dim = solver.grid.nz > 1;
        let n_active = 1 + has_y_dim as usize + has_z_dim as usize;
        let divisor = n_active as f64;
        Zip::from(&mut solver.rhox)
            .and(&mut solver.rhoy)
            .and(&mut solver.rhoz)
            .and(&solver.div_u)
            .par_for_each(|rx, ry, rz, &rho| {
                let share = rho / divisor;
                *rx = share;
                *ry = if has_y_dim { share } else { 0.0 };
                *rz = if has_z_dim { share } else { 0.0 };
            });

        if solver.source_handler.has_initial_pressure()
            && !solver.source_handler.has_initial_velocity()
        {
            solver.initialize_ivp_velocity()?;
        }

        Ok(solver)
    }
}
