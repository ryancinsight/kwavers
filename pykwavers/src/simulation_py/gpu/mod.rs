mod session;

pub use session::GpuPstdSession;

/// GPU-resident PSTD implementation (requires `gpu` feature).
///
/// When the `gpu` feature is disabled this function is compiled but never
/// reachable (the call site is inside `#[cfg(feature = "gpu")]`).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(crate) fn run_gpu_pstd_impl(
    grid: &KwaversGrid,
    medium: &MediumInner,
    time_steps: usize,
    dt: f64,
    alpha_coeff_db: f64,
    alpha_power: f64,
    grid_source: &GridSource,
    sensor: Option<&Sensor>,
    transducer_sensor: Option<&TransducerArray2D>,
    pml_size: Option<usize>,
    pml_size_xyz: Option<(usize, usize, usize)>,
    pml_inside: bool,
    pml_alpha_xyz: Option<(f64, f64, f64)>,
) -> KwaversResult<(ndarray::Array2<f64>, Option<SampledStatistics>)> {
    use kwavers::domain::boundary::cpml::{CPMLConfig, CPMLProfiles};
    use kwavers::domain::medium::traits::Medium as MediumTrait;
    use kwavers::physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
    use kwavers::solver::forward::pstd::gpu_pstd::{
        AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams,
    };
    use ndarray::Array2;

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let total = nx * ny * nz;

    if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
        return Err(KwaversError::Io(std::io::Error::other(format!(
            "GPU PSTD requires power-of-2 grid dimensions, got {nx}×{ny}×{nz}"
        ))));
    }
    if nx > 256 || ny > 256 || nz > 256 {
        return Err(KwaversError::Io(std::io::Error::other(format!(
            "GPU PSTD supports N≤256 per axis, got {nx}×{ny}×{nz}"
        ))));
    }

    let (default_thickness, max_allowed) = Simulation::cpml_thickness_limits(nx, ny, nz);
    let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

    let cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
        let mut cfg = CPMLConfig::with_per_dimension_thickness(px, py, pz);
        if let Some((ax, ay, az)) = pml_alpha_xyz {
            cfg = cfg.with_alpha_xyz(ax, ay, az);
        }
        cfg
    } else {
        let mut cfg = CPMLConfig::with_thickness(thickness);
        if let Some((ax, ay, az)) = pml_alpha_xyz {
            cfg = cfg.with_alpha_xyz(ax, ay, az);
        }
        cfg
    };

    let c_ref = medium.as_medium().max_sound_speed();
    let profiles = CPMLProfiles::new(&cpml_config, grid, c_ref, dt)?;

    let mut pml_sgx_3d = vec![1.0f32; total];
    let mut pml_sgy_3d = vec![1.0f32; total];
    let mut pml_sgz_3d = vec![1.0f32; total];
    let mut pml_x_3d = vec![1.0f32; total];
    let mut pml_y_3d = vec![1.0f32; total];
    let mut pml_z_3d = vec![1.0f32; total];

    if pml_inside {
        let pml_sgx_1d: Vec<f32> = profiles
            .sigma_x_sgx
            .iter()
            .map(|&s| (-s * dt * 0.5).exp() as f32)
            .collect();
        let pml_sgy_1d: Vec<f32> = profiles
            .sigma_y_sgy
            .iter()
            .map(|&s| (-s * dt * 0.5).exp() as f32)
            .collect();
        let pml_sgz_1d: Vec<f32> = profiles
            .sigma_z_sgz
            .iter()
            .map(|&s| (-s * dt * 0.5).exp() as f32)
            .collect();
        let pml_x_1d: Vec<f32> = profiles
            .sigma_x
            .iter()
            .map(|&s| (-s * dt * 0.5).exp() as f32)
            .collect();
        let pml_y_1d: Vec<f32> = profiles
            .sigma_y
            .iter()
            .map(|&s| (-s * dt * 0.5).exp() as f32)
            .collect();
        let pml_z_1d: Vec<f32> = profiles
            .sigma_z
            .iter()
            .map(|&s| (-s * dt * 0.5).exp() as f32)
            .collect();

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let flat = ix * ny * nz + iy * nz + iz;
                    pml_sgx_3d[flat] = pml_sgx_1d[ix];
                    pml_sgy_3d[flat] = pml_sgy_1d[iy];
                    pml_sgz_3d[flat] = pml_sgz_1d[iz];
                    pml_x_3d[flat] = pml_x_1d[ix];
                    pml_y_3d[flat] = pml_y_1d[iy];
                    pml_z_3d[flat] = pml_z_1d[iz];
                }
            }
        }
    }

    let mut c0_flat = vec![c_ref as f32; total];
    let mut rho0_flat = vec![1000.0f32; total];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let flat = ix * ny * nz + iy * nz + iz;
                c0_flat[flat] = medium.as_medium().sound_speed(ix, iy, iz) as f32;
                rho0_flat[flat] = medium.as_medium().density(ix, iy, iz) as f32;
            }
        }
    }

    let sensor_mask = Simulation::create_sensor_mask(grid, sensor, transducer_sensor);
    let mut sensor_indices: Vec<u32> = Vec::new();
    {
        let flat = sensor_mask
            .as_slice()
            .expect("sensor_mask must be C-contiguous");
        for (i, &v) in flat.iter().enumerate() {
            if v {
                sensor_indices.push(i as u32);
            }
        }
    }

    let n_dim_active = [nx > 1, ny > 1, nz > 1]
        .iter()
        .filter(|&&d| d)
        .count()
        .max(1);
    let dx_min = grid.dx.min(grid.dy).min(grid.dz);
    let mass_source_scale = 2.0 * dt / (n_dim_active as f64 * c_ref * dx_min);
    let density_scale = n_dim_active as f64 / 3.0;
    let combined_scale = (mass_source_scale * density_scale) as f32;

    let mut source_indices: Vec<u32> = Vec::new();
    let mut source_signals: Vec<f32> = Vec::new();

    if let (Some(p_mask), Some(p_signal)) = (&grid_source.p_mask, &grid_source.p_signal) {
        let mask_flat = p_mask.as_slice().expect("p_mask must be C-contiguous");
        for (i, &v) in mask_flat.iter().enumerate() {
            if v != 0.0 {
                source_indices.push(i as u32);
            }
        }
        let n_src = source_indices.len();
        let n_sig_rows = p_signal.shape()[0];
        let n_sig_cols = p_signal.shape()[1].min(time_steps);
        source_signals = vec![0.0f32; n_src * time_steps];
        for (src_idx, _) in source_indices.iter().enumerate() {
            let sig_row = if n_sig_rows == 1 {
                0
            } else {
                src_idx.min(n_sig_rows - 1)
            };
            for step in 0..n_sig_cols {
                source_signals[src_idx * time_steps + step] =
                    (p_signal[[sig_row, step]] * combined_scale as f64) as f32;
            }
        }
    }

    let mut vel_x_indices: Vec<u32> = Vec::new();
    let mut vel_x_signals: Vec<f32> = Vec::new();

    if let (Some(u_mask), Some(u_signal)) = (&grid_source.u_mask, &grid_source.u_signal) {
        let mask_flat = u_mask.as_slice().expect("u_mask must be C-contiguous");
        for (i, &v) in mask_flat.iter().enumerate() {
            if v != 0.0 {
                vel_x_indices.push(i as u32);
            }
        }
        let n_vel = vel_x_indices.len();
        let n_sig_srcs = u_signal.shape()[1];
        let n_sig_cols = u_signal.shape()[2].min(time_steps);
        vel_x_signals = vec![0.0f32; n_vel * time_steps];
        for src_idx in 0..n_vel {
            let sig_row = src_idx.min(n_sig_srcs.saturating_sub(1));
            for step in 0..n_sig_cols {
                vel_x_signals[src_idx * time_steps + step] = u_signal[[0, sig_row, step]] as f32;
            }
        }
    }

    let effective_alpha_db = if alpha_coeff_db > 0.0 {
        alpha_coeff_db
    } else {
        medium.as_medium().alpha_coefficient(0.0, 0.0, 0.0, grid)
    };

    let alpha_power = {
        let y_medium = medium.as_medium().alpha_power(0.0, 0.0, 0.0, grid);
        if alpha_coeff_db <= 0.0 && y_medium > 0.0 && (y_medium - 1.0).abs() > 1e-12 {
            y_medium
        } else {
            alpha_power
        }
    };

    let has_nonlinear = medium.as_medium().nonlinearity(0, 0, 0) > 0.0;
    let has_absorption = effective_alpha_db > 0.0;

    let bon_a_flat: Vec<f32> = if has_nonlinear {
        (0..total)
            .map(|flat| {
                let ix = flat / (ny * nz);
                let iy = (flat % (ny * nz)) / nz;
                let iz = flat % nz;
                (medium.as_medium().nonlinearity(ix, iy, iz) / 2.0) as f32
            })
            .collect()
    } else {
        vec![0.0f32; total]
    };

    let (absorb_nabla1_flat, absorb_nabla2_flat, absorb_tau_flat, absorb_eta_flat) =
        if has_absorption {
            use std::f64::consts::PI;
            let dk_x = 2.0 * PI / (nx as f64 * grid.dx);
            let dk_y = 2.0 * PI / (ny as f64 * grid.dy);
            let dk_z = 2.0 * PI / (nz as f64 * grid.dz);
            let singularity_thresh: f64 = 1e-8;
            let y = alpha_power;

            let mut n1 = vec![0.0f32; total];
            let mut n2 = vec![0.0f32; total];
            let mut tau_v = vec![0.0f32; total];
            let mut eta_v = vec![0.0f32; total];

            for flat in 0..total {
                let ix = flat / (ny * nz);
                let iy = (flat % (ny * nz)) / nz;
                let iz = flat % nz;

                let kix = if ix <= nx / 2 {
                    ix as f64
                } else {
                    (nx - ix) as f64
                } * dk_x;
                let kiy = if iy <= ny / 2 {
                    iy as f64
                } else {
                    (ny - iy) as f64
                } * dk_y;
                let kiz = if iz <= nz / 2 {
                    iz as f64
                } else {
                    (nz - iz) as f64
                } * dk_z;
                let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();

                if k_mag > singularity_thresh {
                    n1[flat] = k_mag.powf(y - 2.0) as f32;
                    n2[flat] = k_mag.powf(y - 1.0) as f32;
                }

                let alpha_db_cm = medium.as_medium().absorption(ix, iy, iz);
                let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
                let c0_local = medium.as_medium().sound_speed(ix, iy, iz);
                tau_v[flat] = (-2.0 * alpha_0_si * c0_local.powf(y - 1.0)) as f32;
                eta_v[flat] = (2.0 * alpha_0_si * c0_local.powf(y) * (PI * y / 2.0).tan()) as f32;
            }
            (n1, n2, tau_v, eta_v)
        } else {
            (
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
            )
        };

    let mut solver = GpuPstdSolver::with_auto_device(
        grid,
        MediumArrays {
            c0_flat: &c0_flat,
            rho0_flat: &rho0_flat,
        },
        SolverParams {
            dt,
            nt: time_steps,
            c_ref,
            nonlinear: has_nonlinear,
            absorbing: has_absorption,
        },
        PmlArrays {
            x: &pml_x_3d,
            y: &pml_y_3d,
            z: &pml_z_3d,
            sgx: &pml_sgx_3d,
            sgy: &pml_sgy_3d,
            sgz: &pml_sgz_3d,
        },
        AbsorptionArrays {
            bon_a_flat: &bon_a_flat,
            nabla1: &absorb_nabla1_flat,
            nabla2: &absorb_nabla2_flat,
            tau: &absorb_tau_flat,
            eta: &absorb_eta_flat,
        },
    )
    .map_err(|e| KwaversError::Io(std::io::Error::other(e)))?;

    let sensor_data_f32 = solver.run(
        &sensor_indices,
        &source_indices,
        &source_signals,
        &vel_x_indices,
        &vel_x_signals,
    );

    let n_sensors = sensor_indices.len();
    let mut out = Array2::<f64>::zeros((n_sensors, time_steps));
    for s in 0..n_sensors {
        for t in 0..time_steps {
            out[[s, t]] = sensor_data_f32[s * time_steps + t] as f64;
        }
    }

    Ok((out, None))
}
