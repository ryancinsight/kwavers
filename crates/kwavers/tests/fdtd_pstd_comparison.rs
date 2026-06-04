//! Multi-solver comparison: FDTD, PSTD, Kuznetsov (linear), Westervelt (linear).
//!
//! Runs all four solvers on identical initial conditions and compares pressure
//! fields against each other and, where possible, against the analytical
//! d'Alembert solution. Generates PNG figures for each test scenario.

use kwavers_boundary::{DomainPMLBoundary, DomainPmlConfig};
use kwavers_core::error::KwaversResult;
use kwavers_domain::source::NullSource;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_solver::forward::nonlinear::westervelt_spectral::WesterveltWave;
use kwavers_solver::pstd::numerics::spectral_correction::SpectralCorrectionMethod;
use kwavers_physics::traits::AcousticWaveModel;
use kwavers_solver::fdtd::FdtdConfig;
use kwavers_solver::fdtd::FdtdPlugin;
use kwavers_grid::Grid;
use kwavers_solver::forward::nonlinear::kuznetsov::KuznetsovConfig;
use kwavers_solver::forward::nonlinear::kuznetsov::KuznetsovWave;
use kwavers_solver::pstd::PSTDConfig;
use kwavers_solver::pstd::PSTDPlugin;
use kwavers_solver::plugin::PluginManager;
use ndarray::{Array3, Array4};
use plotters::prelude::*;
use std::fs;

const FIGURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test-figures");

// ─── Color palette ───────────────────────────────────────────────────────────

const COLOR_FDTD: RGBColor = RGBColor(0, 80, 220);
const COLOR_PSTD: RGBColor = RGBColor(200, 0, 0);
const COLOR_KUZNETSOV: RGBColor = RGBColor(220, 120, 0);
const COLOR_WESTERVELT: RGBColor = RGBColor(130, 0, 200);
const COLOR_ANALYTICAL: RGBColor = RGBColor(0, 140, 0);

// ─── Figure helpers ──────────────────────────────────────────────────────────

/// Save a 4-panel solver comparison figure.
///
/// Panels:
///   Top-left:     initial pressure profile p(x,0) along the central x-line.
///   Top-right:    final-time pressure for all solvers + analytical reference.
///   Bottom-left:  absolute error |solver − reference| per solver.
///   Bottom-right: relative error |solver − reference| / max|reference| × 100 %.
///
/// `curves` must be `[(name, data)]` slices ordered FDTD, PSTD, [Kuznetsov, Westervelt].
/// `reference` is the analytical solution; when `None`, PSTD is used as the reference.
fn save_solver_comparison_figure(
    label: &str,
    x_mm: &[f64],
    initial: &[f64],
    curves: &[(&str, Vec<f64>)],
    reference: Option<&[f64]>,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(FIGURE_DIR)?;
    let filename = format!("{}/fdtd_pstd_{}.png", FIGURE_DIR, label);
    let root = BitMapBackend::new(&filename, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let (top, bot) = root.split_vertically(450);
    let (tl, tr) = top.split_horizontally(600);
    let (bl, br) = bot.split_horizontally(600);

    let nx = x_mm.len();
    let x0 = x_mm[0];
    let x1 = x_mm[nx - 1];

    let palette: &[RGBColor] = &[COLOR_FDTD, COLOR_PSTD, COLOR_KUZNETSOV, COLOR_WESTERVELT];

    // ── Top-left: initial profile ─────────────────────────────────────────────
    {
        let p_max = initial.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let amp = if p_max > 0.0 { p_max } else { 1.0 };
        let mut chart = ChartBuilder::on(&tl)
            .caption(format!("{} — initial p(x,0)", label), ("sans-serif", 15))
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(60)
            .build_cartesian_2d(x0..x1, -amp * 1.15..amp * 1.15)?;
        chart
            .configure_mesh()
            .x_desc("x [mm]")
            .y_desc("p [Pa]")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                x_mm.iter().zip(initial.iter()).map(|(&x, &p)| (x, p)),
                BLACK.stroke_width(2),
            ))?
            .label("p(x, t=0)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    // ── Top-right: all solvers + analytical ──────────────────────────────────
    {
        let all_vals = curves
            .iter()
            .flat_map(|(_, d)| d.iter().copied())
            .chain(reference.into_iter().flat_map(|r| r.iter().copied()));
        let p_max = all_vals.map(|v| v.abs()).fold(0.0_f64, f64::max);
        let amp = if p_max > 0.0 { p_max } else { 1.0 };

        let caption = if reference.is_some() {
            "Final pressure: solvers + analytical".to_string()
        } else {
            "Final pressure: all solvers".to_string()
        };
        let mut chart = ChartBuilder::on(&tr)
            .caption(caption, ("sans-serif", 15))
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(60)
            .build_cartesian_2d(x0..x1, -amp * 1.15..amp * 1.15)?;
        chart
            .configure_mesh()
            .x_desc("x [mm]")
            .y_desc("p [Pa]")
            .draw()?;

        for (i, (name, data)) in curves.iter().enumerate() {
            let color = palette[i.min(palette.len() - 1)];
            let name_owned = (*name).to_string();
            chart
                .draw_series(LineSeries::new(
                    x_mm.iter().zip(data.iter()).map(|(&x, &p)| (x, p)),
                    color.stroke_width(2),
                ))?
                .label(name_owned)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });
        }

        if let Some(ref_data) = reference {
            chart
                .draw_series(LineSeries::new(
                    x_mm.iter().zip(ref_data.iter()).map(|(&x, &p)| (x, p)),
                    COLOR_ANALYTICAL.stroke_width(2),
                ))?
                .label("Analytical")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], COLOR_ANALYTICAL.stroke_width(2))
                });
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    // Determine the reference for error panels.
    // If analytical is provided, use it; otherwise use PSTD (index 1).
    let ref_data: Vec<f64> = reference
        .map(|r| r.to_vec())
        .unwrap_or_else(|| curves.get(1).map(|(_, d)| d.clone()).unwrap_or_default());
    let ref_label = if reference.is_some() {
        "analytical"
    } else {
        "PSTD"
    };
    let ref_amp = ref_data.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
    let ref_amp = if ref_amp > 0.0 { ref_amp } else { 1.0 };

    // ── Bottom-left: |solver − reference| ────────────────────────────────────
    {
        let abs_errors: Vec<Vec<f64>> = curves
            .iter()
            .map(|(_, data)| {
                data.iter()
                    .zip(ref_data.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .collect()
            })
            .collect();
        let err_max = abs_errors
            .iter()
            .flat_map(|e| e.iter().copied())
            .fold(0.0_f64, f64::max);
        let amp = if err_max > 0.0 { err_max } else { 1.0 };

        let mut chart = ChartBuilder::on(&bl)
            .caption(
                format!("|solver − {}| absolute error", ref_label),
                ("sans-serif", 15),
            )
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(60)
            .build_cartesian_2d(x0..x1, 0.0_f64..amp * 1.2)?;
        chart
            .configure_mesh()
            .x_desc("x [mm]")
            .y_desc("|Δp| [Pa]")
            .draw()?;

        for (i, ((name, _), errs)) in curves.iter().zip(abs_errors.iter()).enumerate() {
            let color = palette[i.min(palette.len() - 1)];
            let name_owned = (*name).to_string();
            chart
                .draw_series(LineSeries::new(
                    x_mm.iter().zip(errs.iter()).map(|(&x, &e)| (x, e)),
                    color.stroke_width(2),
                ))?
                .label(name_owned)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    // ── Bottom-right: relative error % ───────────────────────────────────────
    {
        let rel_errors: Vec<Vec<f64>> = curves
            .iter()
            .map(|(_, data)| {
                data.iter()
                    .zip(ref_data.iter())
                    .map(|(&a, &b)| (a - b).abs() / ref_amp * 100.0)
                    .collect()
            })
            .collect();
        let err_max = rel_errors
            .iter()
            .flat_map(|e| e.iter().copied())
            .fold(0.0_f64, f64::max);
        let amp = if err_max > 0.0 { err_max } else { 1.0 };

        let mut chart = ChartBuilder::on(&br)
            .caption(
                format!("|solver−{}| / max|ref| (%)", ref_label),
                ("sans-serif", 15),
            )
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(60)
            .build_cartesian_2d(x0..x1, 0.0_f64..amp * 1.2)?;
        chart
            .configure_mesh()
            .x_desc("x [mm]")
            .y_desc("Error (%)")
            .draw()?;

        for (i, ((name, _), rel)) in curves.iter().zip(rel_errors.iter()).enumerate() {
            let color = palette[i.min(palette.len() - 1)];
            let name_owned = (*name).to_string();
            chart
                .draw_series(LineSeries::new(
                    x_mm.iter().zip(rel.iter()).map(|(&x, &e)| (x, e)),
                    color.stroke_width(2),
                ))?
                .label(name_owned)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()?;
    println!("Figure saved: {}", filename);
    Ok(())
}

/// Save a figure showing numerical dispersion: κ(k) for FDTD (4th-order) vs
/// PSTD (sinc-corrected spectral).
///
/// κ(k) = v_phase / c — equals 1.0 for a dispersion-free scheme.
fn save_dispersion_comparison_figure(
    dx_m: f64,
    dt_s: f64,
    c: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(FIGURE_DIR)?;
    let path = format!("{}/fdtd_pstd_dispersion.png", FIGURE_DIR);
    let root = BitMapBackend::new(&path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let k_nyquist = std::f64::consts::PI / dx_m;
    let n_pts = 200usize;
    let ks: Vec<f64> = (0..=n_pts)
        .map(|i| i as f64 / n_pts as f64 * k_nyquist)
        .collect();

    // FDTD 4th-order: effective wavenumber from the stencil k_eff = (8sin(kΔx/2) − sin(kΔx))/(3Δx)
    let fdtd_kappa: Vec<(f64, f64)> = ks
        .iter()
        .map(|&k| {
            let kdx = k * dx_m;
            let kappa = if kdx < 1e-10 {
                1.0
            } else {
                let k_eff = (8.0 * (kdx / 2.0).sin() - kdx.sin()) / (3.0 * dx_m);
                (k_eff / k).clamp(0.0, 2.0)
            };
            (k / k_nyquist, kappa)
        })
        .collect();

    // PSTD / Westervelt spectral: sinc k-space operator κ(k) = sinc(cΔtk/2)
    let pstd_kappa: Vec<(f64, f64)> = ks
        .iter()
        .map(|&k| {
            let arg = c * dt_s * k / 2.0;
            let kappa = if arg.abs() < 1e-10 {
                1.0
            } else {
                arg.sin() / arg
            };
            (k / k_nyquist, kappa)
        })
        .collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Numerical Dispersion: FDTD(4th) vs PSTD/Spectral  \
                 [dx={:.2} mm, dt={:.2} µs, c={:.0} m/s]",
                dx_m * 1e3,
                dt_s * 1e6,
                c
            ),
            ("sans-serif", 15),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0f64..1.0f64, 0.7f64..1.05f64)?;

    chart
        .configure_mesh()
        .x_desc("k / k_Nyquist")
        .y_desc("κ(k) = v_ph / c  [dispersion ratio]")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            [(0.0, 1.0), (1.0, 1.0)],
            BLACK.stroke_width(1),
        ))?
        .label("κ = 1 (dispersion-free)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    chart
        .draw_series(LineSeries::new(fdtd_kappa, COLOR_FDTD.stroke_width(2)))?
        .label("FDTD 4th-order")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], COLOR_FDTD));

    chart
        .draw_series(LineSeries::new(pstd_kappa, COLOR_PSTD.stroke_width(2)))?
        .label("PSTD / spectral (sinc k-space)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], COLOR_PSTD));

    chart
        .draw_series(std::iter::once(Circle::new((0.8, 1.0), 6, GREEN.filled())))?
        .label("80% Nyquist")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, GREEN.filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Figure saved: {}", path);
    Ok(())
}

// ─── Simulation helpers ───────────────────────────────────────────────────────

/// Compute the stable dt for the FDTD 4th-order 3D solver.
///
/// The solver enforces: dt ≤ config.cfl_factor × (1/√15) × min_dx/c.
/// Using the same formula here guarantees the stability check passes.
fn fdtd_dt(grid: &Grid, c: f64) -> f64 {
    let cfl_factor = 0.95;
    let cfl_limit_4th = 1.0_f64 / 15.0_f64.sqrt(); // ≈ 0.258 (4th-order, 3D)
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    cfl_factor * cfl_limit_4th * min_dx / c
}

fn run_fdtd_simulation(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
) -> KwaversResult<Array3<f64>> {
    run_fdtd_simulation_with_time(grid, medium, initial_pressure, 50e-6)
}

fn run_fdtd_simulation_with_time(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
    t_end: f64,
) -> KwaversResult<Array3<f64>> {
    let c = 1500.0;
    let dt = fdtd_dt(grid, c);
    let n_steps = (t_end / dt).ceil() as usize;

    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: 0.95,
        subgridding: false,
        subgrid_factor: 2,
        enable_gpu_acceleration: false,
        nt: n_steps,
        dt,
        sensor_mask: None,
        ..Default::default()
    };

    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(FdtdPlugin::new(config, grid)?))?;

    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields
        .slice_mut(ndarray::s![0, .., .., ..])
        .assign(initial_pressure);

    plugin_manager.initialize(grid, medium)?;

    let sources = Vec::new();
    let mut boundary = DomainPMLBoundary::new(DomainPmlConfig::default().with_thickness(8))?;

    for step in 0..n_steps {
        let t = step as f64 * dt;
        plugin_manager.execute(&mut fields, grid, medium, &sources, &mut boundary, dt, t)?;
    }

    Ok(fields.slice(ndarray::s![0, .., .., ..]).to_owned())
}

fn run_pstd_simulation(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
) -> KwaversResult<Array3<f64>> {
    run_pstd_simulation_with_time(grid, medium, initial_pressure, 50e-6)
}

fn run_pstd_simulation_with_time(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
    t_end: f64,
) -> KwaversResult<Array3<f64>> {
    let mut config = PSTDConfig::default();
    config.spectral_correction.enabled = true;
    config.spectral_correction.method = SpectralCorrectionMethod::SincSpatial;
    config.anti_aliasing.enabled = true;
    config.absorption_mode = AbsorptionMode::Lossless;

    // 3D PSTD stability: dt ≤ Δx/(c·π·√3) ≈ 0.184·Δx/c; 0.15 provides safe margin.
    let cfl_factor = 0.15;
    let c = 1500.0;
    let dt = cfl_factor * grid.dx.min(grid.dy).min(grid.dz) / c;
    let n_steps = (t_end / dt).ceil() as usize;

    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(PSTDPlugin::new(config, grid)?))?;

    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields
        .slice_mut(ndarray::s![0, .., .., ..])
        .assign(initial_pressure);

    plugin_manager.initialize(grid, medium)?;

    let sources = Vec::new();
    let mut boundary = DomainPMLBoundary::new(DomainPmlConfig::default().with_thickness(8))?;

    for step in 0..n_steps {
        let t = step as f64 * dt;
        plugin_manager.execute(&mut fields, grid, medium, &sources, &mut boundary, dt, t)?;
    }

    Ok(fields.slice(ndarray::s![0, .., .., ..]).to_owned())
}

/// Run the Kuznetsov solver in linear mode (AcousticEquationMode::Linear).
///
/// Uses `KuznetsovConfig::linear()` which sets nonlinearity_coefficient = 0 and
/// acoustic_diffusivity = 0, reducing the equation to the linear wave equation
/// solved spectrally.  Shares the FDTD time-step so all solvers span the same
/// physical time (spectral CFL limit ≈ 0.368·Δx/c >> FDTD dt ≈ 0.245·Δx/c).
fn run_kuznetsov_simulation_with_time(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
    t_end: f64,
) -> KwaversResult<Array3<f64>> {
    let c = 1500.0;
    let dt = fdtd_dt(grid, c); // conservative; Kuznetsov spectral CFL is more lenient
    let n_steps = (t_end / dt).ceil() as usize;

    let config = KuznetsovConfig::linear();
    let mut solver = KuznetsovWave::new(config, grid)?;

    // Array4 with 1 slot: pressure lives at axis-0 index 0.
    let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
    fields
        .slice_mut(ndarray::s![0, .., .., ..])
        .assign(initial_pressure);

    let source = NullSource::new();
    let prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

    for step in 0..n_steps {
        let t = step as f64 * dt;
        solver.update_wave(&mut fields, &prev, &source, grid, medium, dt, t)?;
    }

    Ok(fields.slice(ndarray::s![0, .., .., ..]).to_owned())
}

fn run_kuznetsov_simulation(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
) -> KwaversResult<Array3<f64>> {
    run_kuznetsov_simulation_with_time(grid, medium, initial_pressure, 50e-6)
}

/// Run the Westervelt solver in the linear limit (nonlinearity_scaling = 0.0).
///
/// `WesterveltWave` uses a spectral Laplacian leapfrog identical to PSTD in the
/// linear, lossless regime.  Disabling the β term (via set_nonlinearity_scaling)
/// isolates the pure linear spectral solver for comparison.
fn run_westervelt_simulation_with_time(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
    t_end: f64,
) -> KwaversResult<Array3<f64>> {
    let c = 1500.0;
    let dt = fdtd_dt(grid, c); // CFL limit for spectral leapfrog ≈ 0.368·Δx/c >> our dt
    let n_steps = (t_end / dt).ceil() as usize;

    let mut solver = WesterveltWave::new(grid);
    solver.set_nonlinearity_scaling(0.0); // suppress β·p² term → linear limit

    // Array4 with 1 pressure slot (index 0 = UnifiedFieldType::Pressure).
    let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
    fields
        .slice_mut(ndarray::s![0, .., .., ..])
        .assign(initial_pressure);

    let source = NullSource::new();
    let prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

    for step in 0..n_steps {
        let t = step as f64 * dt;
        solver.update_wave(&mut fields, &prev, &source, grid, medium, dt, t)?;
    }

    Ok(fields.slice(ndarray::s![0, .., .., ..]).to_owned())
}

fn run_westervelt_simulation(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
) -> KwaversResult<Array3<f64>> {
    run_westervelt_simulation_with_time(grid, medium, initial_pressure, 50e-6)
}

/// Analytical d'Alembert solution for the standing-wave initial condition.
///
/// IC: p(x, 0) = sin(i·0.2) × 1e5 Pa with dx = 1e-3 m
///     ↔ p(x, 0) = sin(kx) × 1e5   with k = 0.2/dx = 200 rad/m
/// Solution (d'Alembert, periodic BC, k·L = π/5·32 ≈ 6.4 rad, lossless):
///   p(x, t) = sin(kx) · cos(ωt) · 1e5   where ω = k·c = 300 000 rad/s
fn analytical_standing_wave(grid: &Grid, t: f64) -> Vec<f64> {
    let k = 0.2 / grid.dx; // wavenumber [rad/m]
    let omega = k * 1500.0; // angular frequency [rad/s]
    (0..grid.nx)
        .map(|i| {
            let x = i as f64 * grid.dx;
            (k * x).sin() * 1e5 * (omega * t).cos()
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Point-source propagation: verify all four solvers run without NaN and
/// generate a comparison figure of central-line pressure profiles.
#[test]
fn test_plane_wave_propagation() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 0.5e-3, 0.5e-3, 0.5e-3)?;
    let medium = HomogeneousMedium::water(&grid);

    let mut initial_pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    initial_pressure[[grid.nx / 2, grid.ny / 2, grid.nz / 2]] = 1e6;

    let fdtd_result = run_fdtd_simulation(&grid, &medium, &initial_pressure)?;
    let pstd_result = run_pstd_simulation(&grid, &medium, &initial_pressure)?;
    let kuz_result = run_kuznetsov_simulation(&grid, &medium, &initial_pressure)?;
    let wes_result = run_westervelt_simulation(&grid, &medium, &initial_pressure)?;

    // All solvers must complete without NaN.
    for (name, result) in &[
        ("FDTD", &fdtd_result),
        ("PSTD", &pstd_result),
        ("Kuznetsov-L", &kuz_result),
        ("Westervelt-L", &wes_result),
    ] {
        let max_val = result
            .mapv(f64::abs)
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        assert!(
            max_val.is_finite(),
            "{name} produced non-finite values: {max_val}"
        );
    }

    let hy = grid.ny / 2;
    let hz = grid.nz / 2;
    let x_mm: Vec<f64> = (0..grid.nx).map(|i| i as f64 * grid.dx * 1e3).collect();
    let initial_line: Vec<f64> = (0..grid.nx)
        .map(|i| initial_pressure[[i, hy, hz]])
        .collect();
    let fdtd_line: Vec<f64> = (0..grid.nx).map(|i| fdtd_result[[i, hy, hz]]).collect();
    let pstd_line: Vec<f64> = (0..grid.nx).map(|i| pstd_result[[i, hy, hz]]).collect();
    let kuz_line: Vec<f64> = (0..grid.nx).map(|i| kuz_result[[i, hy, hz]]).collect();
    let wes_line: Vec<f64> = (0..grid.nx).map(|i| wes_result[[i, hy, hz]]).collect();

    let curves: &[(&str, Vec<f64>)] = &[
        ("FDTD", fdtd_line),
        ("PSTD", pstd_line),
        ("Kuznetsov-L", kuz_line),
        ("Westervelt-L", wes_line),
    ];

    if let Err(e) =
        save_solver_comparison_figure("point_source", &x_mm, &initial_line, curves, None)
    {
        eprintln!("Warning: figure save failed: {e}");
    }

    Ok(())
}

/// Sinusoidal standing-wave IC: compare all solvers against the analytical
/// d'Alembert solution p(x,t) = sin(kx) · cos(ωt) · 1e5.
#[test]
fn test_standing_wave_analytical() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3)?;
    let medium = HomogeneousMedium::water(&grid);

    let mut initial_pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let hy = grid.ny / 2;
    let hz = grid.nz / 2;
    for i in 0..grid.nx {
        initial_pressure[[i, hy, hz]] = (i as f64 * 0.2).sin() * 1e5;
    }

    let t_end = 50e-6;
    let fdtd_result = run_fdtd_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;
    let pstd_result = run_pstd_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;
    let kuz_result = run_kuznetsov_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;
    let wes_result = run_westervelt_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;

    // All solvers must produce finite values.
    for (name, result) in &[
        ("FDTD", &fdtd_result),
        ("PSTD", &pstd_result),
        ("Kuznetsov-L", &kuz_result),
        ("Westervelt-L", &wes_result),
    ] {
        let max_val = result
            .mapv(f64::abs)
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        assert!(
            max_val.is_finite(),
            "{name} produced non-finite values: {max_val}"
        );
    }

    // Analytical d'Alembert at t_end on the midplane.
    // The Kuznetsov FDTD dt may differ from the exact t_end; use a representative t.
    let dt_approx = fdtd_dt(&grid, 1500.0);
    let n_steps_approx = (t_end / dt_approx).ceil() as usize;
    let t_actual = n_steps_approx as f64 * dt_approx;
    let analytical_line = analytical_standing_wave(&grid, t_actual);

    let x_mm: Vec<f64> = (0..grid.nx).map(|i| i as f64 * grid.dx * 1e3).collect();
    let initial_line: Vec<f64> = (0..grid.nx)
        .map(|i| initial_pressure[[i, hy, hz]])
        .collect();
    let fdtd_line: Vec<f64> = (0..grid.nx).map(|i| fdtd_result[[i, hy, hz]]).collect();
    let pstd_line: Vec<f64> = (0..grid.nx).map(|i| pstd_result[[i, hy, hz]]).collect();
    let kuz_line: Vec<f64> = (0..grid.nx).map(|i| kuz_result[[i, hy, hz]]).collect();
    let wes_line: Vec<f64> = (0..grid.nx).map(|i| wes_result[[i, hy, hz]]).collect();

    let curves: &[(&str, Vec<f64>)] = &[
        ("FDTD", fdtd_line),
        ("PSTD", pstd_line),
        ("Kuznetsov-L", kuz_line),
        ("Westervelt-L", wes_line),
    ];

    if let Err(e) = save_solver_comparison_figure(
        "standing_wave",
        &x_mm,
        &initial_line,
        curves,
        Some(&analytical_line),
    ) {
        eprintln!("Warning: figure save failed: {e}");
    }

    Ok(())
}

/// Numerical dispersion: show FDTD 4th-order vs PSTD/spectral κ(k) curves.
/// Generates a dedicated dispersion comparison figure.
#[test]
fn test_dispersion_characteristics() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3)?;
    let medium = HomogeneousMedium::water(&grid);

    // Use a uniform-amplitude field to exercise all code paths.
    let initial_pressure = Array3::ones((grid.nx, grid.ny, grid.nz)) * 1e5;

    let _ = run_fdtd_simulation(&grid, &medium, &initial_pressure)?;
    let _ = run_pstd_simulation(&grid, &medium, &initial_pressure)?;

    let c = 1500.0_f64;
    let dx = grid.dx;
    let dt = fdtd_dt(&grid, c);

    if let Err(e) = save_dispersion_comparison_figure(dx, dt, c) {
        eprintln!("Warning: dispersion figure save failed: {e}");
    }

    Ok(())
}
