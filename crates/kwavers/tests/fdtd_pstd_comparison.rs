//! Multi-solver comparison: FDTD, PSTD, Kuznetsov (linear), Westervelt (linear).
//!
//! Runs all four solvers on identical initial conditions and compares pressure
//! fields against each other and, where possible, against the analytical
//! d'Alembert solution. Generates PNG figures for each test scenario.

use kwavers_boundary::{DomainPMLBoundary, DomainPmlConfig};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_physics::traits::AcousticWaveModel;
use kwavers_solver::fdtd::FdtdConfig;
use kwavers_solver::fdtd::FdtdPlugin;
use kwavers_solver::forward::nonlinear::kuznetsov::KuznetsovConfig;
use kwavers_solver::forward::nonlinear::kuznetsov::KuznetsovWave;
use kwavers_solver::forward::nonlinear::westervelt_spectral::WesterveltWave;
use kwavers_solver::plugin::PluginManager;
use kwavers_solver::pstd::numerics::spectral_correction::SpectralCorrectionMethod;
use kwavers_solver::pstd::PSTDConfig;
use kwavers_solver::pstd::PSTDPlugin;
use kwavers_source::NullSource;
use leto::{
    Array3,
    Array4,
};
use plotters::prelude::*;
use std::fs;

const FIGURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test-figures");

/// Transparent (no-op) boundary: leaves the field untouched so the spectral
/// solvers run on a lossless **periodic** domain. The PML-specific trait methods
/// default to `apply_acoustic`, so a no-op `apply_acoustic` makes every variant
/// transparent. Used to drive PSTD/FDTD as energy-conserving resonators, against
/// which the analytical d'Alembert standing wave is the correct oracle.
#[derive(Debug)]
struct NullBoundary;

impl kwavers_boundary::Boundary for NullBoundary {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn apply_acoustic(
        &mut self,
        _field: leto::ArrayViewMut3<f64>,
        _grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        Ok(())
    }
    fn apply_acoustic_freq(
        &mut self,
        _field: &mut Array3<kwavers_math::fft::Complex64>,
        _grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        Ok(())
    }
    fn apply_light(
        &mut self,
        _field: leto::ArrayViewMut3<f64>,
        _grid: &Grid,
        _time_step: usize,
    ) {
    }
}

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
        .index_axis_mut::<3>(0, 0).expect("index_axis").assign(initial_pressure);

    plugin_manager.initialize(grid, medium)?;

    let sources = Vec::new();
    let mut boundary = DomainPMLBoundary::new(DomainPmlConfig::default().with_thickness(8))?;

    for step in 0..n_steps {
        let t = step as f64 * dt;
        plugin_manager.execute(&mut fields, grid, medium, &sources, &mut boundary, dt, t)?;
    }

    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
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
    // Treeby & Cox (2010) κ = sinc(c_ref·dt·|k|/2): the k-space leapfrog's TEMPORAL
    // dispersion correction (PSTDConfig default). At CFL 0.15 κ≈1, so the method
    // choice is not load-bearing here, but Treeby2010 is the physically correct one.
    config.spectral_correction.method = SpectralCorrectionMethod::Treeby2010;
    // Anti-aliasing: the default cutoff=0.95 order-8 Butterworth. For a σ=3Δx
    // Gaussian (negligible content above 0.1·Nyquist) this is effectively a no-op;
    // enabled for parity with nonlinear runs.
    config.anti_aliasing.enabled = true;
    config.absorption_mode = AbsorptionMode::Lossless;
    // Boundary: the comparison grid is 32³ but PSTDConfig::default() uses CPML with
    // thickness 20 — a 40-cell profile that smothers the entire 32-cell domain
    // including the [9,22] interior sample region. Edge absorption is already
    // supplied by the external DomainPMLBoundary(thickness 8) applied in the step
    // loop, which leaves [9,22] PML-free, so disable the internal PML to measure the
    // pure k-space leapfrog against the free-space analytical IVP.
    config.boundary = kwavers_solver::pstd::config::BoundaryConfig::None;

    // 3D PSTD stability: dt ≤ Δx/(c·π·√3) ≈ 0.184·Δx/c; 0.15 provides safe margin.
    let cfl_factor = 0.15;
    let c = 1500.0;
    let dt = cfl_factor * grid.dx.min(grid.dy).min(grid.dz) / c;
    // CRITICAL: the solver advances by `config.dt`, NOT by the `dt` used to size
    // n_steps. PSTDConfig::default().dt = 1e-7 (100 ns) — exactly 2× this CFL dt
    // (5e-8) for the 0.5 mm / 1500 m·s⁻¹ grid. Leaving config.dt at the default made
    // the wave propagate twice as far as the analytical reference (ct doubled),
    // which is the ENTIRE source of the prior PSTD "119 %" interior error — not
    // dissipation, dispersion, the IVP seed, the PML, or the κ method (all verified
    // independently in pstd_gaussian_propagation_matches_analytical). With config.dt
    // synced, PSTD interior rel-L2 vs the exact Gaussian IVP is ≈ 0.02.
    config.dt = dt;
    let n_steps = (t_end / dt).ceil() as usize;

    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(PSTDPlugin::new(config, grid)?))?;

    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields
        .index_axis_mut::<3>(0, 0).expect("index_axis").assign(initial_pressure);

    plugin_manager.initialize(grid, medium)?;

    let sources = Vec::new();
    let mut boundary = DomainPMLBoundary::new(DomainPmlConfig::default().with_thickness(8))?;

    for step in 0..n_steps {
        let t = step as f64 * dt;
        plugin_manager.execute(&mut fields, grid, medium, &sources, &mut boundary, dt, t)?;
    }

    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
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
        .index_axis_mut::<3>(0, 0).expect("index_axis").assign(initial_pressure);

    let source = NullSource::new();
    let prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

    for step in 0..n_steps {
        let t = step as f64 * dt;
        solver.update_wave(&mut fields, &prev, &source, grid, medium, dt, t)?;
    }

    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
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
    solver.set_damping_scaling(0.0); // lossless: match the other linear solvers
                                     // (and avoid the conditionally-unstable
                                     // explicit ∇²(∂p/∂t) damping term)

    // Array4 with 1 pressure slot (index 0 = UnifiedFieldType::Pressure).
    let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
    fields
        .index_axis_mut::<3>(0, 0).expect("index_axis").assign(initial_pressure);

    let source = NullSource::new();
    let prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

    for step in 0..n_steps {
        let t = step as f64 * dt;
        solver.update_wave(&mut fields, &prev, &source, grid, medium, dt, t)?;
    }

    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
}

/// Exact solution of the 3-D wave equation for a spherically-symmetric Gaussian
/// initial pressure with zero initial velocity.
///
/// # Theorem (3-D radial d'Alembert)
/// With `u(r,t) = r·p(r,t)`, the radial 3-D wave equation reduces to the 1-D wave
/// equation for `u`. For `p(r,0) = A·exp(−r²/2σ²)`, `∂ₜp(r,0)=0`, the odd
/// extension `g(s)=s·A·exp(−s²/2σ²)` gives
/// ```text
/// p(r,t) = (A / 2r) · [ (r+ct)·E(r+ct) + (r−ct)·E(r−ct) ],   E(s)=exp(−s²/2σ²)
/// ```
/// with the removable singularity at `r→0`:
/// `p(0,t) = A·E(ct)·(1 − (ct)²/σ²)`.
fn analytical_gaussian_ivp(grid: &Grid, sigma: f64, amplitude: f64, c: f64, t: f64) -> Array3<f64> {
    let (cx, cy, cz) = (
        grid.nx as f64 / 2.0,
        grid.ny as f64 / 2.0,
        grid.nz as f64 / 2.0,
    );
    let ct = c * t;
    let two_sigma2 = 2.0 * sigma * sigma;
    let e = |s: f64| (-s * s / two_sigma2).exp();
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r = (((i as f64 - cx) * grid.dx).powi(2)
                    + ((j as f64 - cy) * grid.dy).powi(2)
                    + ((k as f64 - cz) * grid.dz).powi(2))
                .sqrt();
                field[[i, j, k]] = if r < 1e-9 {
                    amplitude * e(ct) * (1.0 - ct * ct / (sigma * sigma))
                } else {
                    amplitude / (2.0 * r) * ((r + ct) * e(r + ct) + (r - ct) * e(r - ct))
                };
            }
        }
    }
    field
}

/// Analytical d'Alembert solution for a sinusoidal standing-wave IVP
/// `p(x,0) = A·sin(kx)`, `u(x,0) = 0` in a **lossless** domain:
///   `p(x, t) = A · sin(kx) · cos(ωt)`,  `ω = c·k`,  c = 1500 m/s.
/// Valid only when `k` is a periodic eigenmode of the box (`k·L = 2π·m`); the
/// central-line samples are returned along x.
fn analytical_standing_wave(grid: &Grid, k: f64, amplitude: f64, t: f64) -> Vec<f64> {
    let omega = k * 1500.0;
    (0..grid.nx)
        .map(|i| {
            let x = i as f64 * grid.dx;
            (k * x).sin() * amplitude * (omega * t).cos()
        })
        .collect()
}

/// Relative L2 error `‖a − b‖₂ / ‖b‖₂` between two central-line profiles.
fn relative_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    num / den.max(1e-30)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Point-source propagation: verify all four solvers run without NaN and
/// generate a comparison figure of central-line pressure profiles.
#[test]
fn test_plane_wave_propagation() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 0.5e-3, 0.5e-3, 0.5e-3)?;
    let medium = HomogeneousMedium::water(&grid);

    // Band-limited Gaussian initial pressure. A single-cell delta excites every
    // mode to the Nyquist limit, which is unrepresentable on the grid and
    // destabilizes explicit schemes — it is not a valid solver-comparison
    // stimulus. A Gaussian of σ = 3Δx has its spectrum well below Nyquist, so all
    // solvers propagate it accurately and should agree.
    let (cx, cy, cz) = (
        grid.nx as f64 / 2.0,
        grid.ny as f64 / 2.0,
        grid.nz as f64 / 2.0,
    );
    let sigma = 3.0 * grid.dx;
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut initial_pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r2 = ((i as f64 - cx) * grid.dx).powi(2)
                    + ((j as f64 - cy) * grid.dy).powi(2)
                    + ((k as f64 - cz) * grid.dz).powi(2);
                initial_pressure[[i, j, k]] = 1e6 * (-r2 / two_sigma2).exp();
            }
        }
    }

    // Short propagation: the spherical wavefront stays within the domain interior
    // (away from the PML) so the comparison is a clean in-domain agreement test.
    let t_end = 2.0e-6;
    let fdtd_result = run_fdtd_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;
    let pstd_result = run_pstd_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;
    let kuz_result = run_kuznetsov_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;
    let wes_result = run_westervelt_simulation_with_time(&grid, &medium, &initial_pressure, t_end)?;

    // All solvers must produce only finite values. NOTE: `fold(0.0, f64::max)`
    // silently ignores NaN (f64::max(0.0, NaN) == 0.0), so it cannot detect a
    // blown-up solver — check every element explicitly instead.
    for (name, result) in &[
        ("FDTD", &fdtd_result),
        ("PSTD", &pstd_result),
        ("Kuznetsov-L", &kuz_result),
        ("Westervelt-L", &wes_result),
    ] {
        assert!(
            result.iter().all(|v| v.is_finite()),
            "{name} produced non-finite (NaN/Inf) values"
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

    // ── Diagnostics: whole-field statistics and energy vs. PSTD ──────────────
    let field_stats = |r: &Array3<f64>| -> (f64, f64) {
        let max = r.mapv(f64::abs).iter().copied().fold(0.0_f64, f64::max);
        let l2 = (r.iter().map(|&v| v * v).sum::<f64>()).sqrt();
        (max, l2)
    };
    let init_max = initial_pressure
        .mapv(f64::abs)
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    eprintln!("── point_source diagnostics (init max = {init_max:.3e} Pa) ──");
    for (name, r) in &[
        ("FDTD", &fdtd_result),
        ("PSTD", &pstd_result),
        ("Kuznetsov-L", &kuz_result),
        ("Westervelt-L", &wes_result),
    ] {
        let (mx, l2) = field_stats(r);
        eprintln!(
            "  {name:12} final max={mx:.4e} Pa  L2={l2:.4e}  (max/init = {:.3e})",
            mx / init_max
        );
    }

    // ── Validation against the exact 3-D Gaussian-IVP solution ───────────────
    // Compare over the PML-free interior (PML thickness = 8 cells) where the
    // infinite-domain analytical solution is valid (the wavefront, radius ct ≈
    // 6 cells at t_end, has not reached the absorbing layer).
    let analytical = analytical_gaussian_ivp(&grid, sigma, 1e6, 1500.0, t_end);
    let lo = 9usize;
    let hi = grid.nx - 9; // interior [9, 22] for nx=32
    let rel_l2_interior = |solver: &Array3<f64>| -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in lo..hi {
            for j in lo..hi {
                for k in lo..hi {
                    let a = analytical[[i, j, k]];
                    let d = solver[[i, j, k]] - a;
                    num += d * d;
                    den += a * a;
                }
            }
        }
        (num / den.max(1e-30)).sqrt()
    };
    let e_fdtd = rel_l2_interior(&fdtd_result);
    let e_pstd = rel_l2_interior(&pstd_result);
    let e_kuz = rel_l2_interior(&kuz_result);
    let e_wes = rel_l2_interior(&wes_result);
    eprintln!(
        "  interior rel-L2 vs analytical: FDTD={e_fdtd:.3}  PSTD={e_pstd:.3}  \
         Kuznetsov={e_kuz:.3}  Westervelt={e_wes:.3}"
    );

    // Derived tolerance: a σ=3Δx Gaussian has negligible spectral content above
    // k·Δx ≈ 1 (E(kσ) at Nyquist ≈ e^{−44}). Over ~24 steps the leading errors are
    // the 4th-order FDTD dispersion and the pressure-leapfrog zero-velocity IVP
    // split; both are bounded well under 20 % for this band-limited pulse. 20 %
    // is a meaningful agreement check (vs. the >100 % seen before).
    const ANALYTIC_TOL: f64 = 0.20;
    for (name, err) in [
        ("FDTD", e_fdtd),
        ("PSTD", e_pstd),
        ("Kuznetsov", e_kuz),
        ("Westervelt", e_wes),
    ] {
        assert!(
            err < ANALYTIC_TOL,
            "{name} interior rel-L2 vs analytical = {err:.3} must be < {ANALYTIC_TOL}"
        );
    }

    // Figure reference is the exact analytical solution.
    let analytical_line: Vec<f64> = (0..grid.nx).map(|i| analytical[[i, hy, hz]]).collect();
    let curves: &[(&str, Vec<f64>)] = &[
        ("FDTD", fdtd_line),
        ("PSTD", pstd_line),
        ("Kuznetsov-L", kuz_line),
        ("Westervelt-L", wes_line),
    ];

    if let Err(e) = save_solver_comparison_figure(
        "point_source",
        &x_mm,
        &initial_line,
        curves,
        Some(&analytical_line),
    ) {
        eprintln!("Warning: figure save failed: {e}");
    }

    Ok(())
}

/// Sinusoidal standing-wave validation against the analytical d'Alembert
/// solution `p(x,t) = A·sin(kx)·cos(ωt)` on a LOSSLESS PERIODIC resonator.
///
/// The original figure compared PML-bounded / non-eigenmode runs against a
/// lossless single-mode oracle (an apples-to-oranges setup that is why the
/// solvers "collapsed"). This test fixes all three mismatches — see
/// `diagnose_standing_wave_collapse_source` for the supporting evidence:
///   * EIGENMODE IC `k·L = 2π·2` — a true periodic eigenmode, so the single-mode
///     analytical is exact (a non-eigenmode k dephases across DFT bins and the
///     closed form no longer describes the field).
///   * Transparent boundary (no PML) — the analytical assumes energy
///     conservation; an absorbing layer drains the standing wave.
///   * Spectral solvers only (Westervelt, Kuznetsov, PSTD), which are lossless
///     and periodic. The 4th-order staggered FDTD is not periodic-stable without
///     its absorbing layer, so it cannot model a lossless resonator here; its
///     zero-velocity IVP is validated against the free-space Gaussian in
///     `pstd_gaussian_propagation_matches_analytical` and the plane-wave figure.
#[test]
fn test_standing_wave_analytical() -> KwaversResult<()> {
    // 16³ keeps each plugin run within the 30 s test budget; the m = 2 eigenmode
    // is exactly representable by the spectral basis, so the physics is unchanged.
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3)?;
    let medium = HomogeneousMedium::water(&grid);
    let c = 1500.0_f64;
    let amplitude = 1e5_f64;
    let l = grid.nx as f64 * grid.dx;

    // Eigenmode k·L = 2π·2 (k·dx ≈ 0.79 ⇒ 8 cells/λ, exactly on a DFT bin).
    let k = 2.0 * std::f64::consts::PI * 2.0 / l;
    let omega = c * k;
    let ic = sinusoidal_ic(&grid, k, amplitude);

    // Shared timeline. dt = the (conservative) FDTD CFL dt; the spectral CFL is
    // more lenient, so this is stable for all three. Stop at ≈ 2.5 periods, where
    // the analytical sits near its −A0 extremum, so the comparison is robust to
    // the sub-1% leapfrog temporal dispersion (phase error has minimal amplitude
    // impact at a cosine extremum).
    let dt = fdtd_dt(&grid, c);
    let period = 2.0 * std::f64::consts::PI / omega;
    let n_steps = (2.5 * period / dt).round() as usize;
    let t_end = n_steps as f64 * dt;

    let wes = run_westervelt_lossless(&grid, &medium, &ic, dt, n_steps, None)?;
    let kuz = run_kuznetsov_lossless(&grid, &medium, &ic, dt, n_steps, None)?;
    let pstd = run_pstd_lossless(&grid, &medium, &ic, dt, n_steps, None)?;

    let hy = grid.ny / 2;
    let hz = grid.nz / 2;
    let line = |f: &Array3<f64>| -> Vec<f64> { (0..grid.nx).map(|i| f[[i, hy, hz]]).collect() };
    let analytical_line = analytical_standing_wave(&grid, k, amplitude, t_end);
    let wes_line = line(&wes);
    let kuz_line = line(&kuz);
    let pstd_line = line(&pstd);

    // Value-semantic check: each lossless spectral solver reproduces the
    // d'Alembert standing wave. Tolerance 8% bounds the leapfrog temporal
    // dispersion (ω_num error ≈ (cΔt·k)²/6 ≈ 4·10⁻⁴ ⇒ < 1% over 2.5 periods),
    // central-line sampling, and the PSTD κ-correction residual.
    const TOL: f64 = 0.08;
    for (name, prof) in [
        ("Westervelt", &wes_line),
        ("Kuznetsov", &kuz_line),
        ("PSTD", &pstd_line),
    ] {
        let err = relative_l2(prof, &analytical_line);
        assert!(
            err < TOL,
            "{name} standing wave deviates from the d'Alembert analytical: relative-L2 = {err:.4} \
             ≥ {TOL} (lossless eigenmode resonator, t = {t_end:.3e} s ≈ 2.5 periods)",
        );
    }

    let x_mm: Vec<f64> = (0..grid.nx).map(|i| i as f64 * grid.dx * 1e3).collect();
    let initial_line = line(&ic);
    let curves: &[(&str, Vec<f64>)] = &[
        ("Westervelt-L", wes_line),
        ("Kuznetsov-L", kuz_line),
        ("PSTD", pstd_line),
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

/// Build a 1-D sinusoidal standing-wave IC `p(x)=A·sin(k·x)`, uniform in y,z,
/// on the central line and across the whole volume (so every solver sees the
/// same field). `k` in rad/m.
fn sinusoidal_ic(grid: &Grid, k: f64, amplitude: f64) -> Array3<f64> {
    let mut p = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        let x = i as f64 * grid.dx;
        let v = amplitude * (k * x).sin();
        for j in 0..grid.ny {
            for kz in 0..grid.nz {
                p[[i, j, kz]] = v;
            }
        }
    }
    p
}

/// Peak `max|p|` over the grid (the standing-wave amplitude envelope sample).
/// Takes a view so per-step sampling does not allocate.
fn max_abs(field: leto::ArrayView3<f64>) -> f64 {
    field.iter().copied().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Drive the lossless linear Westervelt solver (spectral, periodic) for
/// `n_steps` from the pressure-only initial condition `ic` (zero initial
/// velocity, seeded internally on step 0). Returns the final pressure volume;
/// when `amps` is `Some`, also records `max|p|` after each step.
fn run_westervelt_lossless(
    grid: &Grid,
    medium: &dyn kwavers_medium::Medium,
    ic: &Array3<f64>,
    dt: f64,
    n_steps: usize,
    mut amps: Option<&mut Vec<f64>>,
) -> KwaversResult<Array3<f64>> {
    let mut solver = WesterveltWave::new(grid);
    solver.set_nonlinearity_scaling(0.0);
    solver.set_damping_scaling(0.0);
    let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
    fields.index_axis_mut::<3>(0, 0).expect("index_axis").assign(ic);
    let source = NullSource::new();
    let prev = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for step in 0..n_steps {
        solver.update_wave(
            &mut fields,
            &prev,
            &source,
            grid,
            medium,
            dt,
            step as f64 * dt,
        )?;
        if let Some(a) = amps.as_deref_mut() {
            a.push(max_abs(fields.index_axis::<3>(0, 0).expect("index_axis")));
        }
    }
    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
}

/// As [`run_westervelt_lossless`] for the linear Kuznetsov solver.
fn run_kuznetsov_lossless(
    grid: &Grid,
    medium: &dyn kwavers_medium::Medium,
    ic: &Array3<f64>,
    dt: f64,
    n_steps: usize,
    mut amps: Option<&mut Vec<f64>>,
) -> KwaversResult<Array3<f64>> {
    let mut solver = KuznetsovWave::new(KuznetsovConfig::linear(), grid)?;
    let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
    fields.index_axis_mut::<3>(0, 0).expect("index_axis").assign(ic);
    let source = NullSource::new();
    let prev = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for step in 0..n_steps {
        solver.update_wave(
            &mut fields,
            &prev,
            &source,
            grid,
            medium,
            dt,
            step as f64 * dt,
        )?;
        if let Some(a) = amps.as_deref_mut() {
            a.push(max_abs(fields.index_axis::<3>(0, 0).expect("index_axis")));
        }
    }
    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
}

/// As [`run_westervelt_lossless`] for the PSTD solver run with a transparent
/// [`NullBoundary`] (no PML) so the domain is a lossless periodic resonator.
fn run_pstd_lossless(
    grid: &Grid,
    medium: &dyn kwavers_medium::Medium,
    ic: &Array3<f64>,
    dt: f64,
    n_steps: usize,
    mut amps: Option<&mut Vec<f64>>,
) -> KwaversResult<Array3<f64>> {
    let mut config = PSTDConfig::default();
    config.spectral_correction.enabled = true;
    config.spectral_correction.method = SpectralCorrectionMethod::Treeby2010;
    config.absorption_mode = AbsorptionMode::Lossless;
    config.boundary = kwavers_solver::pstd::config::BoundaryConfig::None;
    config.dt = dt;
    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(PSTDPlugin::new(config, grid)?))?;
    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields.index_axis_mut::<3>(0, 0).expect("index_axis").assign(ic);
    plugin_manager.initialize(grid, medium)?;
    let sources = Vec::new();
    let mut boundary = NullBoundary;
    for step in 0..n_steps {
        plugin_manager.execute(
            &mut fields,
            grid,
            medium,
            &sources,
            &mut boundary,
            dt,
            step as f64 * dt,
        )?;
        if let Some(a) = amps.as_deref_mut() {
            a.push(max_abs(fields.index_axis::<3>(0, 0).expect("index_axis")));
        }
    }
    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
}

/// As [`run_pstd_lossless`] for the 4th-order staggered FDTD solver with a
/// transparent boundary. NOTE: the 4th-order staggered stencil is not
/// periodic-stable without its absorbing layer — used only by the diagnostic to
/// demonstrate the resulting growth, never as a validated lossless path.
fn run_fdtd_lossless(
    grid: &Grid,
    medium: &dyn kwavers_medium::Medium,
    ic: &Array3<f64>,
    dt: f64,
    n_steps: usize,
    mut amps: Option<&mut Vec<f64>>,
) -> KwaversResult<Array3<f64>> {
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: 0.95,
        nt: n_steps,
        dt,
        sensor_mask: None,
        ..Default::default()
    };
    let mut plugin_manager = PluginManager::new();
    plugin_manager.add_plugin(Box::new(FdtdPlugin::new(config, grid)?))?;
    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields.index_axis_mut::<3>(0, 0).expect("index_axis").assign(ic);
    plugin_manager.initialize(grid, medium)?;
    let sources = Vec::new();
    let mut boundary = NullBoundary;
    for step in 0..n_steps {
        plugin_manager.execute(
            &mut fields,
            grid,
            medium,
            &sources,
            &mut boundary,
            dt,
            step as f64 * dt,
        )?;
        if let Some(a) = amps.as_deref_mut() {
            a.push(max_abs(fields.index_axis::<3>(0, 0).expect("index_axis")));
        }
    }
    Ok(fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous())
}

/// Isolate the standing-wave "collapse": is it (a) the absorbing PML, (b) a
/// non-eigenmode IC that dephases across the periodic spectral basis, or (c) a
/// real dissipative defect in a propagator's zero-velocity IVP?
///
/// Method: drive the two spectral solvers (Kuznetsov-linear, Westervelt-lossless,
/// both inherently periodic with NO boundary applied) and PSTD with the external
/// PML removed, on a **lossless** domain. For a lossless standing wave
/// `p=A·sin(kx)·cos(ωt)`, `max|p|` returns to `A` once per half period, so the
/// per-period **peak** of the envelope is conserved iff the solver is lossless.
///
/// EIGENMODE IC (`k·L = 2πm`, a true periodic eigenmode) ⇒ a clean single-mode
/// standing wave: the per-period peak MUST stay ≈ A (no dissipation, correct IVP).
/// NON-EIGENMODE IC (`k·L = 6.4`, the original figure's choice) ⇒ energy leaks
/// into neighbouring DFT bins at different ω, so the **projected** amplitude
/// dephases and the envelope shrinks WITHOUT any energy loss — the real cause of
/// the spectral curves sitting near zero in `fig standing_wave`.
#[test]
fn diagnose_standing_wave_collapse_source() -> KwaversResult<()> {
    // 16³ keeps each plugin run within the 30 s test budget; the m = 2 eigenmode
    // (k·dx ≈ 0.79, 8 cells/λ) is exactly representable by the spectral basis, so
    // the physics under test is unchanged.
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3)?;
    let medium = HomogeneousMedium::water(&grid);
    let c = 1500.0_f64;
    let amplitude = 1e5_f64;
    let l = grid.nx as f64 * grid.dx; // 0.016 m

    // Eigenmode: k·L = 2π·m exactly ⇒ a true periodic eigenmode (m = 2).
    let k_eig = 2.0 * std::f64::consts::PI * 2.0 / l;
    // Non-eigenmode: the original figure's k = 0.2/dx = 200 rad/m (k·L = 6.4).
    let k_bad = 0.2 / grid.dx;

    let dt = fdtd_dt(&grid, c);
    let period = |k: f64| 2.0 * std::f64::consts::PI / (c * k);
    let steps_for = |k: f64, periods: f64| (periods * period(k) / dt).ceil() as usize;

    // Per-period envelope-peak ratio (last full period vs first full period).
    // Returns (peak_first, peak_last) of max|p| sampled every step.
    let envelope_peaks = |amps: &[f64], steps_per_period: usize| -> (f64, f64) {
        let first = amps[..steps_per_period.min(amps.len())]
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let last_start = amps.len().saturating_sub(steps_per_period);
        let last = amps[last_start..].iter().copied().fold(0.0_f64, f64::max);
        (first, last)
    };

    // ── Drive each solver losslessly via the shared helpers ──────────────────
    let periods = 2.0;
    let spp_eig = (period(k_eig) / dt).round() as usize;
    let n_eig = steps_for(k_eig, periods);
    let spp_bad = (period(k_bad) / dt).round() as usize;
    let n_bad = steps_for(k_bad, periods);
    let ic_eig = sinusoidal_ic(&grid, k_eig, amplitude);
    let ic_bad = sinusoidal_ic(&grid, k_bad, amplitude);

    let mut wes_eig = Vec::new();
    run_westervelt_lossless(&grid, &medium, &ic_eig, dt, n_eig, Some(&mut wes_eig))?;
    let mut kuz_eig = Vec::new();
    run_kuznetsov_lossless(&grid, &medium, &ic_eig, dt, n_eig, Some(&mut kuz_eig))?;
    let mut pstd_eig = Vec::new();
    run_pstd_lossless(&grid, &medium, &ic_eig, dt, n_eig, Some(&mut pstd_eig))?;
    let mut fdtd_eig = Vec::new();
    run_fdtd_lossless(&grid, &medium, &ic_eig, dt, n_eig, Some(&mut fdtd_eig))?;
    let mut wes_bad = Vec::new();
    run_westervelt_lossless(&grid, &medium, &ic_bad, dt, n_bad, Some(&mut wes_bad))?;

    let (wf, wl) = envelope_peaks(&wes_eig, spp_eig);
    let (kf, kl) = envelope_peaks(&kuz_eig, spp_eig);
    let (pf, pl) = envelope_peaks(&pstd_eig, spp_eig);
    let (ff, fl) = envelope_peaks(&fdtd_eig, spp_eig);
    let (bf, bl) = envelope_peaks(&wes_bad, spp_bad);

    eprintln!("── standing-wave collapse isolation (lossless, NullBoundary) ──");
    eprintln!("EIGENMODE  k·L = 2π·2  (A0 = {amplitude:.3e} Pa, {periods} periods):");
    eprintln!(
        "  Westervelt  env. peak: first {wf:.3e} → last {wl:.3e}  (ratio {:.3})",
        wl / wf
    );
    eprintln!(
        "  Kuznetsov   env. peak: first {kf:.3e} → last {kl:.3e}  (ratio {:.3})",
        kl / kf
    );
    eprintln!(
        "  PSTD(noPML) env. peak: first {pf:.3e} → last {pl:.3e}  (ratio {:.3})",
        pl / pf
    );
    eprintln!(
        "  FDTD(noPML) env. peak: first {ff:.3e} → last {fl:.3e}  (ratio {:.3})",
        fl / ff
    );
    eprintln!("NON-EIGENMODE k·L = 6.4 (original figure IC):");
    eprintln!(
        "  Westervelt  env. peak: first {bf:.3e} → last {bl:.3e}  (ratio {:.3})",
        bl / bf
    );

    // The spectral solvers are lossless + periodic with a correct zero-velocity
    // IVP: the per-period envelope peak is CONSERVED (ratio ≈ 1) AND reaches A0.
    // This proves the `fig standing_wave` collapse is NOT dissipation and NOT the
    // IVP seed for these three solvers.
    for (name, first, last) in [
        ("Westervelt", wf, wl),
        ("Kuznetsov", kf, kl),
        ("PSTD(noPML)", pf, pl),
    ] {
        let ratio = last / first;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "{name} must CONSERVE a lossless eigenmode standing wave: envelope peak \
             {first:.3e} → {last:.3e} (ratio {ratio:.3} ∉ [0.9, 1.1]) — a real IVP/dissipation defect",
        );
        assert!(
            (first - amplitude).abs() <= 0.1 * amplitude,
            "{name} eigenmode first-period peak {first:.3e} must equal A0 = {amplitude:.3e} (±10%)",
        );
    }

    // FDTD's 4th-order staggered stencil is NOT periodic-stable without its
    // absorbing layer: with a transparent boundary the eigenmode grows far above
    // A0 within the first period. This pins why FDTD is excluded from the lossless
    // resonator comparison (its IVP is validated on the free-space Gaussian test).
    // If FDTD ever conserves here, this guard fires → re-add it to the comparison.
    assert!(
        ff > 1.5 * amplitude,
        "expected FDTD(noPML) to be periodic-unstable (first-period peak ≫ A0); got \
         {ff:.3e} vs A0 {amplitude:.3e} — if FDTD now conserves, re-include it in the lossless test",
    );

    // The NON-eigenmode IC ALSO conserves energy (ratio ≈ 1): the figure collapse
    // is single-mode-oracle dephasing, not amplitude loss.
    let bad_ratio = bl / bf;
    assert!(
        (0.9..=1.1).contains(&bad_ratio),
        "non-eigenmode Westervelt must still conserve energy (ratio ≈ 1); got {bad_ratio:.3}",
    );

    Ok(())
}

/// Numerical dispersion: show FDTD 4th-order vs PSTD/spectral κ(k) curves.
/// Generates a dedicated dispersion comparison figure.
#[test]
fn test_dispersion_characteristics() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3)?;
    let medium = HomogeneousMedium::water(&grid);

    // Use a uniform-amplitude field to exercise all code paths.
    let initial_pressure = &Array3::ones((grid.nx, grid.ny, grid.nz)) * 1e5;

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

/// Regression guard: the PSTD zero-velocity Gaussian IVP must reproduce the exact
/// 3-D analytical d'Alembert solution, AND must do so only when the solver's
/// `config.dt` is synced to the CFL time step used to size `n_steps`.
///
/// Root cause of the historical PSTD "119 %" interior error: the solver advances
/// by `config.dt`, whose default (1e-7 s) is exactly 2× the CFL dt (5e-8 s) for the
/// 0.5 mm / 1500 m·s⁻¹ grid. Leaving it at the default propagated the wave twice as
/// far as the analytical reference (ct doubled) — NOT dissipation, dispersion, the
/// IVP seed, the PML, or the κ method. This test asserts both:
///   (a) with `config.dt = dt` synced, interior rel-L2 vs analytical < 0.05;
///   (b) with `config.dt` left at the 2× default, rel-L2 > 0.5 (the bug signature),
/// so any future regression that desyncs the step is caught value-semantically.
#[test]
fn pstd_gaussian_propagation_matches_analytical() -> KwaversResult<()> {
    use kwavers_solver::pstd::config::BoundaryConfig;

    let n = 32usize;
    let dx = 0.5e-3;
    let grid = Grid::new(n, n, n, dx, dx, dx)?;
    let medium = HomogeneousMedium::water(&grid);
    let c = 1500.0;
    let sigma = 3.0 * dx;
    let t_end = 1.0e-6;
    let (cx, cy, cz) = (n as f64 / 2.0, n as f64 / 2.0, n as f64 / 2.0);
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut ic = Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = ((i as f64 - cx) * dx).powi(2)
                    + ((j as f64 - cy) * dx).powi(2)
                    + ((k as f64 - cz) * dx).powi(2);
                ic[[i, j, k]] = 1e6 * (-r2 / two_sigma2).exp();
            }
        }
    }
    let analytical = analytical_gaussian_ivp(&grid, sigma, 1e6, c, t_end);
    let (lo, hi) = (9usize, n - 9);
    let rel_l2 = |sol: &Array3<f64>| -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for i in lo..hi {
            for j in lo..hi {
                for k in lo..hi {
                    let a = analytical[[i, j, k]];
                    let d = sol[[i, j, k]] - a;
                    num += d * d;
                    den += a * a;
                }
            }
        }
        (num / den.max(1e-30)).sqrt()
    };

    let cfl_dt = 0.15 * dx / c;
    let run = |solver_dt: f64| -> KwaversResult<f64> {
        let mut config = PSTDConfig::default();
        config.spectral_correction.enabled = true;
        config.spectral_correction.method = SpectralCorrectionMethod::Treeby2010;
        config.anti_aliasing.enabled = true;
        config.absorption_mode = AbsorptionMode::Lossless;
        config.boundary = BoundaryConfig::None;
        config.dt = solver_dt;
        let n_steps = (t_end / cfl_dt).ceil() as usize;

        let mut pm = PluginManager::new();
        pm.add_plugin(Box::new(PSTDPlugin::new(config, &grid)?))?;
        let mut fields = Array4::<f64>::zeros((17, n, n, n));
        fields.index_axis_mut::<3>(0, 0).expect("index_axis").assign(&ic);
        pm.initialize(&grid, &medium)?;
        let sources = Vec::new();
        let mut boundary = DomainPMLBoundary::new(DomainPmlConfig::default().with_thickness(8))?;
        for step in 0..n_steps {
            pm.execute(
                &mut fields,
                &grid,
                &medium,
                &sources,
                &mut boundary,
                solver_dt,
                step as f64 * solver_dt,
            )?;
        }
        Ok(rel_l2(&fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous()))
    };

    // (a) synced dt → faithful propagation.
    let e_synced = run(cfl_dt)?;
    assert!(
        e_synced < 0.05,
        "PSTD synced-dt interior rel-L2 = {e_synced:.4} must be < 0.05 (faithful IVP propagation)"
    );

    // (b) 2× dt (the default-config bug) → wave propagates twice as far.
    let e_double = run(2.0 * cfl_dt)?;
    assert!(
        e_double > 0.5,
        "2×-dt run rel-L2 = {e_double:.4} must be > 0.5 (the desync bug signature); \
         got a small error means the desync guard is no longer meaningful"
    );

    Ok(())
}

/// The FullKSpace method (exact second-order dispersion-free propagator
/// `pⁿ⁺¹ = 2cos(c·|k|·Δt)·pⁿ − pⁿ⁻¹`) must reproduce the analytical 3-D Gaussian
/// zero-velocity IVP. Regression guard for the diffusion-equation → wave-equation
/// fix in propagate_kspace (the old forward-Euler kernel diverged to Inf).
///
/// For a homogeneous medium the scheme is analytically exact per mode, so the error
/// is bounded only by the σ=3Δx Gaussian's truncation on the discrete grid — the
/// same band-limit that lets StandardPSTD reach ≈0.03. Tolerance 0.05.
#[test]
fn pstd_fullkspace_gaussian_ivp_matches_analytical() -> KwaversResult<()> {
    use kwavers_solver::pstd::config::{BoundaryConfig, KSpaceMethod};

    let n = 32usize;
    let dx = 0.5e-3;
    let grid = Grid::new(n, n, n, dx, dx, dx)?;
    let medium = HomogeneousMedium::water(&grid);
    let c = 1500.0;
    let sigma = 3.0 * dx;
    let t_end = 1.0e-6;
    let (cx, cy, cz) = (n as f64 / 2.0, n as f64 / 2.0, n as f64 / 2.0);
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut ic = Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = ((i as f64 - cx) * dx).powi(2)
                    + ((j as f64 - cy) * dx).powi(2)
                    + ((k as f64 - cz) * dx).powi(2);
                ic[[i, j, k]] = 1e6 * (-r2 / two_sigma2).exp();
            }
        }
    }
    let analytical = analytical_gaussian_ivp(&grid, sigma, 1e6, c, t_end);

    let cfl_dt = 0.15 * dx / c;
    let config = PSTDConfig {
        kspace_method: KSpaceMethod::FullKSpace,
        absorption_mode: AbsorptionMode::Lossless,
        boundary: BoundaryConfig::None,
        dt: cfl_dt,
        ..Default::default()
    };
    let n_steps = (t_end / cfl_dt).ceil() as usize;

    let mut pm = PluginManager::new();
    pm.add_plugin(Box::new(PSTDPlugin::new(config, &grid)?))?;
    let mut fields = Array4::<f64>::zeros((17, n, n, n));
    fields.index_axis_mut::<3>(0, 0).expect("index_axis").assign(&ic);
    pm.initialize(&grid, &medium)?;
    let sources = Vec::new();
    let mut boundary = DomainPMLBoundary::new(DomainPmlConfig::default().with_thickness(8))?;
    for step in 0..n_steps {
        pm.execute(
            &mut fields,
            &grid,
            &medium,
            &sources,
            &mut boundary,
            cfl_dt,
            step as f64 * cfl_dt,
        )?;
    }
    let out = fields.index_axis::<3>(0, 0).expect("index_axis").to_contiguous();

    assert!(
        out.iter().all(|v| v.is_finite()),
        "FullKSpace produced non-finite values (the old forward-Euler kernel went to Inf)"
    );

    let (lo, hi) = (9usize, n - 9);
    let (mut num, mut den) = (0.0, 0.0);
    for i in lo..hi {
        for j in lo..hi {
            for k in lo..hi {
                let a = analytical[[i, j, k]];
                let d = out[[i, j, k]] - a;
                num += d * d;
                den += a * a;
            }
        }
    }
    let rel = (num / den.max(1e-30)).sqrt();
    assert!(
        rel < 0.05,
        "FullKSpace interior rel-L2 vs analytical = {rel:.4} must be < 0.05"
    );

    Ok(())
}
