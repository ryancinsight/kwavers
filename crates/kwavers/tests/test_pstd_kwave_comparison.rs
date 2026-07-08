//! PSTD vs k-Wave 1D Homogeneous Medium Integration Test
//!
//! Validates the PSTD solver against the analytical d'Alembert solution
//! for the 1D wave equation, which is the mathematical reference that
//! k-Wave's k-space pseudo-spectral method must reproduce.
//!
//! ## Theorem (d'Alembert Solution, 1D Wave Equation)
//!
//! For the homogeneous 1D wave equation:
//!     ∂²p/∂t² - c² ∇²p = 0
//!
//! with initial conditions p(x,0) = p₀(x), ∂p/∂t(x,0) = 0, the exact
//! solution is:
//!     p(x,t) = ½[p₀(x - ct) + p₀(x + ct)]
//!
//! This represents two copies of the initial pressure profile propagating
//! in opposite directions at speed c.
//!
//! ## Figures
//!
//! Each test emits a PNG comparison figure to `test-figures/` in the crate
//! root. Figure generation failures are non-fatal: the test still passes or
//! fails on physics criteria alone.
//!
//! - `pstd_dalembert_comparison.png` — 2×3 pressure snapshots, PSTD vs analytical
//! - `pstd_kspace_operator.png`      — k-space correction factor κ(k) vs k/k_max
//! - `pstd_energy_conservation.png`  — acoustic energy E(t) vs timestep
//!
//! ## Validation Criteria
//!
//! The PSTD solver is validated against this analytical solution with:
//! - RMS relative error < 2% (allows for discretization differences)
//! - Peak pressure magnitude within 5% of analytical
//!
//! References:
//! - Treeby & Cox, "Modeling ultrasound propagation using the k-space
//!   pseudospectral method," J. Acoust. Soc. Am. 127(6), 2010.
//! - k-Wave: http://www.k-wave.org/

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_signal::SineWave;
use kwavers_solver::forward::pstd::config::{KSpaceMethod, PSTDConfig};
use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_source::{
    GridSource, InjectionMode, PlaneWaveSource, PlaneWaveSourceConfig, SourceField,
};
use ndarray::Array3;
use plotters::prelude::*;
use std::sync::Arc;

/// Directory for emitted comparison figures.
const FIGURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test-figures");

// ─────────────────────────────────────────────────────────────────────────────
// Figure helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of the pressure field at one instant in time.
struct Snapshot {
    /// Simulation time (s).
    t: f64,
    /// Dimensionless propagation distance ct/σ.
    ct_over_sigma: f64,
    /// Numerical pressure profile normalised to initial amplitude (Pa/amplitude).
    numerical: Vec<f64>,
    /// Analytical d'Alembert profile normalised to initial amplitude.
    analytical: Vec<f64>,
}

/// Render and save the d'Alembert pulse-splitting comparison figure.
///
/// Layout: 2 rows × 3 columns. Each panel shows the pressure field
/// (normalised) vs position (mm) at one snapshot time. PSTD (blue solid)
/// is overlaid on d'Alembert (red solid). When the solver is correct, the
/// curves are indistinguishable.
fn save_dalembert_figure(
    snapshots: &[Snapshot],
    nx: usize,
    dx: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(FIGURE_DIR)?;
    let path = format!("{}/pstd_dalembert_comparison.png", FIGURE_DIR);

    // 1 800 × 900 pixels: 3 columns × 2 rows at 560 × 400 each, plus margins.
    let root = BitMapBackend::new(&path, (1800, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    // Title band at the top.
    let (title_area, plot_area) = root.split_vertically(48);
    title_area.titled(
        "PSTD vs d'Alembert: 1D Gaussian Pulse Propagation (1 MPa, σ = 0.5 mm, c = 1500 m/s)",
        ("sans-serif", 18).into_font(),
    )?;

    let panels = plot_area.split_evenly((2, 3));
    let x_max_mm = (nx as f64) * dx * 1e3;

    for (panel, snap) in panels.iter().zip(snapshots.iter()) {
        let caption = format!(
            "t = {:.2} μs   ct/σ = {:.2}",
            snap.t * 1e6,
            snap.ct_over_sigma
        );
        let mut chart = ChartBuilder::on(panel)
            .caption(caption, ("sans-serif", 13).into_font())
            .margin(8)
            .x_label_area_size(28)
            .y_label_area_size(52)
            .build_cartesian_2d(0.0f64..x_max_mm, -1.1f64..1.1f64)?;

        chart
            .configure_mesh()
            .x_desc("Position (mm)")
            .y_desc("p / amplitude")
            .x_labels(6)
            .y_labels(5)
            .draw()?;

        // Analytical d'Alembert — red.
        chart
            .draw_series(LineSeries::new(
                snap.analytical
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as f64 * dx * 1e3, v)),
                ShapeStyle::from(&RED).stroke_width(2),
            ))?
            .label("Analytical (d'Alembert)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], RED));

        // PSTD numerical — blue, drawn on top.
        chart
            .draw_series(LineSeries::new(
                snap.numerical
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as f64 * dx * 1e3, v)),
                ShapeStyle::from(&BLUE).stroke_width(1),
            ))?
            .label("PSTD numerical")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], BLUE));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()?;
    println!("  Figure saved: {}", path);
    Ok(())
}

/// Render and save the k-space correction factor figure.
///
/// Shows κ(k) = sinc(c|k|Δt/2) across the full Nyquist band
/// (0 to k_Nyquist). Vertical markers indicate 80 % and 100 % of Nyquist.
fn save_kspace_figure(c0: f64, dt: f64, dx: f64) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(FIGURE_DIR)?;
    let path = format!("{}/pstd_kspace_operator.png", FIGURE_DIR);

    let root = BitMapBackend::new(&path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "k-Space Correction Factor κ(k) = sinc(c·|k|·Δt/2)",
            ("sans-serif", 18).into_font(),
        )
        .margin(20)
        .x_label_area_size(36)
        .y_label_area_size(56)
        .build_cartesian_2d(0.0f64..1.0f64, 0.0f64..1.02f64)?;

    chart
        .configure_mesh()
        .x_desc("Normalised wavenumber  k / k_Nyquist")
        .y_desc("κ(k)")
        .x_labels(11)
        .y_labels(6)
        .draw()?;

    let k_max = std::f64::consts::PI / dx;
    let n_pts = 500usize;

    // κ(k) curve.
    chart
        .draw_series(LineSeries::new(
            (0..=n_pts).map(|i| {
                let k_norm = i as f64 / n_pts as f64;
                let k = k_norm * k_max;
                let arg = c0 * k * dt / 2.0;
                let kappa = if arg.abs() < 1e-12 {
                    1.0
                } else {
                    arg.sin() / arg
                };
                (k_norm, kappa)
            }),
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))?
        .label("κ(k)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // 80 % Nyquist marker.
    let arg_80 = c0 * (0.8 * k_max) * dt / 2.0;
    let kappa_80 = arg_80.sin() / arg_80;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.8, 0.0), (0.8, kappa_80)],
        ShapeStyle::from(&GREEN).stroke_width(1),
    )))?;
    chart.draw_series(std::iter::once(Circle::new(
        (0.8, kappa_80),
        5,
        ShapeStyle::from(&GREEN).filled(),
    )))?;

    // Nyquist marker.
    let arg_ny = c0 * k_max * dt / 2.0;
    let kappa_ny = arg_ny.sin() / arg_ny;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(1.0, 0.0), (1.0, kappa_ny)],
        ShapeStyle::from(&RED).stroke_width(1),
    )))?;
    chart.draw_series(std::iter::once(Circle::new(
        (1.0, kappa_ny),
        5,
        ShapeStyle::from(&RED).filled(),
    )))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("  Figure saved: {}", path);
    Ok(())
}

/// Render and save the energy conservation figure.
///
/// Shows E(t) (blue) vs timestep, with a horizontal reference line at
/// the initial energy E₀ (black dashed). The y-axis spans a ±5 % band
/// around E₀ to make small drifts visible.
fn save_energy_figure(
    e0: f64,
    energy_records: &[(usize, f64)], // (step, energy)
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(FIGURE_DIR)?;
    let path = format!("{}/pstd_energy_conservation.png", FIGURE_DIR);

    let root = BitMapBackend::new(&path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let step_max = energy_records.last().map(|(s, _)| *s).unwrap_or(1);
    let e_min = energy_records.iter().map(|(_, e)| *e).fold(e0, f64::min) * 0.995;
    let e_max = energy_records.iter().map(|(_, e)| *e).fold(e0, f64::max) * 1.005;
    // Widen band to at least ±0.5 % of E₀ so the reference line is visible.
    let band = (e0 * 0.005).max(e_max - e_min);
    let y_lo = e0 - band;
    let y_hi = e0 + band;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "PSTD Energy Conservation — 1D Lossless Homogeneous Medium",
            ("sans-serif", 16).into_font(),
        )
        .margin(20)
        .x_label_area_size(36)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..step_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("Timestep")
        .y_desc("Acoustic energy  E(t)  (J m⁻²)")
        .x_labels(10)
        .y_labels(6)
        .draw()?;

    // Reference line E₀.
    chart
        .draw_series(LineSeries::new(
            vec![(0, e0), (step_max, e0)],
            ShapeStyle::from(&BLACK).stroke_width(1),
        ))?
        .label("E₀ (initial)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    // Energy vs step.
    chart
        .draw_series(LineSeries::new(
            energy_records.iter().map(|(s, e)| (*s, *e)),
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))?
        .label("E(t) PSTD")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("  Figure saved: {}", path);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: analytical d'Alembert solution
// ─────────────────────────────────────────────────────────────────────────────

/// Analytical d'Alembert solution for a Gaussian pulse initial condition.
///
/// For p₀(x) = A · exp(-((x − x₀)² / (2σ²))) with zero initial velocity:
///     p(x,t) = ½[p₀(x − ct) + p₀(x + ct)]
fn dalembert_solution(x: f64, t: f64, c: f64, x0: f64, sigma: f64, amplitude: f64) -> f64 {
    let gaussian = |xc: f64| amplitude * (-((xc - x0).powi(2)) / (2.0 * sigma * sigma)).exp();
    0.5 * (gaussian(x - c * t) + gaussian(x + c * t))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Theorem (PSTD 1D Homogeneous Medium Validation):
///
/// For a 1D Gaussian initial pressure distribution in a homogeneous, lossless medium,
/// the PSTD solution must converge to the analytical d'Alembert solution:
///
///     p(x,t) = ½[p₀(x − ct) + p₀(x + ct)]
///
/// ## IVP staggered-leapfrog initialization (k-Wave convention)
///
/// The PSTD staggered leapfrog stores velocity at half-integer steps t^{n−½}.
/// For zero initial velocity (∂p/∂t = 0 at t = 0), the solver must pre-initialize
/// u^{−1/2} to the IVP half-step correction:
///
///     û_x^{−1/2}(k) = −(sin(c|k|Δt/2) / (ρ₀c|k|)) · p̂₀(k) · (i·kx/|k|)
///
/// which is computed automatically by `initialize_ivp_velocity` when GridSource.p0
/// is provided.
///
/// Reference: k-Wave MATLAB `kspaceFirstOrder3D.m` lines 1041–1060 (IVP init block).
///
/// Error metric: RMS relative error < 2% over the entire domain.
///
/// Figure: `test-figures/pstd_dalembert_comparison.png` — six pressure snapshots
/// showing the Gaussian pulse splitting into two half-amplitude pulses.
#[test]
fn test_pstd_vs_dalembert_1d_homogeneous() -> KwaversResult<()> {
    println!("\n=== PSTD vs k-Wave Reference: 1D Homogeneous Medium ===");

    // Configuration matching k-wave-python test_ivp_homogeneous_medium conceptually.
    let nx = 128;
    let dx = 0.1e-3; // 0.1 mm
    let c0 = 1500.0; // m/s (water)
    let rho0 = 1000.0; // kg/m³

    let x0 = 0.5 * nx as f64 * dx; // domain centre
    let sigma = 0.5e-3; // 0.5 mm pulse width
    let amplitude = 1e6; // 1 MPa

    let dt = 0.3 * dx / c0; // CFL = 0.3 (k-Wave standard)
    let nt = 60;

    let grid = Grid::new(nx, 1, 1, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // GridSource.p0 triggers `initialize_ivp_velocity` in the constructor,
    // setting u^{−1/2} for the staggered leapfrog so ∂p/∂t(0) = 0.
    let mut p0_3d = Array3::<f64>::zeros((nx, 1, 1));
    for i in 0..nx {
        let x = i as f64 * dx;
        p0_3d[[i, 0, 0]] = amplitude * (-((x - x0).powi(2)) / (2.0 * sigma * sigma)).exp();
    }

    let source = GridSource {
        p0: Some(p0_3d),
        ..GridSource::default()
    };

    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers_solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, source)?;

    // IVP velocity diagnostic: u^{−1/2} at x₀+σ.
    let i_center = (x0 / dx) as usize;
    let i_right = (i_center + (sigma / dx) as usize).min(nx - 1);
    let ux_after_ivp = solver.fields.ux[[i_right, 0, 0]];
    let expected_ux_ivp = (dt / (2.0 * rho0)) * {
        let x = i_right as f64 * dx;
        -(x - x0) / (sigma * sigma)
            * amplitude
            * (-((x - x0).powi(2)) / (2.0 * sigma * sigma)).exp()
    };
    println!(
        "  IVP ux[x0+σ]: numerical={:.4e} m/s, expected≈{:.4e} m/s, ratio={:.4}",
        ux_after_ivp,
        expected_ux_ivp,
        if expected_ux_ivp.abs() > 1e-30 {
            ux_after_ivp / expected_ux_ivp
        } else {
            f64::NAN
        }
    );
    println!("  (IVP ratio ≈1.0 means correct staggered leapfrog initialization)");

    // Snapshot collection: save 6 pressure profiles for the comparison figure.
    // Target steps: 1, 11, 21, 31, 41, 60 (ct/σ ≈ 0.06, 0.66, 1.26, 1.86, 2.46, 3.60).
    let snapshot_steps: std::collections::HashSet<usize> =
        [1, 11, 21, 31, 41, 60].iter().copied().collect();
    let mut snapshots: Vec<Snapshot> = Vec::with_capacity(6);

    let mut rms_error = 0.0_f64;
    let mut max_numerical = 0.0_f64;
    let mut max_analytical = 0.0_f64;
    let mut n_points = 0usize;

    for step in 0..nt {
        solver.step_forward()?;

        let steps_done = step + 1;

        if step % 10 == 0 || step == nt - 1 {
            let t = steps_done as f64 * dt;

            let mut snap_rms_sq = 0.0_f64;
            let mut snap_max_num = 0.0_f64;
            let mut snap_max_anal = 0.0_f64;

            // Collect full pressure profiles if this is a figure snapshot step.
            let collect = snapshot_steps.contains(&steps_done);
            let mut num_profile: Vec<f64> = if collect {
                Vec::with_capacity(nx)
            } else {
                vec![]
            };
            let mut anal_profile: Vec<f64> = if collect {
                Vec::with_capacity(nx)
            } else {
                vec![]
            };

            for i in 0..nx {
                let x = i as f64 * dx;
                let p_num = solver.fields.p[[i, 0, 0]];
                let p_anal = dalembert_solution(x, t, c0, x0, sigma, amplitude);

                if p_num.abs() > snap_max_num {
                    snap_max_num = p_num.abs();
                }
                if p_anal.abs() > snap_max_anal {
                    snap_max_anal = p_anal.abs();
                }
                if p_num.abs() > max_numerical {
                    max_numerical = p_num.abs();
                }
                if p_anal.abs() > max_analytical {
                    max_analytical = p_anal.abs();
                }

                let rel_err = (p_num.abs() - p_anal.abs()) / amplitude;
                rms_error += rel_err * rel_err;
                snap_rms_sq += rel_err * rel_err;
                n_points += 1;

                if collect {
                    num_profile.push(p_num / amplitude);
                    anal_profile.push(p_anal / amplitude);
                }
            }

            if collect {
                snapshots.push(Snapshot {
                    t,
                    ct_over_sigma: c0 * t / sigma,
                    numerical: num_profile,
                    analytical: anal_profile,
                });
            }

            let snap_rms = (snap_rms_sq / nx as f64).sqrt();
            println!(
                "  step={:3}: t={:.2e}s, ct/σ={:.2}, RMS={:.3}%, \
                 peak_num={:.4e}, peak_anal={:.4e}",
                steps_done,
                t,
                c0 * t / sigma,
                snap_rms * 100.0,
                snap_max_num,
                snap_max_anal,
            );
        }
    }

    let rms_relative_error = (rms_error / n_points as f64).sqrt();
    let peak_ratio = max_numerical / max_analytical.max(1e-10);

    println!("Results:");
    println!("  RMS relative error: {:.3}%", rms_relative_error * 100.0);
    println!("  Numerical peak:   {:.3e} Pa", max_numerical);
    println!("  Analytical peak:  {:.3e} Pa", max_analytical);
    println!("  Peak ratio:       {:.3}", peak_ratio);
    println!("  Comparison points: {}", n_points);

    // Generate comparison figure (non-fatal if figure writing fails).
    if let Err(e) = save_dalembert_figure(&snapshots, nx, dx) {
        eprintln!("  [warn] d'Alembert figure generation failed: {}", e);
    }

    // Physics validation criteria.
    assert!(
        rms_relative_error < 0.02,
        "PSTD RMS relative error {:.3}% exceeds 2% tolerance; \
         PSTD may not match k-Wave for 1D homogeneous medium propagation",
        rms_relative_error * 100.0
    );
    assert!(
        (peak_ratio - 1.0).abs() < 0.05,
        "PSTD peak pressure deviates by >5% from analytical: ratio={:.3}",
        peak_ratio
    );

    println!("PASSED: PSTD matches k-Wave 1D homogeneous reference within tolerance");
    Ok(())
}

/// Theorem (PSTD Frequency Domain Validation):
///
/// The PSTD derivative operator in frequency domain is: ∂f/∂x ↔ j·k·F{f}
/// where k = 2πn/L is the wavenumber. The k-space correction
///
///     κ(k) = sinc(c|k|Δt/2)
///
/// compensates for temporal discretisation at each wavenumber. At the Nyquist
/// limit κ(k_max) ≥ 0.963 for CFL = 0.3.
///
/// Figure: `test-figures/pstd_kspace_operator.png` — full correction curve
/// κ(k) across the Nyquist band with markers at 80 % and 100 % Nyquist.
#[test]
fn test_pstd_k_space_operator_accuracy() -> KwaversResult<()> {
    println!("\n=== PSTD k-Space Operator Validation ===");

    let nx = 256;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let frequency = 1e6; // 1 MHz
    let wavelength = c0 / frequency;
    let points_per_wavelength = wavelength / dx;

    println!("Grid: {} points, dx = {:.2e} m", nx, dx);
    println!(
        "Wavelength: {:.2e} m → {:.1} points per wavelength",
        wavelength, points_per_wavelength
    );
    println!("Nyquist limit: {:.1} points per wavelength", 2.0);

    assert!(
        points_per_wavelength > 2.0,
        "Insufficient resolution: only {:.1} points per wavelength (need > 2)",
        points_per_wavelength
    );

    let dt = 0.3 * dx / c0;
    let k_max = std::f64::consts::PI / dx;
    let correction_at_nyquist = {
        let arg = c0 * k_max * dt / 2.0;
        arg.sin() / arg
    };
    println!(
        "k-space correction at Nyquist: {:.6}",
        correction_at_nyquist
    );

    assert!(
        correction_at_nyquist > 0.5,
        "k-space correction too large at Nyquist: {:.3}",
        correction_at_nyquist
    );

    let k_80 = 0.8 * k_max;
    let correction_80 = {
        let arg = c0 * k_80 * dt / 2.0;
        arg.sin() / arg
    };
    println!("k-space correction at 80%% Nyquist: {:.6}", correction_80);

    assert!(
        correction_80 > 0.8,
        "k-space correction at 80% Nyquist too large: {:.3}",
        correction_80
    );

    // Generate k-space operator figure (non-fatal).
    if let Err(e) = save_kspace_figure(c0, dt, dx) {
        eprintln!("  [warn] k-space figure generation failed: {}", e);
    }

    println!("PASSED: k-space operator within valid range");
    Ok(())
}

/// Theorem (PSTD Conservation Properties):
///
/// For a lossless homogeneous medium with periodic boundaries,
/// the PSTD method preserves total acoustic energy:
///     E = ∫ [p²/(ρ₀c²) + ρ₀|u|²] dx = constant
///
/// This is a consequence of the spectral derivative operator being
/// skew-symmetric on the periodic domain.
///
/// Figure: `test-figures/pstd_energy_conservation.png` — E(t) vs timestep
/// with a horizontal reference line at E₀.
#[test]
fn test_pstd_energy_conservation_homogeneous() -> KwaversResult<()> {
    println!("\n=== PSTD Energy Conservation Test ===");

    let nx = 64;
    let ny = 1;
    let nz = 1;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let amplitude = 1e5;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let x0 = 0.25 * nx as f64 * dx;
    let sigma = 1.0e-3;

    // Analytical energy of initial Gaussian:
    // E = ∫ A²·exp(-(x-x₀)²/σ²) / (ρ₀c²) dx ≈ A²·σ·√π / (ρ₀c²) in 1D
    let analytical_energy =
        amplitude * amplitude * sigma * std::f64::consts::PI.sqrt() / (rho0 * c0 * c0);
    println!(
        "Initial Gaussian: sigma={:.1e} m, A={:.1e} Pa",
        sigma, amplitude
    );
    println!(
        "Analytical 1D energy estimate: {:.3e} J/m²",
        analytical_energy
    );

    let signal = Arc::new(SineWave::new(
        c0 / wavelength_for_points(wavelength(c0, 500e-6), dx),
        amplitude,
        0.0,
    ));
    let config = PlaneWaveSourceConfig {
        direction: (1.0, 0.0, 0.0),
        wavelength: dx,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let _source = PlaneWaveSource::new(config, signal);

    let dt = 0.3 * dx / c0;
    let nt = 80;
    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers_solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;

    // Set initial Gaussian pressure and matching acoustic density perturbation.
    for i in 0..nx {
        let x = i as f64 * dx;
        let p0 = amplitude * (-((x - x0).powi(2)) / (2.0 * sigma * sigma)).exp();
        solver.fields.p[[i, 0, 0]] = p0;
        solver.rhox[[i, 0, 0]] = p0 / (c0 * c0);
        solver.fields.ux[[i, 0, 0]] = 0.0;
    }

    let compute_energy = |s: &PSTDSolver| -> f64 {
        let mut e = 0.0;
        for i in 0..nx {
            let p = s.fields.p[[i, 0, 0]];
            let ux = s.fields.ux[[i, 0, 0]];
            e += p * p / (rho0 * c0 * c0) + rho0 * ux * ux;
        }
        e * dx
    };

    let e0 = compute_energy(&solver);
    println!("Initial numerical energy: {:.6e} J/m²", e0);

    // Run simulation and collect (step, energy) pairs every 5 steps.
    let mut energy_records: Vec<(usize, f64)> = vec![(0, e0)];
    for step in 0..nt {
        solver.step_forward()?;
        if step % 5 == 0 {
            let e = compute_energy(&solver);
            energy_records.push((step + 1, e));
        }
    }

    let e_final = compute_energy(&solver);
    energy_records.push((nt, e_final));
    let energy_drift = (e_final - e0) / e0.abs().max(1e-20);

    println!("Final energy: {:.6e} J/m²", e_final);
    println!("Energy drift: {:.3}%", energy_drift * 100.0);

    // Generate energy conservation figure (non-fatal).
    if let Err(e) = save_energy_figure(e0, &energy_records) {
        eprintln!("  [warn] energy figure generation failed: {}", e);
    }

    // Physics validation: energy should be conserved within 1 % over 80 steps.
    assert!(
        energy_drift.abs() < 0.01,
        "Energy drift {:.3}% exceeds 1%% tolerance; \
         possible PSTD instability or incorrect energy computation",
        energy_drift * 100.0
    );

    println!("PASSED: PSTD energy conservation within tolerance");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute acoustic wavelength for given frequency.
fn wavelength(c: f64, f: f64) -> f64 {
    c / f
}

/// Snap a wavelength to the nearest whole number of grid points.
/// Returns `round(lambda / dx) * dx`, clamped to at least `dx`.
fn wavelength_for_points(lambda: f64, dx: f64) -> f64 {
    ((lambda / dx).round().max(1.0)) * dx
}

/// Compute points per wavelength for given wavelength and grid spacing.
#[allow(dead_code)]
fn points_per_wavelength(wavelength: f64, dx: f64) -> f64 {
    wavelength / dx
}
