//! Grid construction for the 2-D acoustic waveform simulation.

use super::super::config::TheranosticInverseConfig;
use super::super::medium::PreparedTheranosticSlice;
use super::medium::{reference_speed, speed_bounds};
use super::types::{AcousticGrid, CpmlCoeffs};
use super::utils::{linear, point_to_cell};

pub(super) fn acoustic_grid(
    prepared: &PreparedTheranosticSlice,
    layout: &super::super::geometry::DeviceLayout,
    config: &TheranosticInverseConfig,
    baseline_speed: &ndarray::Array2<f64>,
    true_speed: &ndarray::Array2<f64>,
) -> AcousticGrid {
    let (nx, ny) = prepared.sound_speed_m_s.dim();
    let (_, cmax) = speed_bounds(baseline_speed, true_speed);
    // CFL stability: λ_CFL = c·dt/dx ≤ 1/√2 (von Neumann, Fornberg 1988, §4).
    let dt_s = 0.35 * prepared.spacing_m / (std::f64::consts::SQRT_2 * cmax);
    let frequency_hz = config.frequencies_hz[0];
    let aperture_extent = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .map(|point| point.x_m.hypot(point.y_m))
        .fold(0.0, f64::max);
    let domain_extent = 0.5 * prepared.spacing_m * nx.max(ny) as f64;
    let delay_speed_m_s = reference_speed(prepared, baseline_speed);
    let travel_time_s = 2.0 * (aperture_extent + domain_extent) / delay_speed_m_s;
    let pulse_time_s = 5.0 / frequency_hz;
    let time_steps = (((travel_time_s + pulse_time_s) / dt_s).ceil() as usize).max(96);
    let focus = layout.focus_m;
    // Source cells are computed first so the delay law can reference the actual
    // clamped grid positions.  Clinical transducer apertures (focal_radius ≈ 0.14 m)
    // routinely exceed the 2-D simulation domain extent (≈ 0.06 m for 42 cells at
    // 1.4 mm spacing).  point_to_cell clamps every out-of-domain element to the
    // nearest valid interior row ix ∈ [2, nx-3].  If delays were derived from the
    // original physical positions all elements would appear equidistant from the
    // focus (arc geometry), producing near-zero differential delays and no coherent
    // 2-D focusing.  Deriving delays from the clamped cell positions instead
    // ensures constructive interference at the focus within the actual grid.
    let source_cells: Vec<usize> = layout
        .therapy_elements
        .iter()
        .map(|point| point_to_cell(*point, nx, ny, prepared.spacing_m))
        .collect();
    // Grid-centre coordinates (metres) used to convert cell indices to physical
    // positions within the 2-D domain.
    let cx = 0.5 * (nx - 1) as f64;
    let cy = 0.5 * (ny - 1) as f64;
    let cell_distances: Vec<f64> = source_cells
        .iter()
        .map(|&cell| {
            let ix = cell / ny;
            let iy = cell % ny;
            let x_m = (ix as f64 - cx) * prepared.spacing_m;
            let y_m = (iy as f64 - cy) * prepared.spacing_m;
            (x_m - focus.x_m).hypot(y_m - focus.y_m)
        })
        .collect();
    let max_cell_distance = cell_distances
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let source_delays_s: Vec<f64> = cell_distances
        .iter()
        .map(|&d| (max_cell_distance - d) / delay_speed_m_s)
        .collect();
    // Deduplicate: clinical transducer arcs whose elements project outside
    // the 2-D domain all clamp to the same boundary row (ix = nx-3).  For the
    // standard Kidney aperture (focal_radius ≈ 0.142 m, half_angle ≈ 54°) on
    // a 42-cell grid, the left half-aperture (θ ∈ [-54°, -8.1°]) maps entirely
    // to cell (nx-3, ny-3) and the right half-aperture extreme angles map to
    // (nx-3, 2).  Retaining duplicates does not increase aperture coverage;
    // it concentrates ~50 % of the injected source energy at a single corner
    // cell, degrades the effective 2-D aperture from ~9 distinct iy positions
    // to 2, and elevates RTM background cross-correlation above the focal peak,
    // producing CNR < 0.  Deduplication restores the physical intent: each
    // unique 2-D grid position contributes one independent source, and the
    // amplitude scale √N_unique keeps the total injected energy consistent with
    // the receiver count (both forward and adjoint replay use the same scale).
    // First occurrence is retained so the delay ordering is deterministic.
    let (source_cells, source_delays_s): (Vec<usize>, Vec<f64>) = {
        let mut seen = std::collections::HashSet::new();
        source_cells
            .into_iter()
            .zip(source_delays_s)
            .filter(|(cell, _)| seen.insert(*cell))
            .unzip()
    };
    // Normalise injected amplitude by √N_unique so total source energy is
    // independent of element count (matched in forward and adjoint replay).
    let source_scale = config.source_pressure_pa as f32
        / (source_cells.len().max(1) as f32).sqrt();
    let source_frequency_hz = config.frequencies_hz[0];
    let receiver_cells = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .map(|point| point_to_cell(*point, nx, ny, prepared.spacing_m))
        .collect();
    let cpml = build_cpml_coeffs(nx, ny, prepared.spacing_m, dt_s, cmax);
    let alpha_np_per_step = build_alpha_field(prepared, config, nx, ny);
    AcousticGrid {
        nx,
        ny,
        dx_m: prepared.spacing_m,
        dt_s,
        time_steps,
        source_cells,
        receiver_cells,
        source_delays_s,
        cpml,
        alpha_np_per_step,
        source_frequency_hz,
        source_scale,
    }
}

/// Precompute CPML exponential and scaling coefficients.
///
/// Reference: Komatitsch & Martin (2007) Geophysics 72:SM155, §2, Eqs. 8–12.
///
/// - `σ_max = -(m+1)·ln(R_target)·c_max / (2·L_pml)`, m=2, R_target=1e-4
/// - `b_i = exp(-(σ_i/κ_i + α_i)·dt)`, with κ_i=1, α_i=0
/// - `a_i = b_i - 1`  (simplifies when κ=1, α=0)
/// - Interior cells: `b=1, a=0`.
fn build_cpml_coeffs(nx: usize, ny: usize, dx_m: f64, dt_s: f64, c_max: f64) -> CpmlCoeffs {
    const PML_CELLS: usize = 12;
    const PML_ORDER: f64 = 2.0;
    const R_TARGET: f64 = 1.0e-4;

    let l_pml = PML_CELLS as f64 * dx_m;
    let sigma_max = -(PML_ORDER + 1.0) * R_TARGET.ln() * c_max / (2.0 * l_pml);

    let build_1d = |len: usize| -> (Vec<f32>, Vec<f32>) {
        let mut b = vec![1.0_f32; len];
        let mut a = vec![0.0_f32; len];
        for i in 0..len.min(PML_CELLS) {
            let d_norm = (PML_CELLS - i) as f64 / PML_CELLS as f64;
            let sigma = sigma_max * d_norm.powf(PML_ORDER);
            let b_val = (-sigma * dt_s).exp();
            b[i] = b_val as f32;
            a[i] = (b_val - 1.0) as f32;
        }
        for i in len.saturating_sub(PML_CELLS)..len {
            let d_norm = (i - (len - PML_CELLS)) as f64 / PML_CELLS as f64;
            let sigma = sigma_max * d_norm.powf(PML_ORDER);
            let b_val = (-sigma * dt_s).exp();
            b[i] = b_val as f32;
            a[i] = (b_val - 1.0) as f32;
        }
        (b, a)
    };

    let (b_x, a_x) = build_1d(nx);
    let (b_y, a_y) = build_1d(ny);
    CpmlCoeffs { b_x, a_x, b_y, a_y }
}

/// Precompute per-cell amplitude decay factor per time step.
///
/// `α_cell = α_np_per_m_mhz · f₀_MHz · dx_m` (Np/step, first-order approximation).
///
/// Reference: Treeby & Cox (2010) J. Acoust. Soc. Am. 128:2741, §II.A.
fn build_alpha_field(
    prepared: &PreparedTheranosticSlice,
    config: &TheranosticInverseConfig,
    nx: usize,
    ny: usize,
) -> Vec<f32> {
    let f0_mhz = config.frequencies_hz[0] * 1.0e-6;
    let dx = prepared.spacing_m;
    let mut out = vec![0.0_f32; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            let alpha_np_per_m = prepared.attenuation_np_per_m_mhz[[ix, iy]] * f0_mhz;
            out[linear(ix, iy, ny)] = (alpha_np_per_m * dx) as f32;
        }
    }
    out
}
