//! Grid construction for the 2-D acoustic waveform simulation.

use super::super::config::TheranosticInverseConfig;
use super::super::medium::PreparedTheranosticSlice;
use super::medium::{reference_speed, speed_bounds};
use super::types::{AcousticGrid, CpmlCoeffs};
use super::utils::{distance, linear, point_to_cell};

pub(super) fn acoustic_grid(
    prepared: &PreparedTheranosticSlice,
    layout: &super::super::geometry::DeviceLayout,
    config: &TheranosticInverseConfig,
    baseline_speed: &ndarray::Array2<f64>,
    true_speed: &ndarray::Array2<f64>,
) -> AcousticGrid {
    let (nx, ny) = prepared.sound_speed_m_s.dim();
    let (cmin, cmax) = speed_bounds(baseline_speed, true_speed);
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
    let travel_time_s = 2.0 * (aperture_extent + domain_extent) / cmin;
    let pulse_time_s = 5.0 / frequency_hz;
    let time_steps = (((travel_time_s + pulse_time_s) / dt_s).ceil() as usize).max(96);
    let delay_speed_m_s = reference_speed(prepared, baseline_speed);
    let focus = layout.focus_m;
    let source_distances = layout
        .therapy_elements
        .iter()
        .map(|source| distance(*source, focus))
        .collect::<Vec<_>>();
    let max_distance = source_distances
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let source_delays_s = source_distances
        .into_iter()
        .map(|d| (max_distance - d) / delay_speed_m_s)
        .collect();
    let source_cells = layout
        .therapy_elements
        .iter()
        .map(|point| point_to_cell(*point, nx, ny, prepared.spacing_m))
        .collect();
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
