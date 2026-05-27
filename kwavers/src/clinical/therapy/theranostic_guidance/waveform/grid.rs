//! Grid construction for the 2-D acoustic waveform simulation.
//!
//! ## Padded domain layout
//!
//! The simulation grid is sized to encompass both the body slice and the
//! transducer aperture, with coupling water filling the cells outside the
//! body and CPML occupying the outermost `PML_CELLS` strip on every side.
//!
//! Padded half-extent on each axis:
//! `H = max(body_half, aperture_half) + λ_water + PML_CELLS · dx`
//!
//! The body sub-region is embedded centred in the padded grid; outside the
//! body the sound speed is `SOUND_SPEED_WATER_SIM` and the attenuation is zero
//! (water is lossless on the scales considered here).
//!
//! References:
//! - Treeby & Cox (2010), J. Acoust. Soc. Am. 128:2741 — k-Wave padded simulation domain.
//! - Komatitsch & Martin (2007), Geophysics 72:SM155, §2 — CPML on outer strip.

use ndarray::Array2;

use super::super::config::TheranosticInverseConfig;
use super::super::medium::PreparedTheranosticSlice;
use super::medium::{reference_speed, speed_bounds};
use super::types::{AcousticGrid, CpmlCoeffs, PaddedSimulation};
use super::utils::linear;
use crate::clinical::therapy::theranostic_guidance::geometry::Point2;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

const PML_CELLS: usize = 12;

pub(super) fn acoustic_grid(
    prepared: &PreparedTheranosticSlice,
    layout: &super::super::geometry::DeviceLayout,
    config: &TheranosticInverseConfig,
    baseline_speed: &Array2<f64>,
    true_speed: &Array2<f64>,
) -> PaddedSimulation {
    let (nx_b, ny_b) = prepared.sound_speed_m_s.dim();
    let dx = prepared.spacing_m;
    let (_, cmax_body) = speed_bounds(baseline_speed, true_speed);
    let frequency_hz = config.frequencies_hz[0];

    // Aperture half-extent on each axis (max |x|, |y| across all elements + receivers).
    let (aperture_half_x, aperture_half_y) = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .fold((0.0_f64, 0.0_f64), |(hx, hy), p| {
            (hx.max(p.x_m.abs()), hy.max(p.y_m.abs()))
        });

    // Body half-extent uses the same centering convention as point_to_cell:
    // origin is at cell ((n-1)/2, (n-1)/2) so the half-extent is (n-1)/2 * dx.
    let body_half_x = (nx_b.saturating_sub(1) as f64) * 0.5 * dx;
    let body_half_y = (ny_b.saturating_sub(1) as f64) * 0.5 * dx;

    // Padded half-extent: cover the larger of body / aperture, plus one
    // wavelength of water and the PML strip on each side.
    let lambda_water = SOUND_SPEED_WATER_SIM / frequency_hz;
    let margin = lambda_water + PML_CELLS as f64 * dx;
    let half_x_needed = body_half_x.max(aperture_half_x) + margin;
    let half_y_needed = body_half_y.max(aperture_half_y) + margin;

    // Convert half-extent in metres to a padded cell count whose parity
    // matches the body dim so that body_offset = (nx_p - nx_b) / 2 places the
    // body's origin cell exactly on the padded grid's origin cell.
    let nx = padded_dim(nx_b, half_x_needed, dx);
    let ny = padded_dim(ny_b, half_y_needed, dx);
    let body_offset = ((nx - nx_b) / 2, (ny - ny_b) / 2);

    // Embed body fields, fill margin with water (lossless).
    let speed_baseline = embed_with_water(baseline_speed, nx, ny, body_offset);
    let speed_true = embed_with_water(true_speed, nx, ny, body_offset);
    let alpha_np_per_step = build_padded_alpha_field(prepared, config, nx, ny, body_offset);

    // The padded cmax is the larger of the body cmax and water (water cells
    // dominate the padded margin so they typically set the CFL bound).
    let cmax = cmax_body.max(SOUND_SPEED_WATER_SIM);
    // CFL stability: λ_CFL = c·dt/dx ≤ 1/√2 (von Neumann, Fornberg 1988, §4).
    let dt_s = 0.35 * dx / (std::f64::consts::SQRT_2 * cmax);

    // Padded domain extent (diagonal half-extent) drives the two-way travel
    // time budget. Water sound speed governs propagation in the coupling
    // margin where the aperture lives.
    let aperture_extent = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .map(|point| point.x_m.hypot(point.y_m))
        .fold(0.0, f64::max);
    let domain_half_x = (nx.saturating_sub(1) as f64) * 0.5 * dx;
    let domain_half_y = (ny.saturating_sub(1) as f64) * 0.5 * dx;
    let domain_extent = domain_half_x.hypot(domain_half_y);
    let travel_time_s =
        2.0 * (aperture_extent + domain_extent) / SOUND_SPEED_WATER_SIM.max(1.0);
    let pulse_time_s = 5.0 / frequency_hz;
    let time_steps = (((travel_time_s + pulse_time_s) / dt_s).ceil() as usize).max(96);

    let focus = layout.focus_m;

    // Place sources at their TRUE physical positions in the padded grid.
    // The only clamp is to keep the 4th-order stencil halo valid: ix, iy in
    // [2, n-3]. Because the padded grid was sized to encompass the aperture
    // plus PML_CELLS ≥ 2 cells of margin, no element is clamped in practice.
    let source_cells: Vec<usize> = layout
        .therapy_elements
        .iter()
        .map(|point| point_to_padded_cell(*point, nx, ny, dx))
        .collect();

    // Source delays computed from the TRUE element-to-focus distance using
    // the water sound speed: the dominant path between a transducer element
    // and the focal point lies in the coupling water margin for clinical
    // bowl apertures whose focal radius (≈ 0.14 m) exceeds the body slice
    // thickness (≈ 0.07 m). Tissue path heterogeneity is not aberration-
    // corrected here; the inverse pipeline downstream is not phase-error
    // robust enough to absorb a misfocused delay law, so we use the
    // dominant-medium delay matching the standard k-Wave bowl-source
    // convention (Treeby & Cox 2010, §III).
    let max_distance = layout
        .therapy_elements
        .iter()
        .map(|p| (p.x_m - focus.x_m).hypot(p.y_m - focus.y_m))
        .fold(0.0, f64::max);
    let source_delays_s: Vec<f64> = layout
        .therapy_elements
        .iter()
        .map(|p| {
            let d = (p.x_m - focus.x_m).hypot(p.y_m - focus.y_m);
            (max_distance - d) / SOUND_SPEED_WATER_SIM.max(1.0)
        })
        .collect();

    // Deduplicate the (rare) collisions that survive after embedding in the
    // padded grid: two adjacent elements may still round to the same cell at
    // the resolution dx. Keep first occurrence so the delay ordering is
    // deterministic. Source scale is renormalised to √N_unique so total
    // injected energy is invariant to dedup.
    let (source_cells, source_delays_s): (Vec<usize>, Vec<f64>) = {
        let mut seen = std::collections::HashSet::new();
        source_cells
            .into_iter()
            .zip(source_delays_s)
            .filter(|(cell, _)| seen.insert(*cell))
            .unzip()
    };
    let source_scale =
        config.source_pressure_pa as f32 / (source_cells.len().max(1) as f32).sqrt();
    let source_frequency_hz = config.frequencies_hz[0];

    let receiver_cells = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .map(|point| point_to_padded_cell(*point, nx, ny, dx))
        .collect();

    // Carrying-medium reference speed (currently unused except via the public
    // medium helper invariants; keep the call to maintain semantic parity in
    // case future regression tests probe these helpers indirectly).
    let _ = reference_speed(prepared, baseline_speed);

    let cpml = build_cpml_coeffs(nx, ny, dx, dt_s, cmax);

    let grid = AcousticGrid {
        nx,
        ny,
        dx_m: dx,
        dt_s,
        time_steps,
        source_cells,
        receiver_cells,
        source_delays_s,
        cpml,
        alpha_np_per_step,
        source_frequency_hz,
        source_scale,
    };

    PaddedSimulation {
        grid,
        speed_baseline,
        speed_true,
        body_offset,
        body_dims: (nx_b, ny_b),
    }
}

/// Compute the padded cell count along one axis from the half-extent in
/// metres. The result has the same parity as `body_n` so the body grid's
/// origin cell coincides with the padded grid's origin cell after embedding.
fn padded_dim(body_n: usize, half_extent_m: f64, dx_m: f64) -> usize {
    let half_cells = (half_extent_m / dx_m).ceil() as usize;
    let mut n = 2 * half_cells + 1;
    if n < body_n + 2 * PML_CELLS + 4 {
        n = body_n + 2 * PML_CELLS + 4;
    }
    if n.is_multiple_of(2) != body_n.is_multiple_of(2) {
        n += 1;
    }
    n
}

/// Map a physical point to a padded-grid cell using the same centring
/// convention as the body-grid `point_to_cell`. Clamp only to the FD-stencil
/// halo bounds `[2, n-3]`; the padded sizing guarantees no element actually
/// hits the clamp.
fn point_to_padded_cell(point: Point2, nx: usize, ny: usize, spacing_m: f64) -> usize {
    let cx = 0.5 * (nx - 1) as f64;
    let cy = 0.5 * (ny - 1) as f64;
    let ix = (point.x_m / spacing_m + cx)
        .round()
        .clamp(2.0, (nx - 3) as f64) as usize;
    let iy = (point.y_m / spacing_m + cy)
        .round()
        .clamp(2.0, (ny - 3) as f64) as usize;
    linear(ix, iy, ny)
}

/// Embed a body-shaped array into a padded `(nx, ny)` array, filling cells
/// outside the embedded region with `SOUND_SPEED_WATER_SIM`.
fn embed_with_water(
    body: &Array2<f64>,
    nx: usize,
    ny: usize,
    offset: (usize, usize),
) -> Array2<f64> {
    let (nx_b, ny_b) = body.dim();
    let (ox, oy) = offset;
    Array2::from_shape_fn((nx, ny), |(ix, iy)| {
        if ix >= ox && iy >= oy && ix < ox + nx_b && iy < oy + ny_b {
            body[[ix - ox, iy - oy]]
        } else {
            SOUND_SPEED_WATER_SIM
        }
    })
}

/// Precompute CPML exponential and scaling coefficients on the OUTER PML
/// strip of the padded grid.
///
/// Reference: Komatitsch & Martin (2007) Geophysics 72:SM155, §2, Eqs. 8–12.
///
/// - `σ_max = -(m+1)·ln(R_target)·c_max / (2·L_pml)`, m=2, R_target=1e-4
/// - `b_i = exp(-(σ_i/κ_i + α_i)·dt)`, with κ_i=1, α_i=0
/// - `a_i = b_i - 1`  (simplifies when κ=1, α=0)
/// - Interior cells: `b=1, a=0`.
fn build_cpml_coeffs(nx: usize, ny: usize, dx_m: f64, dt_s: f64, c_max: f64) -> CpmlCoeffs {
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

/// Precompute per-cell amplitude decay factor on the padded grid.
///
/// Inside the body region: `α_cell = α_np_per_m_mhz · f₀_MHz · dx_m` (Treeby
/// & Cox 2010, §II.A). Outside the body (coupling water): zero.
fn build_padded_alpha_field(
    prepared: &PreparedTheranosticSlice,
    config: &TheranosticInverseConfig,
    nx: usize,
    ny: usize,
    offset: (usize, usize),
) -> Vec<f32> {
    let f0_mhz = config.frequencies_hz[0] * 1.0e-6;
    let dx = prepared.spacing_m;
    let (nx_b, ny_b) = prepared.sound_speed_m_s.dim();
    let (ox, oy) = offset;
    let mut out = vec![0.0_f32; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            if ix >= ox && iy >= oy && ix < ox + nx_b && iy < oy + ny_b {
                let alpha_np_per_m =
                    prepared.attenuation_np_per_m_mhz[[ix - ox, iy - oy]] * f0_mhz;
                out[linear(ix, iy, ny)] = (alpha_np_per_m * dx) as f32;
            }
        }
    }
    out
}
