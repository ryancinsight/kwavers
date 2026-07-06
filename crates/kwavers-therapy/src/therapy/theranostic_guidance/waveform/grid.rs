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
use super::cavitation::{
    cavitation_burst_duration_s, cavitation_emission_waveform, CAV_MAX_LINE_MULTIPLE,
};
use super::medium::{reference_speed, speed_bounds};
use super::types::{AcousticGrid, CpmlCoeffs, PaddedSimulation};
use super::utils::linear;
use crate::therapy::theranostic_guidance::geometry::Point2;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

const PML_CELLS: usize = 12;

pub(super) fn acoustic_grid(
    prepared: &PreparedTheranosticSlice,
    layout: &super::super::geometry::DeviceLayout,
    config: &TheranosticInverseConfig,
    baseline_speed: &Array2<f64>,
    true_speed: &Array2<f64>,
) -> PaddedSimulation {
    let (nx_b_coarse, ny_b_coarse) = prepared.sound_speed_m_s.dim();
    let dx_coarse = prepared.spacing_m;
    let (_, cmax_body) = speed_bounds(baseline_speed, true_speed);
    let cmin_water = SOUND_SPEED_WATER_SIM.min(cmax_body);
    let frequency_hz = config.frequencies_hz[0];

    // ── Internal grid refinement ─────────────────────────────────────────
    //
    // The caller's body grid is sized for the reduced-Born inverse
    // (~52 cells across the abdomen, dx ≈ 6 mm).  At clinical transmit
    // frequencies that gives `λ / dx ≈ 0.5` — far below the 4th-order FD
    // stencil's ~4-point/wavelength accuracy threshold, so the wave
    // dissipates into numerical noise after a handful of cells and no
    // focal spot forms.  Refine the spatial grid internally so the FDTD
    // sees a properly-resolved wavelength while the caller-visible body
    // grid is preserved.
    let refinement = required_refinement(dx_coarse, cmin_water, frequency_hz);
    let dx = dx_coarse / refinement as f64;
    let nx_b = nx_b_coarse * refinement;
    let ny_b = ny_b_coarse * refinement;

    let baseline_refined = upsample_field(baseline_speed, refinement);
    let true_refined = upsample_field(true_speed, refinement);

    // Aperture half-extent on each axis (max |x|, |y| across all elements + receivers).
    let (aperture_half_x, aperture_half_y) = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .fold((0.0_f64, 0.0_f64), |(hx, hy), p| {
            (hx.max(p.x_m.abs()), hy.max(p.y_m.abs()))
        });

    // Body half-extent: origin is at cell ((n-1)/2, (n-1)/2).  Computed on
    // the REFINED grid so the body sub-region is sized in refined cells.
    let body_half_x = (nx_b.saturating_sub(1) as f64) * 0.5 * dx;
    let body_half_y = (ny_b.saturating_sub(1) as f64) * 0.5 * dx;

    // Padded half-extent: cover the larger of body / aperture, plus one
    // wavelength of water and the PML strip on each side.
    let lambda_water = SOUND_SPEED_WATER_SIM / frequency_hz;
    let margin = lambda_water + PML_CELLS as f64 * dx;
    let half_x_needed = body_half_x.max(aperture_half_x) + margin;
    let half_y_needed = body_half_y.max(aperture_half_y) + margin;

    let nx = padded_dim(nx_b, half_x_needed, dx);
    let ny = padded_dim(ny_b, half_y_needed, dx);
    let body_offset = ((nx - nx_b) / 2, (ny - ny_b) / 2);

    // Embed refined body fields, fill margin with water (lossless).
    // The propagator now sees the FULL heterogeneous speed map (bone,
    // soft tissue, water) — aberration of the focused beam through this
    // medium is compensated by the Eikonal-based focal law below, which
    // is the high-frequency closed-form solution that an aberration-
    // corrected clinical FUS scanner approximates.
    let speed_baseline = embed_with_water(&baseline_refined, nx, ny, body_offset);
    let speed_true = embed_with_water(&true_refined, nx, ny, body_offset);
    let alpha_np_per_step = build_padded_alpha_field_refined(
        prepared,
        config.frequencies_hz[0],
        nx,
        ny,
        body_offset,
        refinement,
    );

    let cmax = cmax_body.max(SOUND_SPEED_WATER_SIM);
    // CFL stability: λ_CFL = c·dt/dx ≤ 1/√2 (von Neumann, Fornberg 1988, §4).
    let dt_s = 0.35 * dx / (std::f64::consts::SQRT_2 * cmax);

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

    // ── Eikonal-based focal-law (full aberration correction) ───────────
    //
    // Solve `|∇T(x)| = 1 / c(x)` on the padded grid from the focus point;
    // `T(x)` is the true first-arrival travel-time from focus to each
    // cell through the heterogeneous medium, INCLUDING refraction at
    // bone / soft-tissue interfaces and the water / body skin step.
    //
    // For each element `e_n`, the focal-law delay
    //   `τ_n = T_max − T(e_n)`
    // ensures the wavefront from every element arrives at the focus
    // simultaneously, exactly compensating for the medium-induced phase
    // error.  This is the high-frequency asymptotic of the wave equation
    // in inhomogeneous media (Sethian 1999, §10) and is the standard
    // closed-form aberration correction used in clinical CT-guided FUS
    // (e.g. ExAblate, Insightec — though those run the Eikonal in 3-D).
    //
    // The Eikonal field is computed on `speed_baseline`, the same speed
    // map the FDTD propagator sees (bone speeds intact); refraction is
    // implicit in the Godunov upwind update.  We do NOT clamp bone here:
    // Eikonal handles refraction correctly.
    let focus_cell = focus_padded_cell(focus, nx, ny, dx);
    let travel_time_field =
        crate::therapy::theranostic_guidance::waveform::eikonal::eikonal_travel_time(
            &speed_baseline,
            dx,
            focus_cell,
        );
    let travel_times_s: Vec<f64> = layout
        .therapy_elements
        .iter()
        .map(|p| sample_travel_time(*p, &travel_time_field, nx, ny, dx))
        .collect();
    let max_travel_time_s = travel_times_s
        .iter()
        .copied()
        .filter(|t| t.is_finite())
        .fold(0.0_f64, f64::max);
    let source_delays_s: Vec<f64> = travel_times_s
        .iter()
        .map(|t| {
            if t.is_finite() {
                max_travel_time_s - t
            } else {
                0.0
            }
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
    let source_scale = config.source_pressure_pa as f32 / (source_cells.len().max(1) as f32).sqrt();
    let source_frequency_hz = config.frequencies_hz[0];

    let receiver_cells = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .map(|point| point_to_padded_cell(*point, nx, ny, dx))
        .collect();

    let travel_time_s =
        acoustic_recording_window_s(layout, body_half_x, body_half_y, &source_delays_s);
    let pulse_time_s = 5.0 / frequency_hz;
    let time_steps = (((travel_time_s + pulse_time_s) / dt_s).ceil() as usize).max(96);

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
        source_waveform: None,
    };

    PaddedSimulation {
        grid,
        speed_baseline,
        speed_true,
        body_offset,
        body_dims: (nx_b, ny_b),
        body_dims_coarse: (nx_b_coarse, ny_b_coarse),
        refinement,
    }
}

fn acoustic_recording_window_s(
    layout: &super::super::geometry::DeviceLayout,
    body_half_x: f64,
    body_half_y: f64,
    source_delays_s: &[f64],
) -> f64 {
    let body_corners = [
        Point2 {
            x_m: -body_half_x,
            y_m: -body_half_y,
        },
        Point2 {
            x_m: -body_half_x,
            y_m: body_half_y,
        },
        Point2 {
            x_m: body_half_x,
            y_m: -body_half_y,
        },
        Point2 {
            x_m: body_half_x,
            y_m: body_half_y,
        },
    ];
    let receivers = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .copied()
        .collect::<Vec<_>>();
    let mut max_distance_m = 0.0_f64;
    for source in &layout.therapy_elements {
        for corner in body_corners {
            max_distance_m = max_distance_m.max(point_distance(*source, corner));
            for receiver in &receivers {
                max_distance_m = max_distance_m
                    .max(point_distance(*source, corner) + point_distance(corner, *receiver));
            }
        }
        for receiver in &receivers {
            max_distance_m = max_distance_m.max(point_distance(*source, *receiver));
        }
    }
    let max_source_delay_s = source_delays_s
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max);
    max_source_delay_s + max_distance_m / SOUND_SPEED_WATER_SIM.max(1.0)
}

fn point_distance(lhs: Point2, rhs: Point2) -> f64 {
    (lhs.x_m - rhs.x_m).hypot(lhs.y_m - rhs.y_m)
}

/// Build a padded simulation for a **passive cavitation emission**.
///
/// Unlike [`acoustic_grid`] (a focused transmit from the aperture with
/// Eikonal focal-law delays), this places acoustic sources at the cavitation
/// cloud cells (`emission_points`, in body-centred metres) emitting
/// *simultaneously* — zero delay, a passive omnidirectional emission — at the
/// `emission_frequency_hz` (e.g. the subharmonic f₀/2 or ultraharmonic 3f₀/2).
/// Receivers are the imaging aperture (`layout.imaging_receivers`). The medium
/// is the prepared heterogeneous slice (no lesion sound-speed contrast: the
/// emission is what is being imaged, not a scattering perturbation), so
/// `speed_baseline == speed_true`.
///
/// The grid refinement is sized for the highest cavitation line
/// (`CAV_MAX_LINE_MULTIPLE · f₀ = 3f₀/2`) so the 4th-order FD stencil keeps ≥ 4
/// points per wavelength across the whole emission spectrum, and the recorded
/// window spans the full cavitation burst plus the emission→receiver travel time.
pub(super) fn passive_emission_grid(
    prepared: &PreparedTheranosticSlice,
    layout: &super::super::geometry::DeviceLayout,
    config: &TheranosticInverseConfig,
    emission_points: &[Point2],
    fundamental_hz: f64,
) -> PaddedSimulation {
    // Size the FDTD for the highest spectral line so every cavitation band is
    // numerically resolved.
    let max_line_hz = CAV_MAX_LINE_MULTIPLE * fundamental_hz;
    let (nx_b_coarse, ny_b_coarse) = prepared.sound_speed_m_s.dim();
    let dx_coarse = prepared.spacing_m;
    let medium_speed = &prepared.sound_speed_m_s;
    let cmax_body = medium_speed
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(SOUND_SPEED_WATER_SIM);
    let cmin_body = medium_speed
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .min(SOUND_SPEED_WATER_SIM);

    let refinement = required_refinement(dx_coarse, cmin_body, max_line_hz);
    let dx = dx_coarse / refinement as f64;
    let nx_b = nx_b_coarse * refinement;
    let ny_b = ny_b_coarse * refinement;
    let speed_refined = upsample_field(medium_speed, refinement);

    let (aperture_half_x, aperture_half_y) = layout
        .imaging_receivers
        .iter()
        .chain(emission_points.iter())
        .fold((0.0_f64, 0.0_f64), |(hx, hy), p| {
            (hx.max(p.x_m.abs()), hy.max(p.y_m.abs()))
        });
    let body_half_x = (nx_b.saturating_sub(1) as f64) * 0.5 * dx;
    let body_half_y = (ny_b.saturating_sub(1) as f64) * 0.5 * dx;
    let lambda_water = SOUND_SPEED_WATER_SIM / max_line_hz;
    let margin = lambda_water + PML_CELLS as f64 * dx;
    let half_x_needed = body_half_x.max(aperture_half_x) + margin;
    let half_y_needed = body_half_y.max(aperture_half_y) + margin;
    let nx = padded_dim(nx_b, half_x_needed, dx);
    let ny = padded_dim(ny_b, half_y_needed, dx);
    let body_offset = ((nx - nx_b) / 2, (ny - ny_b) / 2);
    let speed = embed_with_water(&speed_refined, nx, ny, body_offset);
    // Attenuation scales with the fundamental drive frequency (the dominant
    // spectral component of the emission).
    let alpha_np_per_step =
        build_padded_alpha_field_refined(prepared, fundamental_hz, nx, ny, body_offset, refinement);

    let cmax = cmax_body;
    let dt_s = 0.35 * dx / (std::f64::consts::SQRT_2 * cmax);
    let domain_half_x = (nx.saturating_sub(1) as f64) * 0.5 * dx;
    let domain_half_y = (ny.saturating_sub(1) as f64) * 0.5 * dx;
    let domain_extent = domain_half_x.hypot(domain_half_y);
    // One-way emission → receiver path plus the full cavitation burst duration;
    // the round-trip bound is a conservative upper bound that guarantees the
    // entire burst reaches every receiver within the recorded window.
    let travel_time_s = 2.0 * domain_extent / SOUND_SPEED_WATER_SIM.max(1.0);
    let pulse_time_s = cavitation_burst_duration_s(fundamental_hz);
    let time_steps = (((travel_time_s + pulse_time_s) / dt_s).ceil() as usize).max(96);

    // Cavitation cloud sources fire simultaneously (passive emission → no
    // electronic steering delay). Deduplicate cells that round together.
    let source_cells: Vec<usize> = {
        let mut seen = std::collections::HashSet::new();
        emission_points
            .iter()
            .map(|p| point_to_padded_cell(*p, nx, ny, dx))
            .filter(|cell| seen.insert(*cell))
            .collect()
    };
    let source_delays_s = vec![0.0_f64; source_cells.len()];
    let source_scale = config.source_pressure_pa as f32 / (source_cells.len().max(1) as f32).sqrt();
    // Broadband bubble-cloud emission: subharmonic, fundamental, ultraharmonic.
    let source_waveform = Some(cavitation_emission_waveform(
        time_steps,
        dt_s,
        fundamental_hz,
    ));

    let receiver_cells = layout
        .imaging_receivers
        .iter()
        .map(|point| point_to_padded_cell(*point, nx, ny, dx))
        .collect();

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
        source_frequency_hz: fundamental_hz,
        source_scale,
        source_waveform,
    };

    PaddedSimulation {
        grid,
        speed_baseline: speed.clone(),
        speed_true: speed,
        body_offset,
        body_dims: (nx_b, ny_b),
        body_dims_coarse: (nx_b_coarse, ny_b_coarse),
        refinement,
    }
}

/// Refinement factor chosen so the FDTD sees ≥ 4 points per wavelength at the
/// configured transmit frequency, using the slower of body / water speeds.
/// Capped at 16 to bound memory and runtime.
fn required_refinement(dx_coarse: f64, c_min: f64, frequency_hz: f64) -> usize {
    const TARGET_POINTS_PER_WAVELENGTH: f64 = 4.0;
    const MAX_REFINEMENT: usize = 16;
    if !dx_coarse.is_finite() || dx_coarse <= 0.0 || frequency_hz <= 0.0 {
        return 1;
    }
    let lambda = c_min.max(1.0) / frequency_hz;
    let dx_target = lambda / TARGET_POINTS_PER_WAVELENGTH;
    if dx_target >= dx_coarse {
        return 1;
    }
    let r = (dx_coarse / dx_target).ceil() as usize;
    r.clamp(1, MAX_REFINEMENT)
}

/// Nearest-neighbour upsample: each coarse cell `[ix, iy]` is replicated
/// across the `R × R` refined cells `[ix·R + di, iy·R + dj]` for
/// `di, dj ∈ [0, R)`.  Preserves the embedded body fields exactly at the
/// cell-replication boundaries and avoids smoothing discontinuities like
/// the body / water interface or bone / soft-tissue boundaries.
fn upsample_field(coarse: &Array2<f64>, refinement: usize) -> Array2<f64> {
    if refinement <= 1 {
        return coarse.clone();
    }
    let (nx_c, ny_c) = coarse.dim();
    let nx_r = nx_c * refinement;
    let ny_r = ny_c * refinement;
    Array2::from_shape_fn((nx_r, ny_r), |(ix, iy)| {
        coarse[[ix / refinement, iy / refinement]]
    })
}

/// Per-cell amplitude decay on the refined padded grid.  The body attenuation
/// is nearest-neighbour upsampled to the refined cells; coupling water cells
/// (outside the embedded body sub-region) are lossless.
fn build_padded_alpha_field_refined(
    prepared: &PreparedTheranosticSlice,
    frequency_hz: f64,
    nx: usize,
    ny: usize,
    offset: (usize, usize),
    refinement: usize,
) -> Vec<f32> {
    let f0_mhz = frequency_hz * 1.0e-6;
    let (nx_b_c, ny_b_c) = prepared.attenuation_np_per_m_mhz.dim();
    let nx_b = nx_b_c * refinement;
    let ny_b = ny_b_c * refinement;
    let dx = prepared.spacing_m / refinement as f64;
    let (ox, oy) = offset;
    let mut out = vec![0.0_f32; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            if ix >= ox && iy >= oy && ix < ox + nx_b && iy < oy + ny_b {
                let bx = (ix - ox) / refinement;
                let by = (iy - oy) / refinement;
                let alpha_np_per_m = prepared.attenuation_np_per_m_mhz[[bx, by]] * f0_mhz;
                out[linear(ix, iy, ny)] = (alpha_np_per_m * dx) as f32;
            }
        }
    }
    out
}

/// Convert a physical point `p` (metres, padded-grid coordinates) into the
/// nearest padded-grid cell index, clamped to the interior so the Eikonal
/// solver can place its source there without index-out-of-bounds.
fn focus_padded_cell(p: Point2, nx: usize, ny: usize, dx: f64) -> (usize, usize) {
    let cx = 0.5 * (nx as f64 - 1.0);
    let cy = 0.5 * (ny as f64 - 1.0);
    let ix = (p.x_m / dx + cx).round().clamp(0.0, (nx - 1) as f64) as usize;
    let iy = (p.y_m / dx + cy).round().clamp(0.0, (ny - 1) as f64) as usize;
    (ix, iy)
}

/// Sample the Eikonal travel-time field at the padded-grid cell nearest
/// to physical point `p`.  Returns `+∞` for points outside the padded
/// grid (the aperture margin is sized to contain every clinical element,
/// so this fallback is conservative rather than expected).
fn sample_travel_time(p: Point2, t_field: &Array2<f64>, nx: usize, ny: usize, dx: f64) -> f64 {
    let cx = 0.5 * (nx as f64 - 1.0);
    let cy = 0.5 * (ny as f64 - 1.0);
    let ix_f = p.x_m / dx + cx;
    let iy_f = p.y_m / dx + cy;
    if !ix_f.is_finite() || !iy_f.is_finite() {
        return f64::INFINITY;
    }
    let ix = ix_f.round();
    let iy = iy_f.round();
    if ix < 0.0 || iy < 0.0 {
        return f64::INFINITY;
    }
    let ix = ix as usize;
    let iy = iy as usize;
    if ix >= nx || iy >= ny {
        return f64::INFINITY;
    }
    t_field[[ix, iy]]
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

/// Map a physical point to a padded-grid `(ix, iy)` cell using the same centring
/// convention as the body-grid `point_to_cell`. Clamp only to the FD-stencil
/// halo bounds `[2, n-3]`; the padded sizing guarantees no element actually
/// hits the clamp.
pub(super) fn point_to_padded_cell_2d(
    point: Point2,
    nx: usize,
    ny: usize,
    spacing_m: f64,
) -> (usize, usize) {
    let cx = 0.5 * (nx - 1) as f64;
    let cy = 0.5 * (ny - 1) as f64;
    let ix = (point.x_m / spacing_m + cx)
        .round()
        .clamp(2.0, (nx - 3) as f64) as usize;
    let iy = (point.y_m / spacing_m + cy)
        .round()
        .clamp(2.0, (ny - 3) as f64) as usize;
    (ix, iy)
}

/// Linear padded-grid cell index for a physical point (see
/// [`point_to_padded_cell_2d`]).
fn point_to_padded_cell(point: Point2, nx: usize, ny: usize, spacing_m: f64) -> usize {
    let (ix, iy) = point_to_padded_cell_2d(point, nx, ny, spacing_m);
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
#[allow(dead_code)]
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
                let alpha_np_per_m = prepared.attenuation_np_per_m_mhz[[ix - ox, iy - oy]] * f0_mhz;
                out[linear(ix, iy, ny)] = (alpha_np_per_m * dx) as f32;
            }
        }
    }
    out
}
