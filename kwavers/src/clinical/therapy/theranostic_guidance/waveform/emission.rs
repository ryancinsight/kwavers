//! Genuine passive acoustic mapping (PAM) of a cavitation emission.
//!
//! # Physics
//!
//! During therapy, inertial/stable cavitation at the focus radiates broadband
//! acoustic energy whose subharmonic (f₀/2) and ultraharmonic (3f₀/2) bands are
//! the canonical markers of bubble activity. The therapy array doubles as a
//! passive receive aperture: it records the emission and a passive beamformer
//! localizes the cavitation. This module simulates that forward emission through
//! the heterogeneous CT-derived medium with the same 4th-order-FD / CPML solver
//! used elsewhere in this crate, then beamforms the per-receiver traces with the
//! delay-multiply-and-sum (DMAS) passive imaging condition.
//!
//! # Pipeline
//!
//! 1. [`passive_emission_grid`](super::grid) — sources at the cavitation cloud
//!    cells emitting *simultaneously* (passive, zero delay) a broadband
//!    bubble-cloud spectrum (subharmonic f₀/2, fundamental f₀, ultraharmonic
//!    3f₀/2; see [`super::cavitation`]); receivers at the imaging aperture;
//!    refinement sized for the highest line.
//! 2. [`propagate`](super::forward) — records per-receiver time traces of the
//!    full broadband emission (one forward solve serves every band).
//! 3. Aberration-corrected delays: `eikonal_delay_matrix` computes the
//!    first-arrival travel time from every active-grid pixel to every receiver
//!    through the *heterogeneous* medium (one eikonal solve per receiver, by
//!    reciprocity), keeping the coherent sum aligned through skull/rib/water
//!    speed contrasts where a constant-speed model loses coherence.
//! 4. Spectral PAM: [`DelayAndSumPAM::beamform_signals_with_delays`]
//!    delay-and-sums the *broadband* traces to a per-pixel time series using
//!    those delays (full bandwidth → fine range resolution), then per requested
//!    band a zero-phase Gaussian band-pass of each pixel's series yields the band
//!    energy — localizing the cavitation while remaining specific to the
//!    subharmonic / ultraharmonic line. (Band-passing the raw traces first would
//!    collapse the bandwidth to a single line and ruin range resolution.)
//!
//! # References
//! - Gyöngy, M., Coussios, C.-C. (2010). *Passive spatial mapping of inertial
//!   cavitation during HIFU exposure.* IEEE TBME 57(1), 48–56.
//! - Haworth, K.J. et al. (2012). *Passive imaging with pulsed ultrasound
//!   insonations.* J. Acoust. Soc. Am. 132(1), 544–553 — spectral PAM.
//! - Sukovich, J.R. et al. (2020). *Real-time transcranial histotripsy treatment
//!   localization … using acoustic cavitation emission feedback.* (ACE mapping
//!   localizes cavitation to < 1.5 mm.)

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::super::config::TheranosticInverseConfig;
use super::super::geometry::{DeviceLayout, Point2};
use super::super::medium::PreparedTheranosticSlice;
use super::eikonal::eikonal_travel_time;
use super::forward::propagate;
use super::grid::{passive_emission_grid, point_to_padded_cell_2d};
use super::types::PaddedSimulation;
use crate::analysis::signal_processing::pam::{ApodizationType, DelayAndSumConfig, DelayAndSumPAM};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::apply_spectral_response_1d;

/// Minimum number of passive receivers required for 2-D PAM localization.
const MIN_PAM_RECEIVERS: usize = 3;

/// Band-pass half-bandwidth as a fraction of the fundamental, `f₀/8`. Wide
/// enough to pass a cavitation line (bandwidth ≈ `0.064 f₀`) yet narrow enough
/// to reject the fundamental, which sits `f₀/2 = 4` bandwidths from each band.
const BAND_BANDWIDTH_FRACTION: f64 = 0.125;

/// Simulate one broadband cavitation emission and beamform the recorded receiver
/// traces into a passive-acoustic-mapping intensity for each requested band
/// centre.
///
/// All coordinates are body-centred metres (the convention shared by
/// `active_grid` and `DeviceLayout`). `fundamental_hz` is the therapy drive
/// frequency `f₀`; the emission carries the f₀/2, f₀ and 3f₀/2 lines.
/// `band_centers_hz` lists the bands to image (e.g. `[f₀/2, 3f₀/2]`).
/// `pam_sound_speed_m_s` is the homogeneous steering speed for the delay model
/// (the conventional PAM assumption); the emission propagates through the full
/// heterogeneous medium. A single forward solve serves all bands.
///
/// Returns one intensity vector per band (in `band_centers_hz` order); each is
/// indexed in the same order as `grid_points`.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] if there are fewer than three
/// receivers, no emission points, no grid points, no bands, a non-positive sound
/// speed, or a non-positive fundamental frequency.
pub fn passive_acoustic_maps(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
    grid_points: &[Point2],
    emission_points: &[Point2],
    fundamental_hz: f64,
    band_centers_hz: &[f64],
    pam_sound_speed_m_s: f64,
) -> KwaversResult<Vec<Vec<f64>>> {
    if layout.imaging_receivers.len() < MIN_PAM_RECEIVERS {
        return Err(KwaversError::InvalidInput(format!(
            "passive acoustic mapping needs at least {MIN_PAM_RECEIVERS} receivers, got {}",
            layout.imaging_receivers.len()
        )));
    }
    if emission_points.is_empty() {
        return Err(KwaversError::InvalidInput(
            "passive acoustic mapping requires at least one cavitation emission point".to_owned(),
        ));
    }
    if grid_points.is_empty() {
        return Err(KwaversError::InvalidInput(
            "passive acoustic mapping requires at least one reconstruction grid point".to_owned(),
        ));
    }
    if band_centers_hz.is_empty() {
        return Err(KwaversError::InvalidInput(
            "passive acoustic mapping requires at least one band centre".to_owned(),
        ));
    }
    if !pam_sound_speed_m_s.is_finite() || pam_sound_speed_m_s <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "passive acoustic mapping sound speed must be positive and finite".to_owned(),
        ));
    }
    if !fundamental_hz.is_finite() || fundamental_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "passive acoustic mapping fundamental frequency must be positive and finite".to_owned(),
        ));
    }

    let sim = passive_emission_grid(prepared, layout, config, emission_points, fundamental_hz);
    let run = propagate(&sim.grid, &sim.speed_baseline, false);

    let n_receivers = sim.grid.receiver_cells.len();
    let n_samples = sim.grid.time_steps;
    let dt_s = sim.grid.dt_s;
    // Flattened layout: traces[step * n_receivers + receiver].
    let passive_data = Array2::from_shape_fn((n_receivers, n_samples), |(receiver, step)| {
        f64::from(run.traces[step * n_receivers + receiver])
    });

    let sensors: Vec<[f64; 3]> = layout
        .imaging_receivers
        .iter()
        .map(|p| [p.x_m, p.y_m, 0.0])
        .collect();

    let das_config = DelayAndSumConfig {
        sound_speed: pam_sound_speed_m_s,
        sampling_frequency: 1.0 / dt_s,
        window_size: n_samples,
        apodization: ApodizationType::Uniform,
        ..DelayAndSumConfig::default()
    };
    let pam = DelayAndSumPAM::new(sensors, das_config)?;

    // Aberration-corrected receive: rather than the homogeneous straight-line
    // delay model, use eikonal first-arrival travel times through the actual
    // heterogeneous medium (by reciprocity, solved once per receiver). This
    // keeps the coherent sum aligned through speed contrasts (skull, ribs,
    // water/tissue interfaces) where the constant-speed model loses coherence —
    // the limiting factor for the higher-frequency cavitation bands.
    let grid_cells: Vec<(usize, usize)> = grid_points
        .iter()
        .map(|p| point_to_padded_cell_2d(*p, sim.grid.nx, sim.grid.ny, sim.grid.dx_m))
        .collect();
    let delays = eikonal_delay_matrix(&sim, &grid_cells);

    // Spectral PAM: beamform the *broadband* emission (full f₀/2…3f₀/2 band →
    // fine range resolution), then attribute each pixel's energy to a cavitation
    // band by a zero-phase Gaussian band-pass of its beamformed time series.
    // This preserves broadband spatial resolution while remaining band-specific,
    // unlike band-passing the raw traces first, which would collapse the
    // bandwidth to a single line and destroy range resolution (Gyöngy &
    // Coussios 2010; Haworth et al. 2012).
    let beamformed = pam.beamform_signals_with_delays(passive_data.view(), delays.view())?;
    let fs = 1.0 / dt_s;
    let bandwidth_hz = BAND_BANDWIDTH_FRACTION * fundamental_hz;
    let maps = band_centers_hz
        .iter()
        .map(|&center_hz| band_power_per_point(&beamformed, fs, center_hz, bandwidth_hz))
        .collect();
    Ok(maps)
}

/// Aberration-corrected propagation-delay matrix `[n_grid × n_receivers]` (in
/// samples) via eikonal first-arrival travel times through the heterogeneous
/// medium.
///
/// ## Reciprocity
///
/// The travel time `T(receiver → pixel)` equals `T(pixel → receiver)`, so a
/// single eikonal solve per source cell yields that source's delay to every
/// candidate grid point.
///
/// ## Exact dedup (the only accuracy-preserving sparsity here)
///
/// The eikonal is solved on the **refined** padded grid — the resolution at
/// which the aberration correction was validated (PAM-5). The redundancy removed
/// is exact: receivers mapping to the *same* refined source cell share one
/// solve.
///
/// Two grid-coarsening optimizations were implemented and **rejected** by a
/// convergence test (since removed) against this refined solve: a stride-sampled
/// coarsen and a slowness-(harmonic-)averaged coarsen *both* produced a 13.6-
/// sample per-receiver delay error at the ultraharmonic band (> 2 periods),
/// far above the ⅛-period coherence bound (6.3 samples). The two methods give
/// the identical error because the refined speed map is a nearest-neighbour
/// upsample of the body map (each refined block is uniform, so any block average
/// equals the stride sample); the real limit is the first-order Godunov
/// truncation of the eikonal solve at body resolution, which the refined grid's
/// finer `dx` genuinely resolves near the source and along refracted rays.
/// Full refined resolution is therefore required for the high-frequency
/// correction; grid coarsening is not used.
///
/// Reference: Zhao (2005), fast sweeping method — see [`super::eikonal`].
fn eikonal_delay_matrix(sim: &PaddedSimulation, grid_cells: &[(usize, usize)]) -> Array2<f64> {
    let ny = sim.grid.ny;
    let dx = sim.grid.dx_m;
    let dt = sim.grid.dt_s;
    let n_receivers = sim.grid.receiver_cells.len();

    // Exact dedup: solve once per unique refined source cell.
    let mut unique_cells: Vec<usize> = sim.grid.receiver_cells.clone();
    unique_cells.sort_unstable();
    unique_cells.dedup();

    let solved: HashMap<usize, Vec<f64>> = unique_cells
        .par_iter()
        .map(|&cell| {
            let travel_time = eikonal_travel_time(&sim.speed_baseline, dx, (cell / ny, cell % ny));
            let column: Vec<f64> = grid_cells
                .iter()
                .map(|&(gix, giy)| travel_time[[gix, giy]] / dt)
                .collect();
            (cell, column)
        })
        .collect();

    let mut delays = Array2::<f64>::zeros((grid_cells.len(), n_receivers));
    for (receiver, &cell) in sim.grid.receiver_cells.iter().enumerate() {
        let column = &solved[&cell];
        for (grid_idx, &delay) in column.iter().enumerate() {
            delays[[grid_idx, receiver]] = delay;
        }
    }
    delays
}

/// Per-point energy of a beamformed-signal matrix in a zero-phase Gaussian band
/// about `center_hz` (standard deviation `bandwidth_hz`).
///
/// `beamformed` is `[n_points × window]` (one delay-and-sum time series per grid
/// point). For each point the band-passed signal energy `Σ_t |F·s|²` is the
/// band intensity at that pixel; the band-pass `exp(-½((f−f_c)/Δf)²)` is applied
/// as a real, frequency-folded spectral multiplier (zero phase).
#[must_use]
fn band_power_per_point(
    beamformed: &Array2<f64>,
    fs: f64,
    center_hz: f64,
    bandwidth_hz: f64,
) -> Vec<f64> {
    beamformed
        .outer_iter()
        .map(|row| {
            let series: Array1<f64> = row.to_owned();
            let filtered = apply_spectral_response_1d(&series, fs, |_, freq, nyquist| {
                let f_eff = freq.min(2.0 * nyquist - freq).max(0.0);
                let z = (f_eff - center_hz) / bandwidth_hz;
                (-0.5 * z * z).exp()
            });
            filtered.iter().map(|&value| value * value).sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::therapy::theranostic_guidance::config::AnatomyKind;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use ndarray::Array2;

    /// A point cavitation source in a homogeneous water domain must localize to
    /// within one wavelength of its true position under the genuine PAM
    /// pipeline (broadband emission → receiver traces → subharmonic band-pass →
    /// DMAS beamform).
    #[test]
    fn point_cavitation_source_localizes_within_one_wavelength() {
        let n = 41usize;
        let spacing = 0.0006_f64; // 0.6 mm body grid
        let speed = SOUND_SPEED_WATER_SIM;
        let prepared = water_fixture(n, spacing, speed);

        let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
        config.frequencies_hz = vec![500_000.0];
        config.source_pressure_pa = 1.0e6;
        let fundamental_hz = 500_000.0;
        let subharmonic_hz = 0.5 * fundamental_hz; // f0/2 = 250 kHz band

        // A ring of receivers surrounding the domain (body-centred metres).
        let receiver_radius_m = 0.020;
        let n_receivers = 24;
        let imaging_receivers: Vec<Point2> = (0..n_receivers)
            .map(|i| {
                let theta = std::f64::consts::TAU * i as f64 / n_receivers as f64;
                Point2 {
                    x_m: receiver_radius_m * theta.cos(),
                    y_m: receiver_radius_m * theta.sin(),
                }
            })
            .collect();
        let layout = DeviceLayout {
            therapy_elements: Vec::new(),
            imaging_receivers,
            focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
            skin_contact_m: Point2 {
                x_m: -receiver_radius_m,
                y_m: 0.0,
            },
            model_name: "pam_localization_test".to_owned(),
        };

        // True cavitation source offset from the centre.
        let source = Point2 {
            x_m: 0.0036,
            y_m: -0.0024,
        };

        // Reconstruction grid covering the interior (body-centred).
        let half = 12i32;
        let mut grid_points = Vec::new();
        for ix in -half..=half {
            for iy in -half..=half {
                grid_points.push(Point2 {
                    x_m: ix as f64 * spacing,
                    y_m: iy as f64 * spacing,
                });
            }
        }

        let maps = passive_acoustic_maps(
            &prepared,
            &layout,
            &config,
            &grid_points,
            &[source],
            fundamental_hz,
            &[subharmonic_hz],
            speed,
        )
        .expect("PAM maps");
        let intensity = &maps[0];

        // Argmax pixel.
        let (best_idx, &peak) = intensity
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("non-empty");
        assert!(peak > 0.0, "PAM peak intensity must be positive");
        let localized = grid_points[best_idx];
        let offset_m = (localized.x_m - source.x_m).hypot(localized.y_m - source.y_m);
        let wavelength_m = speed / subharmonic_hz;
        assert!(
            offset_m <= wavelength_m,
            "PAM must localize the cavitation source within one wavelength: \
             offset={offset_m:.4} m, lambda={wavelength_m:.4} m, \
             localized=({:.4},{:.4}), true=({:.4},{:.4})",
            localized.x_m,
            localized.y_m,
            source.x_m,
            source.y_m
        );
    }

    fn water_fixture(n: usize, spacing: f64, speed: f64) -> PreparedTheranosticSlice {
        fixture_with_speed(Array2::from_elem((n, n), speed), spacing)
    }

    fn fixture_with_speed(speed: Array2<f64>, spacing: f64) -> PreparedTheranosticSlice {
        let (nx, ny) = speed.dim();
        let mut target_mask = Array2::from_elem((nx, ny), false);
        target_mask[[nx / 2, ny / 2]] = true;
        PreparedTheranosticSlice {
            anatomy: AnatomyKind::Liver,
            ct_hu: Array2::from_elem((nx, ny), 0.0),
            label: Array2::zeros((nx, ny)),
            sound_speed_m_s: speed,
            attenuation_np_per_m_mhz: Array2::from_elem((nx, ny), 0.0),
            body_mask: Array2::from_elem((nx, ny), true),
            organ_mask: Array2::from_elem((nx, ny), true),
            target_mask,
            spacing_m: spacing,
            source_slice_index: 0,
            source_dimensions: [nx, ny],
            source_spacing_m: [spacing, spacing],
            crop_bounds_index: [0, nx - 1, 0, ny - 1],
        }
    }

    /// Aberration-correction integration test. A high-speed (skull-like) slab
    /// sits between the cavitation source and the +x receivers. The eikonal
    /// delay matrix (a) genuinely differs from the homogeneous straight-line
    /// model along slab-crossing paths, and (b) still localizes the source
    /// within ~1.5 wavelengths — the property the homogeneous model loses
    /// through strong speed contrasts.
    #[test]
    fn eikonal_delays_account_for_aberration_and_localize() {
        use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

        let n = 41usize;
        let spacing = 0.0006_f64;
        let water = SOUND_SPEED_WATER_SIM;
        let slab = 2800.0_f64; // skull-like high-speed aberrator

        let mut speed = Array2::from_elem((n, n), water);
        for ix in 26..33 {
            for iy in 0..n {
                speed[[ix, iy]] = slab;
            }
        }
        let prepared = fixture_with_speed(speed, spacing);

        let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
        config.frequencies_hz = vec![500_000.0];
        config.source_pressure_pa = 1.0e6;
        let fundamental_hz = 500_000.0;
        let subharmonic_hz = 0.5 * fundamental_hz;

        let receiver_radius_m = 0.020;
        let n_receivers = 24;
        let imaging_receivers: Vec<Point2> = (0..n_receivers)
            .map(|i| {
                let theta = std::f64::consts::TAU * i as f64 / n_receivers as f64;
                Point2 {
                    x_m: receiver_radius_m * theta.cos(),
                    y_m: receiver_radius_m * theta.sin(),
                }
            })
            .collect();
        let layout = DeviceLayout {
            therapy_elements: Vec::new(),
            imaging_receivers,
            focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
            skin_contact_m: Point2 {
                x_m: -receiver_radius_m,
                y_m: 0.0,
            },
            model_name: "aberration_test".to_owned(),
        };

        // Source on the lattice (so it is exactly a grid point).
        let source = Point2 {
            x_m: 3.0 * spacing,
            y_m: 0.0,
        };
        let half = 12i32;
        let mut grid_points = Vec::new();
        for ix in -half..=half {
            for iy in -half..=half {
                grid_points.push(Point2 {
                    x_m: ix as f64 * spacing,
                    y_m: iy as f64 * spacing,
                });
            }
        }

        // Replicate the acquisition, then beamform with eikonal delays.
        let sim = passive_emission_grid(&prepared, &layout, &config, &[source], fundamental_hz);
        let run = propagate(&sim.grid, &sim.speed_baseline, false);
        let n_recv = sim.grid.receiver_cells.len();
        let n_samp = sim.grid.time_steps;
        let passive_data = Array2::from_shape_fn((n_recv, n_samp), |(r, s)| {
            f64::from(run.traces[s * n_recv + r])
        });
        let sensors: Vec<[f64; 3]> = layout
            .imaging_receivers
            .iter()
            .map(|p| [p.x_m, p.y_m, 0.0])
            .collect();
        let das_config = DelayAndSumConfig {
            sound_speed: water,
            sampling_frequency: 1.0 / sim.grid.dt_s,
            window_size: n_samp,
            apodization: ApodizationType::Uniform,
            ..DelayAndSumConfig::default()
        };
        let pam = DelayAndSumPAM::new(sensors, das_config).expect("pam");
        let grid_cells: Vec<(usize, usize)> = grid_points
            .iter()
            .map(|p| point_to_padded_cell_2d(*p, sim.grid.nx, sim.grid.ny, sim.grid.dx_m))
            .collect();
        let delays = eikonal_delay_matrix(&sim, &grid_cells);
        let beamformed = pam
            .beamform_signals_with_delays(passive_data.view(), delays.view())
            .expect("beamform");
        let map = band_power_per_point(
            &beamformed,
            1.0 / sim.grid.dt_s,
            subharmonic_hz,
            BAND_BANDWIDTH_FRACTION * fundamental_hz,
        );

        // (a) Localization within ~1.5 λ.
        let (best_idx, _) = map
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("non-empty");
        let localized = grid_points[best_idx];
        let offset_m = (localized.x_m - source.x_m).hypot(localized.y_m - source.y_m);
        let wavelength_m = water / subharmonic_hz;
        assert!(
            offset_m <= 1.5 * wavelength_m,
            "aberration-corrected PAM must localize within 1.5 λ: offset={offset_m:.4} m, \
             λ={wavelength_m:.4} m"
        );

        // (b) The eikonal delays genuinely differ from the homogeneous
        // straight-line model along slab-crossing paths (so correction is real,
        // not a no-op).
        let source_idx = grid_points
            .iter()
            .position(|p| (p.x_m - source.x_m).abs() < 1e-9 && (p.y_m - source.y_m).abs() < 1e-9)
            .expect("source is a grid point");
        let dt = sim.grid.dt_s;
        let mut max_rel_diff = 0.0_f64;
        for (r, receiver) in layout.imaging_receivers.iter().enumerate() {
            let straight_m = (receiver.x_m - source.x_m).hypot(receiver.y_m - source.y_m);
            let homogeneous_samples = straight_m / water / dt;
            let eikonal_samples = delays[[source_idx, r]];
            let rel = ((eikonal_samples - homogeneous_samples) / homogeneous_samples).abs();
            max_rel_diff = max_rel_diff.max(rel);
        }
        assert!(
            max_rel_diff > 0.02,
            "eikonal delays must differ from the homogeneous model through the aberrator: \
             max relative difference = {max_rel_diff:.4}"
        );
    }

    /// Exact-dedup correctness for the aberration-corrected delay matrix:
    /// receivers that map to the same refined source cell must receive identical
    /// delay columns (the dedup solves each unique cell once and reuses it), and
    /// every delay must be finite and non-negative.
    #[test]
    fn eikonal_delay_matrix_dedups_coincident_receivers() {
        use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

        let n = 31usize;
        let spacing = 0.0008_f64;
        let prepared = water_fixture(n, spacing, SOUND_SPEED_WATER_SIM);
        let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
        config.frequencies_hz = vec![500_000.0];

        // Three distinct receivers plus a deliberate duplicate of the first.
        let r0 = Point2 { x_m: 0.018, y_m: 0.0 };
        let r1 = Point2 { x_m: 0.0, y_m: 0.018 };
        let r2 = Point2 { x_m: -0.018, y_m: 0.0 };
        let layout = DeviceLayout {
            therapy_elements: Vec::new(),
            imaging_receivers: vec![r0, r1, r2, r0],
            focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
            skin_contact_m: Point2 { x_m: -0.018, y_m: 0.0 },
            model_name: "dedup_test".to_owned(),
        };
        let source = Point2 { x_m: 2.0 * spacing, y_m: 0.0 };
        let grid_points: Vec<Point2> = (-8i32..=8)
            .flat_map(|ix| {
                (-8i32..=8).map(move |iy| Point2 {
                    x_m: ix as f64 * spacing,
                    y_m: iy as f64 * spacing,
                })
            })
            .collect();

        let sim = passive_emission_grid(&prepared, &layout, &config, &[source], 500_000.0);
        let grid_cells: Vec<(usize, usize)> = grid_points
            .iter()
            .map(|p| point_to_padded_cell_2d(*p, sim.grid.nx, sim.grid.ny, sim.grid.dx_m))
            .collect();
        let delays = eikonal_delay_matrix(&sim, &grid_cells);

        // r0 (column 0) and its duplicate (column 3) map to the same source
        // cell, so their delay columns are bit-identical.
        for g in 0..grid_cells.len() {
            assert_eq!(
                delays[[g, 0]], delays[[g, 3]],
                "coincident receivers must share an identical delay column"
            );
        }
        assert!(
            delays.iter().all(|d| d.is_finite() && *d >= 0.0),
            "all eikonal delays must be finite and non-negative"
        );
    }
}
