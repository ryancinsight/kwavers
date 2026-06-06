use super::*;
use crate::therapy::theranostic_guidance::config::AnatomyKind;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
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
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

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
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

    let n = 31usize;
    let spacing = 0.0008_f64;
    let prepared = water_fixture(n, spacing, SOUND_SPEED_WATER_SIM);
    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.frequencies_hz = vec![500_000.0];

    // Three distinct receivers plus a deliberate duplicate of the first.
    let r0 = Point2 {
        x_m: 0.018,
        y_m: 0.0,
    };
    let r1 = Point2 {
        x_m: 0.0,
        y_m: 0.018,
    };
    let r2 = Point2 {
        x_m: -0.018,
        y_m: 0.0,
    };
    let layout = DeviceLayout {
        therapy_elements: Vec::new(),
        imaging_receivers: vec![r0, r1, r2, r0],
        focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
        skin_contact_m: Point2 {
            x_m: -0.018,
            y_m: 0.0,
        },
        model_name: "dedup_test".to_owned(),
    };
    let source = Point2 {
        x_m: 2.0 * spacing,
        y_m: 0.0,
    };
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
            delays[[g, 0]],
            delays[[g, 3]],
            "coincident receivers must share an identical delay column"
        );
    }
    assert!(
        delays.iter().all(|d| d.is_finite() && *d >= 0.0),
        "all eikonal delays must be finite and non-negative"
    );
}
