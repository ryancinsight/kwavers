use leto::Array2;

use super::*;
use crate::therapy::theranostic_guidance::config::AnatomyKind;
use crate::therapy::theranostic_guidance::geometry::Point2;
use kwavers_core::constants::fundamental::{SOUND_SPEED_AIR, SOUND_SPEED_TISSUE};

#[test]
fn peak_pressure_exposure_records_bounded_workspace() {
    let prepared = prepared_fixture(Array2::from_elem((28, 28), SOUND_SPEED_TISSUE));
    let config = exposure_config();
    let layout = exposure_layout();

    let result = simulate_peak_pressure_exposure(&prepared, &layout, &config);
    let exposure_peak = result.exposure.iter().copied().fold(0.0, f64::max);
    let raw_peak = result.raw_peak_pressure.iter().copied().fold(0.0, f64::max);

    assert_eq!(result.model_name, THERANOSTIC_WAVE_EXPOSURE_MODEL);
    assert_eq!(result.backend_name, THERANOSTIC_WAVE_EXPOSURE_BACKEND);
    assert!(!result.uses_hybrid_pstd_fdtd);
    assert_eq!(result.source_count, layout.therapy_elements.len());
    // Padded simulation grid encompasses body + aperture + λ_water +
    // PML_CELLS margin on each side; allocated workspace must dominate
    // the bare-body 6·28·28 lower bound and remain a multiple of 6.
    assert!(result.workspace_values >= 6 * 28 * 28);
    assert_eq!(result.workspace_values % 6, 0);
    assert!(result.time_steps >= 96);
    assert!(result.dt_s > 0.0);
    assert!(raw_peak > 0.0);
    assert!(
        (exposure_peak - config.source_pressure_pa).abs() <= config.source_pressure_pa * 1.0e-6,
        "exposure_peak={exposure_peak}, expected={}",
        config.source_pressure_pa
    );
}

#[test]
fn peak_pressure_exposure_responds_to_internal_gas_scattering() {
    let config = exposure_config();
    let layout = exposure_layout();
    let homogeneous = simulate_peak_pressure_exposure(
        &prepared_fixture(Array2::from_elem((28, 28), SOUND_SPEED_TISSUE)),
        &layout,
        &config,
    );

    let mut speed = Array2::from_elem((28, 28), SOUND_SPEED_TISSUE);
    for ix in 12..16 {
        for iy in 8..20 {
            speed[[ix, iy]] = SOUND_SPEED_AIR;
        }
    }
    let scattered = simulate_peak_pressure_exposure(&prepared_fixture(speed), &layout, &config);

    let downstream_difference =
        mean_abs_difference(&homogeneous.exposure, &scattered.exposure, 18..24, 8..20);
    assert!(
            downstream_difference > 1.0e-3,
            "heterogeneous wave solve must alter downstream peak pressure; diff={downstream_difference}"
        );
}

fn prepared_fixture(speed: Array2<f64>) -> PreparedTheranosticSlice {
    let (nx, ny) = speed.dim();
    let mut target_mask = Array2::from_elem((nx, ny), false);
    for ix in 13..15 {
        for iy in 13..15 {
            target_mask[[ix, iy]] = true;
        }
    }
    PreparedTheranosticSlice {
        anatomy: AnatomyKind::Liver,
        ct_hu: Array2::from_elem((nx, ny), 40.0),
        label: Array2::zeros((nx, ny)),
        sound_speed_m_s: speed,
        attenuation_np_per_m_mhz: Array2::from_elem((nx, ny), 0.1),
        body_mask: Array2::from_elem((nx, ny), true),
        organ_mask: Array2::from_elem((nx, ny), true),
        target_mask,
        spacing_m: 0.002,
        source_slice_index: 0,
        source_dimensions: [nx, ny],
        source_spacing_m: [0.002, 0.002],
        crop_bounds_index: [0, nx - 1, 0, ny - 1],
    }
}

fn exposure_config() -> TheranosticInverseConfig {
    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.frequencies_hz = vec![800_000.0];
    config.source_pressure_pa = 10_000.0;
    config
}

fn exposure_layout() -> DeviceLayout {
    let therapy_elements = (0..8)
        .map(|idx| Point2 {
            x_m: -0.022,
            y_m: -0.014 + idx as f64 * 0.004,
        })
        .collect::<Vec<_>>();
    DeviceLayout {
        therapy_elements,
        imaging_receivers: Vec::new(),
        focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
        skin_contact_m: Point2 {
            x_m: -0.026,
            y_m: 0.0,
        },
        model_name: "test_linear_array".to_owned(),
    }
}

fn mean_abs_difference(
    lhs: &Array2<f64>,
    rhs: &Array2<f64>,
    x_range: std::ops::Range<usize>,
    y_range: std::ops::Range<usize>,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for ix in x_range {
        for iy in y_range.clone() {
            sum += (lhs[[ix, iy]] - rhs[[ix, iy]]).abs();
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

/// A focused bowl in a homogeneous-water domain must place its
/// peak-pressure at (or within one wavelength of) the geometric focus.
/// This isolates the FDTD focal-law from any tissue-heterogeneity or
/// body/water-interface effects: any failure here is a bug in the source
/// model, delay computation, padded-grid embedding, or downsampling.
#[test]
fn focused_bowl_in_water_peaks_at_geometric_focus() {
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

    // 41 × 41 body grid at 0.4 mm spacing (≈ 16 mm domain) — the
    // "body" is the entire grid, all water-speed, so the whole grid
    // is the body sub-region inside the padded simulation.  16 bowl
    // elements on a half-circle of radius 25 mm (well outside the body
    // grid: the padded margin grows to encompass them).  Half-aperture
    // 60° facing the focus from the −x side.  Focus at the grid
    // centre.  Frequency 500 kHz → λ = 3 mm = 7.5 cells, well above the
    // 4-pts/wavelength FD stencil resolution threshold.
    let n = 41usize;
    let spacing = 0.0004_f64;
    let speed = SOUND_SPEED_WATER_SIM;
    let prepared = water_fixture(n, spacing, speed);

    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.frequencies_hz = vec![500_000.0];
    config.source_pressure_pa = 1.0e6;
    let bowl_radius_m = 0.025;
    let half_aperture_rad = 60.0_f64.to_radians();
    let elements = focused_bowl_arc(16, bowl_radius_m, half_aperture_rad);
    let layout = DeviceLayout {
        therapy_elements: elements,
        imaging_receivers: Vec::new(),
        focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
        skin_contact_m: Point2 {
            x_m: -bowl_radius_m,
            y_m: 0.0,
        },
        model_name: "homogeneous_bowl_test".to_owned(),
    };

    let result = simulate_peak_pressure_exposure(&prepared, &layout, &config);
    let raw = &result.raw_peak_pressure;
    let (px, py, peak_value) = arg_max_2d(raw);

    let cx = 0.5 * (n as f64 - 1.0);
    let peak_x_m = (px as f64 - cx) * spacing;
    let peak_y_m = (py as f64 - cx) * spacing;
    let offset_m = peak_x_m.hypot(peak_y_m);
    let wavelength_m = speed / config.frequencies_hz[0];

    let in_body_mean: f64 = raw.iter().sum::<f64>() / (n * n) as f64;
    let peak_over_mean = peak_value / in_body_mean.max(1.0e-30);

    assert!(
        offset_m <= 2.0 * wavelength_m,
        "homogeneous-water focused bowl: peak at cell ({px},{py}) = \
             ({peak_x_m:.4} m, {peak_y_m:.4} m), offset {offset_m:.4} m \
             from focus, exceeds 2 wavelengths ({:.4} m).  Coherent gain \
             peak/mean = {peak_over_mean:.2}, peak = {peak_value:.3e} Pa.",
        2.0 * wavelength_m
    );
    assert!(
        peak_over_mean >= 3.0,
        "homogeneous-water bowl peak/mean ratio = {peak_over_mean:.2} \
             is below the coherent-focusing threshold (≥ 3 expected for 16 \
             elements at this aperture / focal-length geometry)"
    );
}

/// Clinical-scale bowl + body geometry: 256-element bowl at 140 mm
/// focal radius, soft-tissue body of 75 mm radius, focus at body
/// centre.  Verifies the focal-law steers the peak to the geometric
/// focus at the actual liver/kidney scale (not a scaled-down toy).
#[test]
fn clinical_scale_bowl_through_tissue_body_peaks_at_geometric_focus() {
    use kwavers_core::constants::fundamental::{SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM};

    // 51 × 51 body grid at 3 mm spacing — matches typical clinical
    // solver resolution (the waveform module internally refines this
    // by 8× to ~0.4 mm so λ/dx_refined ≈ 7.5 at 500 kHz).
    let n = 51usize;
    let spacing = 0.003_f64;
    let cx_cells = 0.5 * (n as f64 - 1.0);
    let body_radius_m = 0.075;
    let mut speed = Array2::from_elem((n, n), SOUND_SPEED_WATER_SIM);
    for ix in 0..n {
        for iy in 0..n {
            let x = (ix as f64 - cx_cells) * spacing;
            let y = (iy as f64 - cx_cells) * spacing;
            if x.hypot(y) <= body_radius_m {
                speed[[ix, iy]] = SOUND_SPEED_TISSUE;
            }
        }
    }
    let prepared = water_fixture(n, spacing, SOUND_SPEED_TISSUE);
    let prepared = PreparedTheranosticSlice {
        sound_speed_m_s: speed.clone(),
        ..prepared
    };

    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.frequencies_hz = vec![500_000.0];
    config.source_pressure_pa = 28.0e6;
    let bowl_radius_m = 0.140;
    let half_aperture_rad = 54.0_f64.to_radians();
    let elements = focused_bowl_arc(256, bowl_radius_m, half_aperture_rad);
    let layout = DeviceLayout {
        therapy_elements: elements,
        imaging_receivers: Vec::new(),
        focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
        skin_contact_m: Point2 {
            x_m: -body_radius_m,
            y_m: 0.0,
        },
        model_name: "clinical_scale_bowl_test".to_owned(),
    };

    let result = simulate_peak_pressure_exposure(&prepared, &layout, &config);
    let raw = &result.raw_peak_pressure;
    let (px, py, peak_value) = arg_max_2d(raw);

    let peak_x_m = (px as f64 - cx_cells) * spacing;
    let peak_y_m = (py as f64 - cx_cells) * spacing;
    let offset_m = peak_x_m.hypot(peak_y_m);
    let wavelength_m = SOUND_SPEED_TISSUE / config.frequencies_hz[0];

    let in_body_mean: f64 = raw.iter().sum::<f64>() / (n * n) as f64;
    let peak_over_mean = peak_value / in_body_mean.max(1.0e-30);

    assert!(
        offset_m <= 4.0 * wavelength_m,
        "clinical-scale bowl: peak at cell ({px},{py}) = \
             ({peak_x_m:.4} m, {peak_y_m:.4} m), offset {offset_m:.4} m \
             from focus, exceeds 4 wavelengths ({:.4} m).  256-element \
             bowl should converge tightly at the geometric focus.  \
             peak = {peak_value:.3e} Pa, peak/mean = {peak_over_mean:.2}.",
        4.0 * wavelength_m
    );
}

/// Same clinical-scale geometry, but with the focus offset from the
/// body centre (matches the real liver/kidney case where the focus
/// lands on the off-centre tumor).  Uses the PRODUCTION spacing (6 mm
/// body grid, which the waveform module refines internally by 8× →
/// 0.75 mm, giving λ/dx_refined = 4 at 500 kHz — exactly the
/// production resolution).  Bowl axis is computed from focus → nearest
/// skin contact; elements wrap a hemisphere about that axis.
/// Verifies the focal-law still steers the peak to the off-centre focus
/// at production-grade resolution and not just the toy 3 mm spacing.
#[test]
fn clinical_scale_bowl_off_centre_focus_peaks_at_focus() {
    use kwavers_core::constants::fundamental::{SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM};

    let n = 51usize;
    let spacing = 0.006_f64;
    let cx_cells = 0.5 * (n as f64 - 1.0);
    let body_radius_m = 0.075;
    // Body disk centred at the grid origin.
    let mut speed = Array2::from_elem((n, n), SOUND_SPEED_WATER_SIM);
    for ix in 0..n {
        for iy in 0..n {
            let x = (ix as f64 - cx_cells) * spacing;
            let y = (iy as f64 - cx_cells) * spacing;
            if x.hypot(y) <= body_radius_m {
                speed[[ix, iy]] = SOUND_SPEED_TISSUE;
            }
        }
    }
    let prepared = water_fixture(n, spacing, SOUND_SPEED_TISSUE);
    let prepared = PreparedTheranosticSlice {
        sound_speed_m_s: speed.clone(),
        ..prepared
    };

    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.frequencies_hz = vec![500_000.0];
    config.source_pressure_pa = 28.0e6;

    // Focus 30 mm off-centre (matches the liver tumour offset scale).
    let focus = Point2 {
        x_m: -0.030,
        y_m: -0.020,
    };
    // Skin contact: nearest body skin point along the line from focus
    // to the body centre and outward — for a body disk centred at the
    // origin and focus at (−30, −20) mm, the line from focus through
    // origin exits the body at radius body_radius_m in direction (+x, +y),
    // and the OPPOSITE direction (skin contact on the focus side) is
    // at angle atan2(−20, −30) at radius body_radius_m: ~(−62, −41) mm.
    let focus_norm = focus.x_m.hypot(focus.y_m);
    let skin_contact = Point2 {
        x_m: focus.x_m / focus_norm * body_radius_m,
        y_m: focus.y_m / focus_norm * body_radius_m,
    };

    let bowl_radius_m = 0.140;
    let half_aperture_rad = 54.0_f64.to_radians();
    // Elements: hemisphere arc about the (focus → skin_contact) axis,
    // anchored at radius `bowl_radius_m` from focus.
    let axis_x = (skin_contact.x_m - focus.x_m)
        / (skin_contact.x_m - focus.x_m).hypot(skin_contact.y_m - focus.y_m);
    let axis_y = (skin_contact.y_m - focus.y_m)
        / (skin_contact.x_m - focus.x_m).hypot(skin_contact.y_m - focus.y_m);
    let perp_x = -axis_y;
    let perp_y = axis_x;
    let elements: Vec<Point2> = (0..256)
        .map(|idx| {
            let t = idx as f64 / 255.0;
            let theta = -half_aperture_rad + 2.0 * half_aperture_rad * t;
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let dx_e = bowl_radius_m * (cos_t * axis_x + sin_t * perp_x);
            let dy_e = bowl_radius_m * (cos_t * axis_y + sin_t * perp_y);
            Point2 {
                x_m: focus.x_m + dx_e,
                y_m: focus.y_m + dy_e,
            }
        })
        .collect();

    let layout = DeviceLayout {
        therapy_elements: elements,
        imaging_receivers: Vec::new(),
        focus_m: focus,
        skin_contact_m: skin_contact,
        model_name: "clinical_offcentre_bowl_test".to_owned(),
    };

    let result = simulate_peak_pressure_exposure(&prepared, &layout, &config);
    let raw = &result.raw_peak_pressure;
    let (px, py, peak_value) = arg_max_2d(raw);

    let peak_x_m = (px as f64 - cx_cells) * spacing;
    let peak_y_m = (py as f64 - cx_cells) * spacing;
    let dx_e = peak_x_m - focus.x_m;
    let dy_e = peak_y_m - focus.y_m;
    let offset_m = dx_e.hypot(dy_e);
    let wavelength_m = SOUND_SPEED_TISSUE / config.frequencies_hz[0];
    let in_body_mean: f64 = raw.iter().sum::<f64>() / (n * n) as f64;
    let peak_over_mean = peak_value / in_body_mean.max(1.0e-30);

    assert!(
        offset_m <= 6.0 * wavelength_m,
        "off-centre focus: peak at cell ({px},{py}) = ({peak_x_m:.4}, \
             {peak_y_m:.4}) m, focus at ({:.4}, {:.4}) m, offset {offset_m:.4} m \
             exceeds 6 wavelengths ({:.4} m).  peak = {peak_value:.3e} Pa, \
             peak/mean = {peak_over_mean:.2}.",
        focus.x_m,
        focus.y_m,
        6.0 * wavelength_m
    );
}

/// Add a soft-tissue disk inside the water domain (mimicking the
/// abdominal slice) and verify that the path-integrated focal-law keeps
/// the focal peak inside the body, near the geometric focus.
#[test]
fn focused_bowl_through_tissue_disk_peaks_at_geometric_focus() {
    use kwavers_core::constants::fundamental::{SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM};

    let n = 41usize;
    let spacing = 0.0004_f64;
    let cx_cells = 0.5 * (n as f64 - 1.0);
    // Soft-tissue disk of radius 6 mm (15 cells) centred at the focus.
    let body_radius_m = 0.006_f64;
    let mut speed = Array2::from_elem((n, n), SOUND_SPEED_WATER_SIM);
    for ix in 0..n {
        for iy in 0..n {
            let x = (ix as f64 - cx_cells) * spacing;
            let y = (iy as f64 - cx_cells) * spacing;
            if x.hypot(y) <= body_radius_m {
                speed[[ix, iy]] = SOUND_SPEED_TISSUE;
            }
        }
    }
    let prepared = water_fixture(n, spacing, SOUND_SPEED_TISSUE);
    let prepared = PreparedTheranosticSlice {
        sound_speed_m_s: speed.clone(),
        ..prepared
    };

    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.frequencies_hz = vec![500_000.0];
    config.source_pressure_pa = 1.0e6;
    let bowl_radius_m = 0.025;
    let half_aperture_rad = 60.0_f64.to_radians();
    let elements = focused_bowl_arc(16, bowl_radius_m, half_aperture_rad);
    let layout = DeviceLayout {
        therapy_elements: elements,
        imaging_receivers: Vec::new(),
        focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
        skin_contact_m: Point2 {
            x_m: -body_radius_m,
            y_m: 0.0,
        },
        model_name: "heterogeneous_bowl_test".to_owned(),
    };

    let result = simulate_peak_pressure_exposure(&prepared, &layout, &config);
    let raw = &result.raw_peak_pressure;
    let (px, py, peak_value) = arg_max_2d(raw);

    let peak_x_m = (px as f64 - cx_cells) * spacing;
    let peak_y_m = (py as f64 - cx_cells) * spacing;
    let offset_m = peak_x_m.hypot(peak_y_m);
    let wavelength_m = SOUND_SPEED_TISSUE / config.frequencies_hz[0];

    assert!(
        offset_m <= 3.0 * wavelength_m,
        "heterogeneous (water + tissue disk) bowl: peak at cell \
             ({px},{py}) = ({peak_x_m:.4} m, {peak_y_m:.4} m), offset \
             {offset_m:.4} m from focus, exceeds 3 wavelengths \
             ({:.4} m).  Path-integrated focal-law should hold the focus \
             in the body interior.  Peak = {peak_value:.3e} Pa.",
        3.0 * wavelength_m
    );
}

fn water_fixture(n: usize, spacing: f64, speed: f64) -> PreparedTheranosticSlice {
    let mut target_mask = Array2::from_elem((n, n), false);
    let mid = n / 2;
    for ix in (mid - 1)..=(mid + 1) {
        for iy in (mid - 1)..=(mid + 1) {
            target_mask[[ix, iy]] = true;
        }
    }
    PreparedTheranosticSlice {
        anatomy: AnatomyKind::Liver,
        ct_hu: Array2::from_elem((n, n), 0.0),
        label: Array2::zeros((n, n)),
        sound_speed_m_s: Array2::from_elem((n, n), speed),
        attenuation_np_per_m_mhz: Array2::from_elem((n, n), 0.0),
        body_mask: Array2::from_elem((n, n), true),
        organ_mask: Array2::from_elem((n, n), true),
        target_mask,
        spacing_m: spacing,
        source_slice_index: 0,
        source_dimensions: [n, n],
        source_spacing_m: [spacing, spacing],
        crop_bounds_index: [0, n - 1, 0, n - 1],
    }
}

fn focused_bowl_arc(count: usize, radius_m: f64, half_angle_rad: f64) -> Vec<Point2> {
    (0..count)
        .map(|idx| {
            let t = if count > 1 {
                idx as f64 / (count - 1) as f64
            } else {
                0.5
            };
            let theta = -half_angle_rad + 2.0 * half_angle_rad * t;
            Point2 {
                x_m: -radius_m * theta.cos(),
                y_m: radius_m * theta.sin(),
            }
        })
        .collect()
}

fn arg_max_2d(field: &Array2<f64>) -> (usize, usize, f64) {
    let mut best = (0usize, 0usize, f64::NEG_INFINITY);
    for ((ix, iy), &v) in field.indexed_iter() {
        if v > best.2 {
            best = (ix, iy, v);
        }
    }
    best
}
