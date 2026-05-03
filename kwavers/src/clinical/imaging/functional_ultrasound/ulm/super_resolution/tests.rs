//! Tests for super-resolution reconstruction.

use super::reconstructor::SuperResReconstructor;
use super::types::{RenderMode, SuperResConfig};
use crate::clinical::imaging::functional_ultrasound::ulm::microbubble_detection::BubbleDetection;
use crate::clinical::imaging::functional_ultrasound::ulm::tracking::BubbleTrack;

fn make_track(positions: &[(f64, f64)]) -> BubbleTrack {
    let dets: Vec<BubbleDetection> = positions
        .iter()
        .enumerate()
        .map(|(i, &(x, z))| BubbleDetection {
            x,
            z,
            amplitude: 1.0,
            sigma: 1.0,
            background: 0.0,
            frame: i,
        })
        .collect();
    BubbleTrack {
        id: 0,
        detections: dets,
        last_frame: positions.len().saturating_sub(1),
        gap: 0,
        active: false,
    }
}

#[test]
fn test_histogram_single_localization() {
    let config = SuperResConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 5e-6,
        mode: RenderMode::Histogram,
        smooth_halfwidth: 0,
        ..Default::default()
    };
    let mut recon = SuperResReconstructor::new(config).unwrap();
    let track = make_track(&[(100e-6, 200e-6)]);
    recon.accumulate(&[track]);

    let ix = (100e-6_f64 / 5e-6) as usize;
    let iz = (200e-6_f64 / 5e-6) as usize;
    assert_eq!(
        recon.image()[[ix, iz]],
        1.0,
        "Single localization should give count=1"
    );

    let total: f64 = recon.image().sum();
    assert!((total - 1.0).abs() < 1e-10, "total={total}");
}

#[test]
fn test_histogram_multiple_localizations() {
    let config = SuperResConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 5e-6,
        mode: RenderMode::Histogram,
        smooth_halfwidth: 0,
        ..Default::default()
    };
    let mut recon = SuperResReconstructor::new(config).unwrap();
    let track = make_track(&[(100e-6, 200e-6), (101e-6, 200e-6), (102e-6, 200e-6)]);
    recon.accumulate(&[track]);
    let total: f64 = recon.image().sum();
    assert!((total - 3.0).abs() < 1e-10, "total count must be 3");
}

#[test]
fn test_gaussian_splat_peak_below_one() {
    let config = SuperResConfig {
        x_extent: 500e-6,
        z_extent: 500e-6,
        pixel_size: 5e-6,
        gauss_sigma: 20e-6,
        mode: RenderMode::GaussianSplat,
        smooth_halfwidth: 0,
        ..Default::default()
    };
    let mut recon = SuperResReconstructor::new(config).unwrap();
    let track = make_track(&[(250e-6, 250e-6)]);
    recon.accumulate(&[track]);

    let peak = recon
        .image()
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(peak < 1.0, "Gaussian splatted peak {peak:.4} must be < 1");
    assert!(peak > 0.0, "Peak must be positive");
}

#[test]
fn test_density_normalization() {
    let config = SuperResConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 5e-6,
        mode: RenderMode::Histogram,
        smooth_halfwidth: 0,
        total_time_s: Some(2.0),
        ..Default::default()
    };
    let mut recon = SuperResReconstructor::new(config).unwrap();
    let track = make_track(&[(100e-6, 200e-6)]);
    recon.accumulate(&[track]);
    let density = recon.density_image().unwrap();

    let ix = (100e-6_f64 / 5e-6) as usize;
    let iz = (200e-6_f64 / 5e-6) as usize;
    assert!(
        (density[[ix, iz]] - 0.5).abs() < 1e-10,
        "1 count / 2 s = 0.5 Hz, got {}",
        density[[ix, iz]]
    );
}

#[test]
fn test_density_image_error_without_time() {
    let config = SuperResConfig {
        total_time_s: None,
        ..Default::default()
    };
    let recon = SuperResReconstructor::new(config).unwrap();
    assert!(recon.density_image().is_err());
}

#[test]
fn test_sliding_average_smoothing_constant_track() {
    let config = SuperResConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 5e-6,
        smooth_halfwidth: 2,
        ..Default::default()
    };
    let recon = SuperResReconstructor::new(config).unwrap();
    let track = make_track(&[(200e-6, 300e-6); 5]);
    let positions = recon.smooth_track(&track);
    for (x, z) in positions {
        assert!((x - 200e-6).abs() < 1e-12, "x should be constant");
        assert!((z - 300e-6).abs() < 1e-12, "z should be constant");
    }
}

#[test]
fn test_sliding_average_reduces_noise() {
    let config = SuperResConfig {
        x_extent: 2e-3,
        z_extent: 2e-3,
        pixel_size: 5e-6,
        smooth_halfwidth: 2,
        ..Default::default()
    };
    let recon = SuperResReconstructor::new(config).unwrap();
    let eps = 100e-6;
    let center_x = 1e-3;
    let center_z = 1e-3;
    let raw: Vec<(f64, f64)> = (0..10)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            (center_x + sign * eps, center_z)
        })
        .collect();
    let track = make_track(&raw);
    let smoothed = recon.smooth_track(&track);

    for (x, _) in &smoothed[2..8] {
        let dev = (x - center_x).abs();
        assert!(
            dev < eps,
            "Smoothed deviation {dev:.4e} >= raw deviation {eps:.4e}"
        );
    }
}

#[test]
fn test_grid_size_matches_config() {
    let config = SuperResConfig {
        x_extent: 0.01,
        z_extent: 0.012,
        pixel_size: 5e-6,
        ..Default::default()
    };
    let recon = SuperResReconstructor::new(config).unwrap();
    let (nx, nz) = recon.grid_size();
    assert_eq!(nx, 2000, "10 mm / 5 μm = 2000");
    assert_eq!(nz, 2400, "12 mm / 5 μm = 2400");
}

#[test]
fn test_out_of_bounds_localizations_ignored() {
    let config = SuperResConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 5e-6,
        mode: RenderMode::Histogram,
        smooth_halfwidth: 0,
        ..Default::default()
    };
    let mut recon = SuperResReconstructor::new(config).unwrap();
    let track = make_track(&[(2e-3, 500e-6)]);
    recon.accumulate(&[track]);
    let total: f64 = recon.image().sum();
    assert!(
        (total - 0.0).abs() < 1e-10,
        "Out-of-bounds localization must be ignored"
    );
}
