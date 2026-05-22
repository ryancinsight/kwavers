use std::f64::consts::PI;

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;

use super::{GaussianBeam, KwaveAnalyticalPlaneWave, KwaveErrorMetrics, SphericalWave};

#[test]
fn test_plane_wave_creation() {
    let wave = KwaveAnalyticalPlaneWave::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, [1.0, 0.0, 0.0], 0.0).unwrap();
    assert_eq!(wave.amplitude, 1e5);
    assert_eq!(wave.frequency, 1e6);
    assert!((wave.wavelength() - 1.5e-3).abs() < 1e-10);
}

#[test]
fn test_plane_wave_direction_normalization() {
    let wave = KwaveAnalyticalPlaneWave::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, [3.0, 4.0, 0.0], 0.0).unwrap();
    let norm =
        (wave.direction[0].powi(2) + wave.direction[1].powi(2) + wave.direction[2].powi(2)).sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_plane_wave_pressure_temporal_periodicity() {
    let wave = KwaveAnalyticalPlaneWave::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, [1.0, 0.0, 0.0], 0.0).unwrap();
    let period = 1.0 / wave.frequency;
    let p1 = wave.pressure(0.0, 0.0, 0.0, 0.0);
    let p2 = wave.pressure(0.0, 0.0, 0.0, period);
    let relative_error = (p1 - p2).abs() / wave.amplitude.max(1e-12);
    assert!(
        relative_error < 1e-12,
        "Temporal periodicity violated: p(t=0)={}, p(t=T)={}, relative_error={}",
        p1,
        p2,
        relative_error
    );
}

#[test]
fn test_gaussian_beam_paraxial_check() {
    assert!(GaussianBeam::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, 1e-3, 0.0).is_err());
    assert!(GaussianBeam::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, 5e-3, 0.0).is_ok());
}

#[test]
fn test_gaussian_beam_rayleigh_range() {
    let beam = GaussianBeam::new(1e5, 2e6, SOUND_SPEED_WATER_SIM, 5e-3, 0.0).unwrap();
    let wavelength = beam.sound_speed / beam.frequency;
    let z_r = PI * beam.waist_radius.powi(2) / wavelength;
    assert!((beam.rayleigh_range() - z_r).abs() < 1e-10);
}

#[test]
fn test_gaussian_beam_width_at_rayleigh() {
    let beam = GaussianBeam::new(1e5, 2e6, SOUND_SPEED_WATER_SIM, 5e-3, 0.0).unwrap();
    let z_r = beam.rayleigh_range();
    let w_at_zr = beam.beam_width(z_r);
    assert!((w_at_zr / beam.waist_radius - 2.0f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_spherical_wave_geometric_spreading() {
    let wave = SphericalWave::new(1e3, 1e6, SOUND_SPEED_WATER_SIM, [0.0, 0.0, 0.0], 0.0).unwrap();
    let r1 = 0.01;
    let r2 = 0.02;
    let p1 = wave.pressure(r1, 0.0, 0.0, 0.0);
    let p2 = wave.pressure(r2, 0.0, 0.0, 0.0);
    let ratio = (p1 * r1).abs() / (p2 * r2).abs();
    assert!((ratio - 1.0).abs() < 0.1);
}

#[test]
fn test_error_metrics_perfect_match() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let wave = KwaveAnalyticalPlaneWave::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, [1.0, 0.0, 0.0], 0.0).unwrap();
    let field1 = wave.pressure_field(&grid, 0.0);
    let field2 = wave.pressure_field(&grid, 0.0);
    let metrics = KwaveErrorMetrics::compute(field1.view(), field2.view());
    assert!(metrics.l2_error < 1e-10);
    assert!(metrics.linf_error < 1e-10);
    assert!(metrics.phase_error < 1e-10);
    assert!((metrics.correlation - 1.0).abs() < 1e-10);
    assert!(metrics.meets_acceptance_criteria());
}

#[test]
fn test_error_metrics_phase_shifted() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let wave1 = KwaveAnalyticalPlaneWave::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, [1.0, 0.0, 0.0], 0.0).unwrap();
    let wave2 = KwaveAnalyticalPlaneWave::new(1e5, 1e6, SOUND_SPEED_WATER_SIM, [1.0, 0.0, 0.0], PI / 4.0).unwrap();
    let field1 = wave1.pressure_field(&grid, 0.0);
    let field2 = wave2.pressure_field(&grid, 0.0);
    let metrics = KwaveErrorMetrics::compute(field1.view(), field2.view());
    assert!(metrics.phase_error > 0.1);
    assert!(!metrics.meets_acceptance_criteria());
}
