//! Tests for the [`ElasticWave`] state holder.
//!
//! Unit tests for field types (`StressFields`, `VelocityFields`),
//! spectral round-trips (`SpectralStressFields`, `SpectralVelocityFields`),
//! metrics (`ElasticWaveMetrics`), and properties (`AnisotropicElasticProperties`)
//! are co-located with their respective modules.

use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_domain::grid::Grid;
use crate::acoustics::mechanics::elastic_wave::{
    mode_conversion::{ModeConversionConfig, ViscoelasticConfig},
    properties::AnisotropicElasticProperties,
    spectral_fields::SpectralStressFields,
    ElasticWave,
};
use ndarray::Array4;
use std::time::Duration;

// ── ElasticWave::new / wavenumber axes ────────────────────────────────────────

/// `ElasticWave::new` succeeds on a small grid and produces a struct whose
/// wavenumber axis lengths match the grid dimensions.
#[test]
fn test_elastic_wave_constructor_initialises_wavenumber_axes() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let ew = ElasticWave::new(&grid).unwrap();
    assert_eq!(ew.kx.dim(), (32, 1, 1));
    assert_eq!(ew.ky.dim(), (32, 1, 1));
    assert_eq!(ew.kz.dim(), (32, 1, 1));
}

/// Wavenumber axis satisfies the FFT-frequency layout:
/// k[0] = 0, k[1] = dk, k[N/2] = (N/2 − N)·dk (first negative bin).
/// Analytical: dk = 2π / (N·dx).
#[test]
fn elastic_wave_wavenumber_axis_dc_and_positive_frequencies() {
    let nx = 8usize;
    let dx = 0.001_f64;
    let grid = Grid::new(nx, 2, 2, dx, dx, dx).unwrap();
    let ew = ElasticWave::new(&grid).unwrap();

    let dk = TWO_PI / (nx as f64 * dx);
    assert_eq!(ew.kx[[0, 0, 0]], 0.0, "DC bin must be zero");
    assert!((ew.kx[[1, 0, 0]] - dk).abs() < 1e-12, "first positive bin");
    let nyquist_expected = (nx as f64 / 2.0 - nx as f64) * dk;
    assert!(
        (ew.kx[[nx / 2, 0, 0]] - nyquist_expected).abs() < 1e-12,
        "Nyquist bin"
    );
}

/// Grid with zero-length dimension must be rejected.
#[test]
fn elastic_wave_rejects_zero_dimension() {
    let result = Grid::new(0, 4, 4, 0.001, 0.001, 0.001);
    if let Ok(grid) = result {
        assert!(ElasticWave::new(&grid).is_err());
    }
}

// ── ElasticWave::set_anisotropic ─────────────────────────────────────────────

/// `set_anisotropic` raises the flag and stores the given properties.
#[test]
fn elastic_wave_set_anisotropic_sets_flag_and_stores_properties() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let mut ew = ElasticWave::new(&grid).unwrap();
    assert!(!ew.is_anisotropic);
    assert!(ew.anisotropic_properties.is_none());

    // Aluminum: λ = 51 GPa, μ = 26 GPa → c11 = 103 GPa
    let props = AnisotropicElasticProperties::isotropic(2700.0, 51.0e9, 26.0e9).unwrap();
    ew.set_anisotropic(props);

    assert!(ew.is_anisotropic);
    let stored = ew.anisotropic_properties.as_ref().unwrap();
    let c11_expected = 51.0e9 + 2.0 * 26.0e9;
    assert!((stored.stiffness[0][0] - c11_expected).abs() < 1.0);
}

// ── ElasticWave::enable_mode_conversion ───────────────────────────────────────

/// Stores the `ModeConversionConfig` verbatim.
#[test]
fn elastic_wave_enable_mode_conversion_stores_config() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let mut ew = ElasticWave::new(&grid).unwrap();
    assert!(ew.mode_conversion.is_none());

    let cfg = ModeConversionConfig {
        enable_p_to_s: false,
        conversion_efficiency: 0.7,
        ..ModeConversionConfig::default()
    };
    ew.enable_mode_conversion(cfg);

    let stored = ew.mode_conversion.as_ref().unwrap();
    assert!(!stored.enable_p_to_s);
    assert!((stored.conversion_efficiency - 0.7).abs() < 1e-15);
}

// ── ElasticWave::enable_viscoelastic ──────────────────────────────────────────

/// Stores the `ViscoelasticConfig` verbatim.
#[test]
fn elastic_wave_enable_viscoelastic_stores_config() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let mut ew = ElasticWave::new(&grid).unwrap();
    assert!(ew.viscoelastic.is_none());

    let cfg = ViscoelasticConfig {
        q_p: 200.0,
        q_s: 80.0,
        ..ViscoelasticConfig::default()
    };
    ew.enable_viscoelastic(cfg);

    let stored = ew.viscoelastic.as_ref().unwrap();
    assert!((stored.q_p - 200.0).abs() < 1e-15);
    assert!((stored.q_s - 80.0).abs() < 1e-15);
}

// ── ElasticWave::set_stiffness_tensors ────────────────────────────────────────

/// Stores the stiffness tensor array.
#[test]
fn elastic_wave_set_stiffness_tensors_stores_array() {
    let (nx, ny, nz) = (4, 4, 4);
    let grid = Grid::new(nx, ny, nz, 0.001, 0.001, 0.001).unwrap();
    let mut ew = ElasticWave::new(&grid).unwrap();
    assert!(ew.stiffness_tensors.is_none());

    let tensors = Array4::from_elem((6, nx, ny, nz), 1.5e9_f64);
    ew.set_stiffness_tensors(tensors);

    let stored = ew.stiffness_tensors.as_ref().unwrap();
    assert_eq!(stored.dim(), (6, nx, ny, nz));
    assert_eq!(stored[[0, 0, 0, 0]], 1.5e9);
}

// ── ElasticWave::detect_interfaces ────────────────────────────────────────────

/// Inserts a two-fold density jump at i = nx/2 and verifies the interface
/// mask is set at boundary-adjacent cells while interior cells remain clear.
#[test]
fn elastic_wave_detect_interfaces_marks_density_jump() {
    use kwavers_domain::medium::heterogeneous::HeterogeneousMedium;
    use kwavers_domain::medium::HomogeneousMedium;

    let (nx, ny, nz) = (10usize, 4, 4);
    let grid = Grid::new(nx, ny, nz, 0.001, 0.001, 0.001).unwrap();
    let water = HomogeneousMedium::water(&grid);
    // ρ_water ≈ 1000 kg/m³; inject a 100% jump at the midplane.
    let mut medium = HeterogeneousMedium::from_homogeneous(&water, &grid);
    let jump_i = nx / 2;
    for j in 0..ny {
        for k in 0..nz {
            medium.density[[jump_i, j, k]] = 2000.0;
        }
    }

    let mut ew = ElasticWave::new(&grid).unwrap();
    // 50% threshold: a 100% jump triggers the interface flag.
    ew.detect_interfaces(&medium, &grid, 0.5);

    let mask = ew.interface_mask.as_ref().unwrap();
    assert_eq!(mask.dim(), (nx, ny, nz));
    assert!(
        mask[[jump_i - 1, 1, 1]],
        "cell left of jump must be flagged"
    );
    assert!(
        mask[[jump_i + 1, 1, 1]],
        "cell right of jump must be flagged"
    );
    assert!(!mask[[1, 1, 1]], "uniform interior must not be flagged");
}

// ── ElasticWave::metrics / reset_metrics ──────────────────────────────────────

/// `metrics()` returns a live reference to the embedded metrics struct.
#[test]
fn elastic_wave_metrics_accessor_returns_embedded_metrics() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let mut ew = ElasticWave::new(&grid).unwrap();
    ew.metrics.increment_steps();
    ew.metrics.update_max_velocity(3.0);
    assert_eq!(ew.metrics().total_steps, 1);
    assert!((ew.metrics().max_velocity - 3.0).abs() < 1e-15);
}

/// After `reset_metrics`, all counters and maxima return to zero.
#[test]
fn elastic_wave_reset_metrics_zeros_all_fields() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let mut ew = ElasticWave::new(&grid).unwrap();
    ew.metrics.increment_steps();
    ew.metrics.update_max_velocity(5.0);
    ew.metrics.add_fft_time(Duration::from_millis(10));
    ew.reset_metrics();
    assert_eq!(ew.metrics().total_steps, 0);
    assert_eq!(ew.metrics().max_velocity, 0.0);
    assert_eq!(ew.metrics().fft_time, Duration::ZERO);
}
