//! Tests for the [`ElasticWave`] state holder.
//!
//! Unit tests for field types (`StressFields`, `VelocityFields`),
//! spectral round-trips (`SpectralStressFields`, `SpectralVelocityFields`),
//! metrics (`ElasticWaveMetrics`), and properties (`AnisotropicElasticProperties`)
//! are co-located with their respective modules.

use crate::core::constants::fundamental::DENSITY_WATER_NOMINAL;
use crate::domain::grid::Grid;
use crate::physics::acoustics::mechanics::elastic_wave::{
    mode_conversion::{ModeConversionConfig, ViscoelasticConfig},
    properties::AnisotropicElasticProperties,
    spectral_fields::SpectralStressFields,
    ElasticWave,
};
use ndarray::Array4;
use std::time::Duration;
use crate::core::constants::numerical::{TWO_PI};

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
    use crate::domain::medium::heterogeneous::HeterogeneousMedium;
    use crate::domain::medium::HomogeneousMedium;

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

// ── Acoustic-fluid limit (spectral domain) ────────────────────────────────────

/// When μ ≡ 0, the spectral stress kernel produces zero shear stress for any
/// non-trivial velocity field. This is the executable counterpart of the
/// acoustic-fluid-limit theorem documented on the plugin module.
#[test]
fn pstd_elastic_plugin_reduces_to_acoustic_when_mu_is_zero() {
    use crate::physics::acoustics::mechanics::elastic_wave::parameters::StressUpdateParams;
    use crate::solver::forward::pstd::extensions::PstdElasticPlugin;
    use ndarray::Array3;
    use num_complex::Complex;

    let (nx, ny, nz) = (8usize, 8, 8);
    let make_v = || {
        let mut v = Array3::<Complex<f64>>::zeros((nx, ny, nz));
        for ((i, j, k), x) in v.indexed_iter_mut() {
            *x = Complex::new((i + j + k) as f64 + 1.0, (i * j + 1) as f64);
        }
        v
    };
    let vx_fft = make_v();
    let vy_fft = make_v();
    let vz_fft = make_v();

    let mut dkx_op = Array3::<Complex<f64>>::zeros((nx, 1, 1));
    let mut dky_op = Array3::<Complex<f64>>::zeros((ny, 1, 1));
    let mut dkz_op = Array3::<Complex<f64>>::zeros((nz, 1, 1));
    for i in 0..nx {
        dkx_op[[i, 0, 0]] = Complex::new(0.0, (i + 1) as f64 * 0.1);
    }
    for j in 0..ny {
        dky_op[[j, 0, 0]] = Complex::new(0.0, (j + 1) as f64 * 0.1);
    }
    for k in 0..nz {
        dkz_op[[k, 0, 0]] = Complex::new(0.0, (k + 1) as f64 * 0.1);
    }

    let lame_lambda = Array3::<f64>::from_elem((nx, ny, nz), 2.25e9);
    let lame_mu = Array3::<f64>::zeros((nx, ny, nz));
    let density = Array3::<f64>::from_elem((nx, ny, nz), DENSITY_WATER_NOMINAL);
    let stress_current = SpectralStressFields::new(nx, ny, nz);
    let unit_kappa = Array3::<f64>::ones((nx, ny, nz));

    let params = StressUpdateParams {
        vx_fft: &vx_fft,
        vy_fft: &vy_fft,
        vz_fft: &vz_fft,
        txx_fft: &stress_current.txx,
        tyy_fft: &stress_current.tyy,
        tzz_fft: &stress_current.tzz,
        txy_fft: &stress_current.txy,
        txz_fft: &stress_current.txz,
        tyz_fft: &stress_current.tyz,
        dkx_op: &dkx_op,
        dky_op: &dky_op,
        dkz_op: &dkz_op,
        lame_lambda: &lame_lambda,
        lame_mu: &lame_mu,
        density: density.view(),
        dt: 1e-7,
        kappa: &unit_kappa,
    };

    let mut out = SpectralStressFields::new(nx, ny, nz);
    let plugin = PstdElasticPlugin::default();
    plugin.apply_stress_update_in_place(&params, &mut out);

    let zero = Complex::new(0.0, 0.0);
    for x in out.txy.iter().chain(out.txz.iter()).chain(out.tyz.iter()) {
        assert_eq!(*x, zero, "shear stress must be zero when μ = 0");
    }
    let any_normal_nonzero = out
        .txx
        .iter()
        .chain(out.tyy.iter())
        .chain(out.tzz.iter())
        .any(|x| *x != zero);
    assert!(
        any_normal_nonzero,
        "normal stresses must be non-zero for a non-trivial velocity field"
    );
}
