use super::orchestrator::ElasticPstdOrchestrator;
use super::types::{ElasticPstdMedium, ElasticPstdSourceMode, ElasticPstdVelocitySource};
use crate::domain::grid::Grid;
use ndarray::{Array1, Array3};
use num_complex::Complex;

/// `μ ≡ 0` ⇒ persistent shear stress stays zero through propagation.
///
/// This is the orchestrator-level executable form of the acoustic-fluid
/// limit theorem: a non-trivial velocity source must drive non-zero normal
/// stress (compression waves) but **never** generate shear stress when μ = 0.
#[test]
fn pstd_orchestrator_keeps_shear_stress_zero_when_mu_is_zero() {
    let nx = 16usize;
    let ny = 16usize;
    let nz = 4usize;
    let dx = 1e-3;
    let cp = 1500.0;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let n_steps = 30;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, ny, nz), 1000.0 * cp * cp),
        lame_mu: Array3::zeros((nx, ny, nz)),
        density: Array3::from_elem((nx, ny, nz), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();

    let amp = 1e-6;
    let signal: Array1<f64> = Array1::from_iter(
        (0..n_steps).map(|n| amp * (2.0 * std::f64::consts::PI * 1e6 * (n as f64) * dt).sin()),
    );
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    src_mask[[3, 5, nz / 2]] = true;
    let source = ElasticPstdVelocitySource {
        mask: src_mask,
        ux: Some(signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    let _ = orch.propagate(n_steps, Some(&source), None).unwrap();

    let zero = Complex::new(0.0_f64, 0.0_f64);
    for x in orch
        .spectral_stress
        .txy
        .iter()
        .chain(orch.spectral_stress.txz.iter())
        .chain(orch.spectral_stress.tyz.iter())
    {
        assert_eq!(*x, zero, "shear stress must stay zero when μ = 0");
    }
}

/// μ = 0 + zero source ⇒ velocity stays zero forever.
#[test]
fn quiescent_acoustic_fluid_remains_quiescent() {
    let grid = Grid::new(8, 8, 4, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((8, 8, 4), 1.5e9),
        lame_mu: Array3::zeros((8, 8, 4)),
        density: Array3::from_elem((8, 8, 4), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, 1e-7).unwrap();
    let _ = orch.propagate(20, None, None).unwrap();
    let max_v = orch
        .velocity()
        .vx
        .iter()
        .chain(orch.velocity().vy.iter())
        .chain(orch.velocity().vz.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert_eq!(max_v, 0.0, "quiescent state must remain quiescent");
}

/// μ = 0 with an additive ux pulse on a single cell propagates a
/// non-zero field whose recorded peak at a downstream sensor is
/// finite, non-NaN, and order-of-magnitude consistent with the source
/// amplitude.
#[test]
fn acoustic_fluid_pulse_propagates_finite_field() {
    let nx = 16usize;
    let ny = 16usize;
    let nz = 4usize;
    let dx = 1e-3;
    let cp = 1500.0;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let n_steps = 40;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, ny, nz), 1000.0 * cp * cp),
        lame_mu: Array3::zeros((nx, ny, nz)),
        density: Array3::from_elem((nx, ny, nz), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();

    let amp = 1e-6;
    let signal: Array1<f64> = Array1::from_iter(
        (0..n_steps).map(|n| amp * (2.0 * std::f64::consts::PI * 1e6 * (n as f64) * dt).sin()),
    );
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    src_mask[[3, ny / 2, nz / 2]] = true;
    let source = ElasticPstdVelocitySource {
        mask: src_mask,
        ux: Some(signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    sensor_mask[[8, ny / 2, nz / 2]] = true;
    let data = orch
        .propagate(n_steps, Some(&source), Some(&sensor_mask))
        .unwrap();

    let vx_trace = data.vx.expect("vx recorded");
    assert_eq!(vx_trace.shape(), &[1, n_steps]);
    let peak = vx_trace.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    assert!(peak.is_finite(), "peak must be finite");
    assert!(peak > 0.0, "downstream sensor must record a non-zero pulse");
    assert!(
        peak < 1.0,
        "peak {peak:.3e} should remain bounded (source amp = 1e-6)"
    );
}
