//! Smoke tests for the [`ElasticWave`] state holder + the consolidated
//! spectral-elastic primitives in
//! [`crate::solver::forward::pstd::extensions::elastic`].

#[cfg(test)]
use crate::domain::grid::Grid;
#[cfg(test)]
use crate::physics::acoustics::mechanics::elastic_wave::ElasticWave;

/// `ElasticWave::new` succeeds on a small grid and produces a struct whose
/// wavenumber axis lengths match the grid dimensions. Acts as a constructor
/// guard for the state holder used by the planned ElasticPSTD orchestrator.
#[test]
fn test_elastic_wave_constructor_initialises_wavenumber_axes() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let elastic_wave = ElasticWave::new(&grid).unwrap();
    // kx/ky/kz are stored as (n, 1, 1) wavenumber arrays — see
    // `create_wavenumber_array` in mod.rs.
    assert_eq!(elastic_wave.kx.dim(), (32, 1, 1));
    assert_eq!(elastic_wave.ky.dim(), (32, 1, 1));
    assert_eq!(elastic_wave.kz.dim(), (32, 1, 1));
}

/// The spectral-elastic stress kernel reduces to the acoustic-fluid limit
/// when `μ ≡ 0`: the shear-stress outputs (txy/txz/tyz) are exactly zero
/// regardless of the velocity field. This is the executable counterpart of
/// the acoustic-fluid-limit theorem documented on the plugin module.
#[test]
fn pstd_elastic_plugin_reduces_to_acoustic_when_mu_is_zero() {
    use crate::physics::acoustics::mechanics::elastic_wave::parameters::StressUpdateParams;
    use crate::physics::acoustics::mechanics::elastic_wave::spectral_fields::SpectralStressFields;
    use crate::solver::forward::pstd::extensions::PstdElasticPlugin;
    use ndarray::Array3;
    use num_complex::Complex;

    let nx = 8usize;
    let ny = 8usize;
    let nz = 8usize;

    // Non-trivial spectral velocity field — every cell carries a unique
    // complex sample so any spurious shear contribution would surface.
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

    // Complex spectral derivative operators `D_α = i·k_α` (collocated; no
    // half-cell shift), shaped `(N_α, 1, 1)` and indexed by axis position.
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
    let lame_mu = Array3::<f64>::zeros((nx, ny, nz)); // acoustic-fluid limit
    let density = Array3::<f64>::from_elem((nx, ny, nz), 1000.0);

    // Zero current stress — initial-rest condition replicates the prior
    // single-step semantics so the theorem assertion below is exact.
    let stress_current = SpectralStressFields::new(nx, ny, nz);

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
    };

    let mut out = SpectralStressFields::new(nx, ny, nz);
    let plugin = PstdElasticPlugin::default();
    plugin.apply_stress_update_in_place(&params, &mut out);

    let zero = Complex::new(0.0, 0.0);
    for x in out.txy.iter().chain(out.txz.iter()).chain(out.tyz.iter()) {
        assert_eq!(*x, zero, "shear-stress must be zero when μ = 0");
    }
    // Normal stresses should be non-zero (driven by ∇·v).
    let any_normal_nonzero = out
        .txx
        .iter()
        .chain(out.tyy.iter())
        .chain(out.tzz.iter())
        .any(|x| *x != zero);
    assert!(
        any_normal_nonzero,
        "normal stresses should be non-zero from a non-trivial velocity field"
    );
}
