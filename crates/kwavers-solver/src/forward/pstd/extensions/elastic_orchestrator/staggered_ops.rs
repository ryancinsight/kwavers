use kwavers_math::fft::Complex64;
use leto::Array3;

/// Pre-built complex spectral derivative operators for the staggered-grid
/// PSTD scheme used by KWave.jl `pstd_elastic_2d` (Treeby & Cox 2010, eq.
/// 16). For each axis α, the operator stored here is
///
/// ```text
///   D_α^±[k_α] = i · k_α · exp(± i · k_α · Δα / 2)
/// ```
///
/// so the plugin's spectral derivative collapses to `D_α^± · F_α` instead
/// of the collocated `i · k_α · F_α`. The `+` set is used for `∇·σ`
/// (velocity update — sampled at the velocity grid), the `−` set for
/// `∇v` (stress update — sampled at the stress grid). Without the shift
/// the orchestrator runs a collocated-grid scheme, which numerically
/// disagrees with KWave.jl at non-trivial wavenumbers (matched-mode
/// peak_ratio sat at 0.13–0.23 instead of ≈ 1.0 prior to this change).
#[derive(Debug)]
pub(super) struct StaggeredDerivativeOps {
    pub(super) dkx_pos: Array3<Complex64>,
    pub(super) dky_pos: Array3<Complex64>,
    pub(super) dkz_pos: Array3<Complex64>,
    pub(super) dkx_neg: Array3<Complex64>,
    pub(super) dky_neg: Array3<Complex64>,
    pub(super) dkz_neg: Array3<Complex64>,
}

impl StaggeredDerivativeOps {
    pub(super) fn build(
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Self {
        let make = |k: &Array3<f64>, d: f64, sign: f64| -> Array3<Complex64> {
            let [nx, ny, nz] = k.shape();
            let mut out = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
            for i in 0..nx {
                for j in 0..ny {
                    for l in 0..nz {
                        let kv = k[[i, j, l]];
                        let shift = Complex64::new(0.0, sign * kv * d * 0.5).exp();
                        out[[i, j, l]] = Complex64::new(0.0, kv) * shift;
                    }
                }
            }
            out
        };
        Self {
            dkx_pos: make(kx, dx, 1.0),
            dky_pos: make(ky, dy, 1.0),
            dkz_pos: make(kz, dz, 1.0),
            dkx_neg: make(kx, dx, -1.0),
            dky_neg: make(ky, dy, -1.0),
            dkz_neg: make(kz, dz, -1.0),
        }
    }
}
