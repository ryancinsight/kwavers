//! Plug-and-Play (PnP) priors for iterative reconstruction.
//!
//! Modern MRI (compressed sensing; Lustig et al. 2007) and CT (model-based
//! iterative reconstruction, MBIR) reconstruct by alternating a data-fidelity
//! step with an edge-preserving **denoiser** applied to the image estimate — the
//! Plug-and-Play paradigm (Venkatakrishnan et al. 2013; Romano et al. 2017, RED).
//! The same idea transfers to ultrasound FWI: between FWI gradient updates, denoise
//! the model with a TV proximal operator to suppress streak/checkerboard artefacts
//! while preserving tissue boundaries.
//!
//! This module provides the canonical PnP prior: the **ROF total-variation
//! denoiser** solved by Chambolle's (2004) dual projection algorithm, the
//! proximal map of `λ·TV`:
//! ```text
//! prox_{λ TV}(f) = argmin_u  ½‖u − f‖² + λ·TV(u).
//! ```

use leto::Array3;

/// Edge-preserving TV denoiser (Chambolle 2004), applied in-plane to every
/// `z`-slice of a 3-D field. This is the proximal operator of `λ·TV` and the
/// canonical Plug-and-Play / compressed-sensing prior.
///
/// - `image`: the field to denoise (e.g. an FWI sound-speed estimate).
/// - `weight` (`λ ≥ 0`): TV strength; larger ⇒ flatter (more denoising). `0`
///   returns a copy.
/// - `iterations`: dual-ascent steps (20–100 typical).
/// - `frozen`: optional mask; `true` voxels are left exactly equal to `image`
///   (e.g. a known, fixed skull) and excluded from the difference stencils so the
///   high-contrast boundary is not smeared into the reconstructed region.
///
/// # Panics
/// - Panics if `frozen` is supplied with a shape differing from `image`.
#[must_use]
pub fn tv_denoise_chambolle(
    image: &Array3<f64>,
    weight: f64,
    iterations: usize,
    frozen: Option<&leto::Array3<bool>>,
) -> Array3<f64> {
    if let Some(m) = frozen {
        assert_eq!(
            m.shape(),
            image.shape(),
            "frozen mask must match image shape"
        );
    }
    if weight <= 0.0 {
        return image.clone();
    }
    let [nx, ny, nz] = image.shape();
    // Chambolle–Pock primal-dual for ROF (στ‖∇‖² ≤ 1, ‖∇‖² ≤ 8).
    let sigma = 0.35_f64;
    let tau = 0.35_f64;
    let active = |i: usize, j: usize, k: usize| frozen.is_none_or(|m| !m[[i, j, k]]);

    let mut out = image.clone();
    for k in 0..nz {
        let idx = |i: usize, j: usize| j * nx + i;
        let f = |i: usize, j: usize| image[[i, j, k]];
        let mut u = vec![0.0_f64; nx * ny];
        let mut ubar = vec![0.0_f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                u[idx(i, j)] = f(i, j);
                ubar[idx(i, j)] = f(i, j);
            }
        }
        let mut px = vec![0.0_f64; nx * ny];
        let mut py = vec![0.0_f64; nx * ny];

        for _ in 0..iterations {
            // Dual ascent: p ← proj_{‖·‖₂ ≤ weight}( p + σ ∇ū ).
            for j in 0..ny {
                for i in 0..nx {
                    if !active(i, j, k) {
                        continue;
                    }
                    let gx = if i + 1 < nx && active(i + 1, j, k) {
                        ubar[idx(i + 1, j)] - ubar[idx(i, j)]
                    } else {
                        0.0
                    };
                    let gy = if j + 1 < ny && active(i, j + 1, k) {
                        ubar[idx(i, j + 1)] - ubar[idx(i, j)]
                    } else {
                        0.0
                    };
                    let qx = px[idx(i, j)] + sigma * gx;
                    let qy = py[idx(i, j)] + sigma * gy;
                    let n = (qx * qx + qy * qy).sqrt();
                    let scale = if n > weight { weight / n } else { 1.0 };
                    px[idx(i, j)] = qx * scale;
                    py[idx(i, j)] = qy * scale;
                }
            }
            // Primal descent: u⁺ = prox_{τ·½‖·−f‖²}( u + τ·div p ) = (u + τ div p + τ f)/(1+τ).
            for j in 0..ny {
                for i in 0..nx {
                    if !active(i, j, k) {
                        continue;
                    }
                    let dpx = px[idx(i, j)]
                        - if i > 0 && active(i - 1, j, k) {
                            px[idx(i - 1, j)]
                        } else {
                            0.0
                        };
                    let dpy = py[idx(i, j)]
                        - if j > 0 && active(i, j - 1, k) {
                            py[idx(i, j - 1)]
                        } else {
                            0.0
                        };
                    let div = dpx + dpy;
                    let un = (u[idx(i, j)] + tau * div + tau * f(i, j)) / (1.0 + tau);
                    ubar[idx(i, j)] = 2.0 * un - u[idx(i, j)];
                    u[idx(i, j)] = un;
                }
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                if active(i, j, k) {
                    out[[i, j, k]] = u[idx(i, j)];
                }
            }
        }
    }
    let _ = 0usize; // placeholder kept in scope for future n-D generalisation
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array3;

    /// TV denoising of a noisy piecewise-constant image moves it closer to the
    /// clean image while preserving the central edge (a flat-region/edge test).
    #[test]
    fn tv_denoise_reduces_noise_and_preserves_edge() {
        let (nx, ny) = (40usize, 40);
        let mut clean = Array3::zeros([nx, ny, 1]);
        for j in 0..ny {
            for i in 0..nx {
                clean[[i, j, 0]] = if i < nx / 2 { 1.0 } else { 2.0 }; // step edge
            }
        }
        // Deterministic pseudo-noise.
        let mut lcg: u64 = 0x1234_5678_9abc_def1;
        let mut noisy = clean.clone();
        for v in noisy.iter_mut() {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Zero-mean uniform noise in [−0.25, 0.25].
            let u = 2.0 * (((lcg >> 33) as f64) / ((1u64 << 31) as f64)) - 1.0;
            *v += 0.25 * u;
        }

        let err = |a: &Array3<f64>| {
            a.iter()
                .zip(clean.iter())
                .map(|(x, c)| {
                    let d = x - c;
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        };
        let noisy_err = err(&noisy);
        let den = tv_denoise_chambolle(&noisy, 0.4, 200, None);
        let den_err = err(&den);
        assert!(
            den_err < 0.6 * noisy_err,
            "TV denoise must reduce error vs clean: noisy={noisy_err:.4}, denoised={den_err:.4}"
        );
        // Edge contrast preserved: mean(right half) − mean(left half) ≈ 1.0.
        let mean = |lo: usize, hi: usize| -> f64 {
            let mut s = 0.0;
            let mut n = 0;
            for j in 5..ny - 5 {
                for i in lo..hi {
                    s += den[[i, j, 0]];
                    n += 1;
                }
            }
            s / n as f64
        };
        let contrast = mean(nx / 2 + 4, nx - 4) - mean(4, nx / 2 - 4);
        assert!(
            (contrast - 1.0).abs() < 0.1,
            "edge contrast must be preserved (~1.0); got {contrast:.4}"
        );
    }

    /// `weight = 0` is a no-op; frozen voxels are left untouched.
    #[test]
    fn zero_weight_and_frozen_are_identity() {
        let mut img = Array3::zeros([8, 8, 1]);
        for (n, v) in img.iter_mut().enumerate() {
            *v = (n % 5) as f64;
        }
        assert_eq!(tv_denoise_chambolle(&img, 0.0, 10, None), img);

        let mut frozen = Array3::from_elem((8, 8, 1), false);
        frozen[[2, 2, 0]] = true;
        let out = tv_denoise_chambolle(&img, 0.2, 20, Some(&frozen));
        assert_eq!(
            out[[2, 2, 0]],
            img[[2, 2, 0]],
            "frozen voxel must be unchanged"
        );
    }
}
