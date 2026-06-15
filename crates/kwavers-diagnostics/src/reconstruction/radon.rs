//! Parallel-beam Radon transform and filtered backprojection (acoustic CT).
//!
//! Transmission acoustic CT measures the travel time of rays through the medium;
//! each travel time is the line integral of slowness `s(r) = 1/c(r)` — the
//! **Radon transform** of the slowness field. Filtered backprojection (FBP)
//! inverts it: ramp-filter each projection, then backproject. See Inverse
//! Problems §6 and ADR 013.
//!
//! Straight-ray (first-order) geometry; bent-ray correction is the iterative
//! SIRT path (`reconstruction::real_time_sirt`).
//!
//! # References
//! - Kak, A. C., & Slaney, M. (1988). *Principles of Computerized Tomographic
//!   Imaging*. IEEE Press. (Ram-Lak filter, §3.)
//! - Radon, J. (1917). "Über die Bestimmung von Funktionen durch ihre
//!   Integralwerte längs gewisser Mannigfaltigkeiten."

use ndarray::Array2;

const PI: f64 = std::f64::consts::PI;

/// Bilinear sample of `img` at continuous coordinates `(x, y)` (column, row);
/// returns 0 outside the grid.
#[inline]
fn sample(img: &Array2<f64>, x: f64, y: f64) -> f64 {
    let (ny, nx) = img.dim();
    if x < 0.0 || y < 0.0 || x > (nx - 1) as f64 || y > (ny - 1) as f64 {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let top = img[[y0, x0]] * (1.0 - fx) + img[[y0, x1]] * fx;
    let bot = img[[y1, x0]] * (1.0 - fx) + img[[y1, x1]] * fx;
    top * (1.0 - fy) + bot * fy
}

/// Forward parallel-beam Radon transform.
///
/// Returns a sinogram of shape `(n_angles, n_det)` where `n_det = max(nx, ny)`
/// detector bins, projections taken at angles uniformly spanning `[0, π)`.
#[must_use]
pub fn radon_transform(image: &Array2<f64>, n_angles: usize) -> Array2<f64> {
    let (ny, nx) = image.dim();
    if ny == 0 || nx == 0 || n_angles == 0 {
        return Array2::zeros((n_angles, nx.max(ny)));
    }
    let n_det = nx.max(ny);
    let cx = (nx - 1) as f64 / 2.0;
    let cy = (ny - 1) as f64 / 2.0;
    let half = n_det as f64 / 2.0;
    let n_samples = n_det; // along-ray samples

    let mut sino = Array2::<f64>::zeros((n_angles, n_det));
    for k in 0..n_angles {
        let theta = PI * k as f64 / n_angles as f64;
        let (st, ct) = theta.sin_cos();
        for td in 0..n_det {
            let u = td as f64 - half; // detector offset
            let mut acc = 0.0;
            for s in 0..n_samples {
                let v = s as f64 - half; // along-ray offset
                let x = cx + u * ct - v * st;
                let y = cy + u * st + v * ct;
                acc += sample(image, x, y);
            }
            sino[[k, td]] = acc;
        }
    }
    sino
}

/// Discrete Ram-Lak (ramp) filter impulse response of half-width `n`
/// (length `2n+1`, centred). Kak & Slaney (1988), unit sampling.
fn ram_lak_kernel(n: usize) -> Vec<f64> {
    let len = 2 * n + 1;
    let mut h = vec![0.0_f64; len];
    for (i, hv) in h.iter_mut().enumerate() {
        let m = i as i64 - n as i64;
        *hv = if m == 0 {
            0.25
        } else if m % 2 == 0 {
            0.0
        } else {
            -1.0 / (PI * PI * (m * m) as f64)
        };
    }
    h
}

/// Convolve one projection with the Ram-Lak kernel (`valid`-centred, same length).
fn ramp_filter(proj: &[f64]) -> Vec<f64> {
    let n = proj.len();
    let kernel = ram_lak_kernel(n);
    let kc = n; // kernel centre index
    let mut out = vec![0.0_f64; n];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &p) in proj.iter().enumerate() {
            let kidx = kc as i64 + i as i64 - j as i64;
            if kidx >= 0 && (kidx as usize) < kernel.len() {
                acc += p * kernel[kidx as usize];
            }
        }
        *o = acc;
    }
    out
}

/// Linear-interpolated read of `proj` at continuous detector index `t`.
#[inline]
fn interp(proj: &[f64], t: f64) -> f64 {
    if t < 0.0 || t > (proj.len() - 1) as f64 {
        return 0.0;
    }
    let i0 = t.floor() as usize;
    let i1 = (i0 + 1).min(proj.len() - 1);
    let f = t - i0 as f64;
    proj[i0] * (1.0 - f) + proj[i1] * f
}

/// Filtered backprojection inverse of [`radon_transform`].
///
/// `sinogram` has shape `(n_angles, n_det)`; returns the reconstructed image of
/// shape `(output_size, output_size)`.
#[must_use]
pub fn filtered_backprojection(sinogram: &Array2<f64>, output_size: usize) -> Array2<f64> {
    let (n_angles, n_det) = sinogram.dim();
    if n_angles == 0 || n_det == 0 || output_size == 0 {
        return Array2::zeros((output_size, output_size));
    }
    // ramp-filter each projection
    let mut filtered = Array2::<f64>::zeros((n_angles, n_det));
    for k in 0..n_angles {
        let row: Vec<f64> = (0..n_det).map(|j| sinogram[[k, j]]).collect();
        let f = ramp_filter(&row);
        for (j, &fv) in f.iter().enumerate() {
            filtered[[k, j]] = fv;
        }
    }

    let c = (output_size - 1) as f64 / 2.0;
    let half = n_det as f64 / 2.0;
    let scale = PI / n_angles as f64;
    let mut img = Array2::<f64>::zeros((output_size, output_size));
    for oy in 0..output_size {
        for ox in 0..output_size {
            let xc = ox as f64 - c;
            let yc = oy as f64 - c;
            let mut acc = 0.0;
            for k in 0..n_angles {
                let theta = PI * k as f64 / n_angles as f64;
                let (st, ct) = theta.sin_cos();
                let t = xc * ct + yc * st + half; // detector coordinate
                let row = filtered.row(k);
                acc += interp(row.as_slice().unwrap(), t);
            }
            img[[oy, ox]] = acc * scale;
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk(n: usize, cx: f64, cy: f64, r: f64) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((n, n));
        for y in 0..n {
            for x in 0..n {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if dx * dx + dy * dy <= r * r {
                    a[[y, x]] = 1.0;
                }
            }
        }
        a
    }

    fn positive_centroid(a: &Array2<f64>) -> (f64, f64) {
        let (ny, nx) = a.dim();
        let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
        for y in 0..ny {
            for x in 0..nx {
                let w = a[[y, x]].max(0.0);
                sx += w * x as f64;
                sy += w * y as f64;
                sw += w;
            }
        }
        (sx / sw, sy / sw)
    }

    #[test]
    fn fbp_round_trip_recovers_centred_disk() {
        let n = 64;
        let phantom = disk(n, 31.5, 31.5, 12.0);
        let sino = radon_transform(&phantom, 180);
        let recon = filtered_backprojection(&sino, n);
        let r = kwavers_math::statistics::pearson(
            &phantom.iter().copied().collect::<Vec<_>>(),
            &recon.iter().copied().collect::<Vec<_>>(),
        );
        assert!(r > 0.8, "FBP recon correlation {r} too low");
        // reconstructed mass centroid is at the image centre
        let (cx, cy) = positive_centroid(&recon);
        assert!(
            (cx - 31.5).abs() < 2.0 && (cy - 31.5).abs() < 2.0,
            "centroid ({cx},{cy})"
        );
    }

    #[test]
    fn fbp_localizes_off_centre_disk() {
        let n = 64;
        // disk in the upper-right quadrant (x large, y small)
        let phantom = disk(n, 44.0, 20.0, 7.0);
        let sino = radon_transform(&phantom, 180);
        let recon = filtered_backprojection(&sino, n);
        let (cx, cy) = positive_centroid(&recon);
        assert!(cx > 36.0, "expected centroid right of centre, got x={cx}");
        assert!(cy < 28.0, "expected centroid above centre, got y={cy}");
        assert!(
            (cx - 44.0).abs() < 4.0 && (cy - 20.0).abs() < 4.0,
            "centroid ({cx},{cy})"
        );
    }

    #[test]
    fn radon_of_empty_is_zero() {
        let sino = radon_transform(&Array2::<f64>::zeros((16, 16)), 30);
        assert!(sino.iter().all(|&v| v == 0.0));
        let recon = filtered_backprojection(&sino, 16);
        assert!(recon.iter().all(|&v| v.abs() < 1e-12));
    }
}
