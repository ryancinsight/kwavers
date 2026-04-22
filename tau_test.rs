use std::f64::consts::PI;
use num_complex::Complex64;
// We need to write a little nufft_test here with real Rust code.
// Instead of rustfft, we will just use O(N^2) direct evaluation to find the true error
// for given tau, to avoid adding dependencies.

fn main() {
    let n = 64usize;
    let l = 1.5_f64;
    let sigma = 2;
    let w_max = 6i64;
    let m = sigma * n;

    let positions: Vec<f64> = (0..50)
        .map(|i| ((i as f64 * 2.0_f64.sqrt() + 0.1) % l).abs())
        .collect();
    let values: Vec<Complex64> = (0..50)
        .map(|i| Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.17).sin()))
        .collect();

    // Direct result
    let mut direct = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let k_signed = if k <= n / 2 { k as i64 } else { k as i64 - n as i64 };
        for (&x, &v) in positions.iter().zip(values.iter()) {
            let angle = -2.0 * PI * k_signed as f64 * x / l;
            direct[k] += v * Complex64::new(angle.cos(), angle.sin());
        }
    }

    // Sweep tau
    for &tau in &[0.2, 0.3, 0.4, 0.477, 0.5, 0.6, 0.675, 0.7, 0.8, 0.9, 1.0] {
        let mut grid = vec![Complex64::new(0.0, 0.0); m];
        for (&x, &v) in positions.iter().zip(values.iter()) {
            let x_mod = x.rem_euclid(l);
            let t = m as f64 * x_mod / l;
            let m0 = t.round() as i64;
            let d = t - m0 as f64;
            for p in -w_max..=w_max {
                let weight = (-(p as f64 - d).powi(2) / (4.0 * tau)).exp();
                let m_idx = (m0 + p).rem_euclid(m as i64) as usize;
                grid[m_idx] += v * weight;
            }
        }

        // Exact DFT of grid
        let mut fast = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            let k_signed = if k <= n / 2 { k as i64 } else { k as i64 - n as i64 };
            let ks = k_signed as f64;
            let m_f = m as f64;

            // Compute DFT component k of grid
            let mut sum = Complex64::new(0.0, 0.0);
            for mm in 0..m {
                let angle = -2.0 * PI * ks * (mm as f64) / m_f;
                sum += grid[mm] * Complex64::new(angle.cos(), angle.sin());
            }

            // Deconvolve
            let g_hat = (4.0 * PI * tau).sqrt() * (-4.0 * PI * PI * tau * ks * ks / (m_f * m_f)).exp();
            fast[k] = sum / g_hat;
        }

        let mut max_rel_err = 0.0_f64;
        for k in 0..n {
            let rel = (fast[k] - direct[k]).norm() / (direct[k].norm().max(1e-30));
            if rel > max_rel_err { max_rel_err = rel; }
        }
        println!("tau={:.3}: max rel err = {:.3e}", tau, max_rel_err);
    }
}
