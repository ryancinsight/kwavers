use num_complex::Complex64;
use std::f64::consts::PI;

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

    let mut direct = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let k_signed = if k <= n / 2 {
            k as i64
        } else {
            k as i64 - n as i64
        };
        for (&x, &v) in positions.iter().zip(values.iter()) {
            let angle = -2.0 * PI * k_signed as f64 * x / l;
            direct[k] += v * Complex64::new(angle.cos(), angle.sin());
        }
    }

    // Sweep from 0.40 to 0.75 in 0.01 increments
    for i in 40..=75 {
        let tau = i as f64 / 100.0;
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

        let mut fast = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            let k_signed = if k <= n / 2 {
                k as i64
            } else {
                k as i64 - n as i64
            };
            let ks = k_signed as f64;
            let m_f = m as f64;
            let mut sum = Complex64::new(0.0, 0.0);
            for mm in 0..m {
                let angle = -2.0 * PI * ks * (mm as f64) / m_f;
                sum += grid[mm] * Complex64::new(angle.cos(), angle.sin());
            }
            let g_hat =
                (4.0 * PI * tau).sqrt() * (-4.0 * PI * PI * tau * ks * ks / (m_f * m_f)).exp();
            fast[k] = sum / g_hat;
        }

        let mut max_rel_err = 0.0_f64;
        for k in 0..n {
            let rel = (fast[k] - direct[k]).norm() / (direct[k].norm().max(1e-30));
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
        println!("tau={:.3}: max rel err = {:.3e}", tau, max_rel_err);
    }
}
