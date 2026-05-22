#[cfg(test)]
mod tests {
    use super::super::helpers::*;
    use crate::core::constants::thermodynamic::THERMAL_DIFFUSIVITY_TISSUE;
    use ndarray::Array1;
    use std::f64::consts::PI;

    #[test]
    fn test_fd_laplacian_convergence_order() {
        let l = 1.0_f64; // domain length [m]
        let k_wave = 2.0 * PI / l; // wavenumber of manufactured solution

        let mut errors: Vec<f64> = Vec::new();
        for &n in &[16_usize, 32, 64, 128] {
            let dx = l / n as f64;
            // Manufactured solution u(xᵢ) = sin(2π xᵢ / L)
            let u: Array1<f64> = Array1::from_iter((0..n).map(|i| (k_wave * i as f64 * dx).sin()));
            // Analytical Laplacian: ∇²u = −k² sin(kx)
            let lap_exact: Array1<f64> = Array1::from_iter(
                (0..n).map(|i| -(k_wave * k_wave) * (k_wave * i as f64 * dx).sin()),
            );
            let lap_num = compute_laplacian_1d(&u, dx);

            // L∞ error on interior nodes (skip boundary ghost cells)
            let err = (1..n - 1)
                .map(|i| (lap_num[i] - lap_exact[i]).abs())
                .fold(0.0_f64, f64::max);
            errors.push(err);
        }

        // Richardson extrapolation: p̂ = log₂(E_n / E_{2n}) should be ≈ 2.0
        for w in errors.windows(2) {
            let order = (w[0] / w[1]).log2();
            assert!(
                order > 1.85,
                "FD Laplacian convergence order {:.4} < 1.85 (expected ≥ 2.0): \
                 errors = {:?}",
                order,
                errors
            );
        }
    }

    #[test]
    fn test_heat_equation_mms_convergence() {
        let alpha = THERMAL_DIFFUSIVITY_TISSUE; // thermal diffusivity [m²/s] (tissue, Duck 1990)
        let l = 1.0e-2_f64; // domain length [m] (1 cm)
        let k = 2.0 * PI / l; // wave number of manufactured solution
        let t_final = 5.0e-3_f64; // simulation time [s]

        let mut errors: Vec<f64> = Vec::new();
        for &n in &[16_usize, 32, 64, 128] {
            let dx = l / n as f64;
            // Fixed Fourier number r = α Δt / Δx² = 0.45 < 0.5 (stable)
            let dt = 0.45 * dx * dx / alpha;
            let n_steps = (t_final / dt).ceil() as usize;
            let dt_actual = t_final / n_steps as f64; // adjust dt to hit t_final exactly

            // Initial condition: T(x, 0) = sin(k x)
            let mut t_state: Vec<f64> = (0..n).map(|i| (k * i as f64 * dx).sin()).collect();

            for _ in 0..n_steps {
                let mut t_next = t_state.clone();
                for i in 0..n {
                    let ip1 = (i + 1) % n;
                    let im1 = (i + n - 1) % n;
                    let lap = (t_state[ip1] - 2.0 * t_state[i] + t_state[im1]) / (dx * dx);
                    t_next[i] = t_state[i] + dt_actual * alpha * lap;
                }
                t_state = t_next;
            }

            // Exact solution: T_exact(x, t_final) = exp(−α k² t_final) · sin(k x)
            let decay = (-alpha * k * k * t_final).exp();
            let err = (0..n)
                .map(|i| {
                    let t_exact = decay * (k * i as f64 * dx).sin();
                    (t_state[i] - t_exact).abs()
                })
                .fold(0.0_f64, f64::max);
            errors.push(err);
        }

        // Richardson extrapolation: p̂ = log₂(E_n / E_{2n}) ≥ 1.85
        for w in errors.windows(2) {
            let order = (w[0] / w[1]).log2();
            assert!(
                order > 1.85,
                "Heat equation convergence order {:.4} < 1.85 (expected ≥ 2.0): \
                 errors = {:?}",
                order,
                errors
            );
        }
    }
}
