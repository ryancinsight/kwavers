use super::super::wave_field::NonlinearElasticWaveField;
use super::NonlinearElasticWaveSolver;

impl NonlinearElasticWaveSolver {
    /// Generate harmonic components using Chen (2013) harmonic motion detection.
    ///
    /// ## Theory
    /// Nonlinear wave equation with quadratic nonlinearity:
    /// `∂²u/∂t² = c²∇²u + β u ∇²u`
    ///
    /// Perturbation expansion: `u = u₁ + u₂ + u₃ + …`
    /// - `u₂` satisfies: `∂²u₂/∂t² − c²∇²u₂ = β u₁ ∇²u₁`
    /// - `u₃` satisfies: `∂²u₃/∂t² − c²∇²u₃ = β(u₁∇²u₂ + u₂∇²u₁ + 2∇u₁·∇u₂)`
    /// - Harmonic amplitudes: `Aₙ ∝ β^(n-1) / n`
    ///
    /// Reference: Chen, S., et al. (2013). IEEE Trans. Med. Imaging, 32(5), 863-874.
    pub(super) fn generate_harmonics(&self, field: &mut NonlinearElasticWaveField, dt: f64) {
        let beta = self.config.nonlinearity_parameter;
        let (nx, ny, nz) = self.grid.dimensions();

        // Second harmonic generation proportional to fundamental squared
        let harmonic_factor = (beta * 1e-6).min(1e-8);

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let u1 = field.u_fundamental[[i, j, k]];
                    let second_harmonic_amplitude = harmonic_factor * u1 * u1.abs();
                    field.u_second[[i, j, k]] += second_harmonic_amplitude * dt;
                }
            }
        }

        // Third harmonic via cascading (Chen 2013, Section III, Eq. 12)
        if !field.u_harmonics.is_empty() {
            for k in 1..nz - 1 {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let u1 = field.u_fundamental[[i, j, k]];
                        let u2 = field.u_second[[i, j, k]];

                        let laplacian_u1 = self.numerics.laplacian(i, j, k, &field.u_fundamental);
                        let laplacian_u2 = self.numerics.laplacian(i, j, k, &field.u_second);

                        let term1 = u1 * laplacian_u2;
                        let term2 = u2 * laplacian_u1;
                        let term3 = 2.0
                            * self.numerics.divergence_product(
                                i,
                                j,
                                k,
                                &field.u_fundamental,
                                &field.u_second,
                            );

                        let third_harmonic_source = beta * (term1 + term2 + term3);
                        let laplacian_u3 = self.numerics.laplacian(i, j, k, &field.u_harmonics[0]);
                        let acceleration_u3 = self.config.sound_speed().powi(2).mul_add(laplacian_u3, third_harmonic_source);
                        field.u_harmonics[0][[i, j, k]] += dt * dt * acceleration_u3;
                    }
                }
            }
        }

        // Fourth and higher harmonics through continued cascading
        for harmonic_idx in 1..field.u_harmonics.len() {
            let harmonic_order = harmonic_idx + 3;
            let amplitude_factor = beta.powi(harmonic_order as i32 - 1) / harmonic_order as f64;

            for k in 1..nz - 1 {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let u1 = field.u_fundamental[[i, j, k]];
                        let u_prev = if harmonic_idx == 1 {
                            field.u_second[[i, j, k]]
                        } else {
                            field.u_harmonics[harmonic_idx - 1][[i, j, k]]
                        };

                        let laplacian_u1 = self.numerics.laplacian(i, j, k, &field.u_fundamental);
                        let laplacian_u_prev = if harmonic_idx == 1 {
                            self.numerics.laplacian(i, j, k, &field.u_second)
                        } else {
                            self.numerics
                                .laplacian(i, j, k, &field.u_harmonics[harmonic_idx - 1])
                        };

                        let higher_harmonic_source = amplitude_factor
                            * beta
                            * u1.mul_add(laplacian_u_prev, u_prev * laplacian_u1);

                        let laplacian_u_n =
                            self.numerics
                                .laplacian(i, j, k, &field.u_harmonics[harmonic_idx]);
                        let acceleration_u_n = self.config.sound_speed().powi(2).mul_add(laplacian_u_n, higher_harmonic_source);

                        field.u_harmonics[harmonic_idx][[i, j, k]] += dt * dt * acceleration_u_n;
                    }
                }
            }
        }
    }
}
