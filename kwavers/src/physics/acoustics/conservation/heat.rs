//! Acoustic-to-thermal coupling source terms.

use ndarray::Array3;

/// Compute volumetric heat source from acoustic absorption [W/m^3].
#[must_use]
pub fn acoustic_heat_source(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
) -> Array3<f64> {
    let (nx, ny, nz) = pressure.dim();
    let mut q = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                if rho > 0.0 && c > 0.0 {
                    let p = pressure[[i, j, k]];
                    let vx = velocity_x[[i, j, k]];
                    let vy = velocity_y[[i, j, k]];
                    let vz = velocity_z[[i, j, k]];
                    let alpha = absorption[[i, j, k]];
                    let energy_density =
                        (0.5 * rho).mul_add(vz.mul_add(vz, vx.mul_add(vx, vy * vy)), p * p / (2.0 * rho * c * c));
                    q[[i, j, k]] = 2.0 * alpha * c * energy_density;
                }
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn uniform(s: (usize, usize, usize), val: f64) -> Array3<f64> {
        Array3::from_elem(s, val)
    }

    /// Zero pressure and zero velocity → zero heat source at every cell.
    ///
    /// energy_density = 0 → q = 2·α·c·0 = 0 regardless of α.
    #[test]
    fn heat_source_zero_for_zero_acoustic_fields() {
        let s = (4, 4, 4);
        let zero  = Array3::<f64>::zeros(s);
        let rho   = uniform(s, 1000.0);
        let c     = uniform(s, 1500.0);
        let alpha = uniform(s, 5.0);
        let q = acoustic_heat_source(&zero, &zero, &zero, &zero, &rho, &c, &alpha);
        assert!(q.iter().all(|&v| v == 0.0), "zero acoustic fields must give zero heat source");
    }

    /// Zero absorption → zero heat source regardless of field amplitude.
    ///
    /// q = 2·α·c·e = 0 when α = 0.
    #[test]
    fn heat_source_zero_for_zero_absorption() {
        let s = (4, 4, 4);
        let p     = uniform(s, 5000.0);
        let v     = uniform(s, 0.1);
        let rho   = uniform(s, 1000.0);
        let c     = uniform(s, 1500.0);
        let alpha = Array3::<f64>::zeros(s);
        let q = acoustic_heat_source(&p, &v, &v, &v, &rho, &c, &alpha);
        assert!(q.iter().all(|&v| v == 0.0), "zero absorption must give zero heat source");
    }

    /// q = α·P²/(ρ·c) for pressure-only field (velocity = 0).
    ///
    /// Derivation: energy_density = P²/(2ρc²), q = 2·α·c·P²/(2ρc²) = α·P²/(ρ·c).
    #[test]
    fn heat_source_matches_formula_pressure_only_field() {
        let s = (4, 4, 4);
        let p0    = 2000.0_f64;
        let rho0  = 1000.0_f64;
        let c0    = 1500.0_f64;
        let a0    = 3.0_f64;
        let p     = uniform(s, p0);
        let v     = Array3::<f64>::zeros(s);
        let rho   = uniform(s, rho0);
        let c     = uniform(s, c0);
        let alpha = uniform(s, a0);

        let q = acoustic_heat_source(&p, &v, &v, &v, &rho, &c, &alpha);

        let expected = a0 * p0 * p0 / (rho0 * c0); // α·P²/(ρ·c)
        for &qv in q.iter() {
            assert!(
                (qv - expected).abs() / expected < 1e-12,
                "q must equal α·P²/(ρ·c)={expected:.6e} (got {qv:.6e})"
            );
        }
    }

    /// Heat source is non-negative for physical inputs (α≥0, ρ>0, c>0, any p/v).
    #[test]
    fn heat_source_nonnegative_for_physical_fields() {
        let s = (4, 4, 4);
        let p     = uniform(s, 1000.0);
        let v     = uniform(s, 0.5);
        let rho   = uniform(s, 1000.0);
        let c     = uniform(s, 1500.0);
        let alpha = uniform(s, 2.0);
        let q = acoustic_heat_source(&p, &v, &v, &v, &rho, &c, &alpha);
        assert!(
            q.iter().all(|&qv| qv >= 0.0),
            "heat source must be non-negative for physical inputs"
        );
    }
}
