//! Acoustic-to-thermal coupling source terms.

use ndarray::Array3;

/// Compute volumetric heat source from acoustic absorption [W/m^3].
///
/// ## Formula
/// ```text
/// Q = 2α c e    where e = P²/(2ρc²) + ½ρ|v|²
/// ```
/// Expanding: `Q = α·P²/(ρc) + αρc|v|²`
#[must_use]
pub fn acoustic_heat_source(
    pressure: &leto::Array3<f64>,
    velocity_x: &leto::Array3<f64>,
    velocity_y: &leto::Array3<f64>,
    velocity_z: &leto::Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
) -> Array3<f64> {
    let s = pressure.shape();
    let (nx, ny, nz) = (s[0], s[1], s[2]);
    let mut q = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = pressure[[i, j, k]];
                let vx = velocity_x[[i, j, k]];
                let vy = velocity_y[[i, j, k]];
                let vz = velocity_z[[i, j, k]];
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                let alpha = absorption[[i, j, k]];
                if rho > 0.0 && c > 0.0 {
                    let v_sq = vz.mul_add(vz, vx.mul_add(vx, vy * vy));
                    let energy_density = (0.5 * rho)
                        .mul_add(v_sq, super::acoustic_potential_energy_density(p, rho, c));
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
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

    fn uniform_leto(s: [usize; 3], val: f64) -> leto::Array3<f64> {
        let mut arr = leto::Array3::<f64>::zeros(s);
        arr.fill(val);
        arr
    }

    fn uniform_ndarray(s: (usize, usize, usize), val: f64) -> Array3<f64> {
        Array3::from_elem(s, val)
    }

    /// Zero pressure and zero velocity → zero heat source at every cell.
    ///
    /// energy_density = 0 → q = 2·α·c·0 = 0 regardless of α.
    #[test]
    fn heat_source_zero_for_zero_acoustic_fields() {
        let s3 = [4, 4, 4];
        let s = (4, 4, 4);
        let zero = leto::Array3::<f64>::zeros(s3);
        let rho = uniform_ndarray(s, DENSITY_WATER_NOMINAL);
        let c = uniform_ndarray(s, SOUND_SPEED_WATER_SIM);
        let alpha = uniform_ndarray(s, 5.0);
        let q = acoustic_heat_source(&zero, &zero, &zero, &zero, &rho, &c, &alpha);
        assert!(
            q.iter().all(|&v| v == 0.0),
            "zero acoustic fields must give zero heat source"
        );
    }

    /// Zero absorption → zero heat source regardless of field amplitude.
    ///
    /// q = 2·α·c·e = 0 when α = 0.
    #[test]
    fn heat_source_zero_for_zero_absorption() {
        let s3 = [4, 4, 4];
        let s = (4, 4, 4);
        let p = uniform_leto(s3, 5000.0);
        let v = uniform_leto(s3, 0.1);
        let rho = uniform_ndarray(s, DENSITY_WATER_NOMINAL);
        let c = uniform_ndarray(s, SOUND_SPEED_WATER_SIM);
        let alpha = Array3::<f64>::zeros(s);
        let q = acoustic_heat_source(&p, &v, &v, &v, &rho, &c, &alpha);
        assert!(
            q.iter().all(|&v| v == 0.0),
            "zero absorption must give zero heat source"
        );
    }

    /// q = α·P²/(ρ·c) for pressure-only field (velocity = 0).
    ///
    /// Derivation: energy_density = P²/(2ρc²), q = 2·α·c·P²/(2ρc²) = α·P²/(ρ·c).
    #[test]
    fn heat_source_matches_formula_pressure_only_field() {
        let s3 = [4, 4, 4];
        let s = (4, 4, 4);
        let p0 = 2000.0_f64;
        let rho0 = DENSITY_WATER_NOMINAL;
        let c0 = SOUND_SPEED_WATER_SIM;
        let a0 = 3.0_f64;
        let p = uniform_leto(s3, p0);
        let v = leto::Array3::<f64>::zeros(s3);
        let rho = uniform_ndarray(s, rho0);
        let c = uniform_ndarray(s, c0);
        let alpha = uniform_ndarray(s, a0);

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
        let s3 = [4, 4, 4];
        let s = (4, 4, 4);
        let p = uniform_leto(s3, 1000.0); // 1000 Pa test pressure
        let v = uniform_leto(s3, 0.5);
        let rho = uniform_ndarray(s, DENSITY_WATER_NOMINAL);
        let c = uniform_ndarray(s, SOUND_SPEED_WATER_SIM);
        let alpha = uniform_ndarray(s, 2.0);
        let q = acoustic_heat_source(&p, &v, &v, &v, &rho, &c, &alpha);
        assert!(
            q.iter().all(|&qv| qv >= 0.0),
            "heat source must be non-negative for physical inputs"
        );
    }
}
