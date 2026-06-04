//! Acoustic intensity and power flux.

use kwavers_grid::Grid;
use ndarray::{Array3, Zip};

/// Compute acoustic intensity vector field `I = p v` [W/m^2].
#[must_use]
pub fn acoustic_intensity(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let ix = Zip::from(pressure)
        .and(velocity_x)
        .map_collect(|&p, &vx| p * vx);
    let iy = Zip::from(pressure)
        .and(velocity_y)
        .map_collect(|&p, &vy| p * vy);
    let iz = Zip::from(pressure)
        .and(velocity_z)
        .map_collect(|&p, &vz| p * vz);
    (ix, iy, iz)
}

/// Compute total acoustic power through a z-plane (W).
#[must_use]
pub fn acoustic_power_through_z_plane(
    pressure: &Array3<f64>,
    velocity_z: &Array3<f64>,
    k_plane: usize,
    grid: &Grid,
) -> f64 {
    let da = grid.dx * grid.dy;
    let mut power = 0.0_f64;
    let (nx, ny, _) = pressure.dim();
    for i in 0..nx {
        for j in 0..ny {
            power += pressure[[i, j, k_plane]] * velocity_z[[i, j, k_plane]] * da;
        }
    }
    power
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::Grid;
    use ndarray::Array3;

    fn uniform(val: f64) -> Array3<f64> {
        Array3::from_elem((4, 4, 4), val)
    }

    /// `acoustic_intensity`: Ix = p·vx, Iy = p·vy, Iz = p·vz (element-wise).
    ///
    /// At p=3, vx=2, vy=0, vz=0: Ix=6, Iy=0, Iz=0.
    #[test]
    fn acoustic_intensity_matches_pv_product() {
        let p = uniform(3.0);
        let vx = uniform(2.0);
        let vy = Array3::zeros((4, 4, 4));
        let vz = Array3::zeros((4, 4, 4));

        let (ix, iy, iz) = acoustic_intensity(&p, &vx, &vy, &vz);

        for &v in ix.iter() {
            assert!((v - 6.0).abs() < 1e-14, "Ix must be p·vx = 6 (got {v:.3e})");
        }
        for &v in iy.iter() {
            assert_eq!(v, 0.0, "Iy must be 0");
        }
        for &v in iz.iter() {
            assert_eq!(v, 0.0, "Iz must be 0");
        }
    }

    /// Zero pressure → all intensity components zero.
    #[test]
    fn acoustic_intensity_zero_for_zero_pressure() {
        let zero = Array3::<f64>::zeros((4, 4, 4));
        let vel = uniform(5.0);
        let (ix, iy, iz) = acoustic_intensity(&zero, &vel, &vel, &vel);
        assert!(ix.iter().all(|&v| v == 0.0));
        assert!(iy.iter().all(|&v| v == 0.0));
        assert!(iz.iter().all(|&v| v == 0.0));
    }

    /// `acoustic_power_through_z_plane` = Σ p·vz·dA over nx×ny cells.
    ///
    /// At uniform p=P, vz=V: power = P·V·nx·ny·dx·dy.
    #[test]
    fn acoustic_power_through_z_plane_matches_formula() {
        let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
        let p = uniform(200.0);
        let vz = uniform(0.5);

        let power = acoustic_power_through_z_plane(&p, &vz, 2, &grid);
        let da = grid.dx * grid.dy;
        let expected = 200.0 * 0.5 * (grid.nx * grid.ny) as f64 * da;
        assert!(
            (power - expected).abs() / expected < 1e-12,
            "power must equal P·V·Nx·Ny·dA (got {power:.6e}, expected {expected:.6e})"
        );
    }
}
