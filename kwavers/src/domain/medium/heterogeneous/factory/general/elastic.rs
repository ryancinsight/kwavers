//! Elastic heterogeneous medium factory from wave speed arrays.

use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use crate::core::constants::thermodynamic::ROOM_TEMPERATURE_K;
use super::HeterogeneousFactory;
use crate::domain::medium::heterogeneous::core::HeterogeneousMedium;
use ndarray::{Array3, ArrayView3};

impl HeterogeneousFactory {
    /// Create a heterogeneous elastic medium from per-voxel wave speed and density arrays.
    ///
    /// Lamé parameters are computed per voxel from the isotropic elastic dispersion relations:
    ///   μ   = ρ · c_s²
    ///   λ   = ρ · (c_p² − 2 · c_s²)
    ///
    /// Stability (non-negative Poisson ratio) requires 2·c_s² ≤ c_p² at every voxel.
    /// A zero shear speed (`c_s = 0`) is valid and produces a fluid-like medium (μ = 0).
    ///
    /// # Arguments
    /// * `c_compression`  - 3D array of P-wave speeds (m/s)
    /// * `c_shear`        - 3D array of S-wave speeds (m/s); 0 for fluid voxels
    /// * `density`        - 3D array of densities [kg/m³]
    /// * `reference_frequency` - Reference frequency for any absorption (Hz)
    ///
    /// # Errors
    /// Returns an error if any array shape mismatches or if stability is violated.
    pub fn from_elastic_arrays(
        c_compression: ArrayView3<f64>,
        c_shear: ArrayView3<f64>,
        density: ArrayView3<f64>,
        reference_frequency: f64,
    ) -> Result<HeterogeneousMedium, String> {
        let shape = c_compression.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        macro_rules! check_shape {
            ($arr:expr, $name:expr) => {
                if $arr.shape() != shape {
                    return Err(format!(
                        "{} shape {:?} doesn't match c_compression shape {:?}",
                        $name,
                        $arr.shape(),
                        shape
                    ));
                }
            };
        }
        check_shape!(c_shear, "c_shear");
        check_shape!(density, "density");

        let mut lame_lambda = Array3::zeros((nx, ny, nz));
        let mut lame_mu = Array3::zeros((nx, ny, nz));
        let mut sound_speed = Array3::zeros((nx, ny, nz));
        let mut shear_sound_speed = Array3::zeros((nx, ny, nz));
        let mut density_arr = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let cp = c_compression[[i, j, k]];
                    let cs = c_shear[[i, j, k]];
                    let rho = density[[i, j, k]];

                    if rho <= 0.0 {
                        return Err(format!(
                            "density must be positive at voxel ({},{},{}): got {}",
                            i, j, k, rho
                        ));
                    }
                    if cp <= 0.0 {
                        return Err(format!(
                            "c_compression must be positive at voxel ({},{},{}): got {}",
                            i, j, k, cp
                        ));
                    }
                    if cs < 0.0 {
                        return Err(format!(
                            "c_shear must be non-negative at voxel ({},{},{}): got {}",
                            i, j, k, cs
                        ));
                    }
                    // Stability: λ ≥ 0 ↔ 2·c_s² ≤ c_p²
                    if 2.0 * cs * cs > cp * cp {
                        return Err(format!(
                            "Stability violated at voxel ({},{},{}): 2·c_s²={:.4} > c_p²={:.4}",
                            i,
                            j,
                            k,
                            2.0 * cs * cs,
                            cp * cp
                        ));
                    }

                    lame_mu[[i, j, k]] = rho * cs * cs;
                    lame_lambda[[i, j, k]] = rho * cp.mul_add(cp, -(2.0 * cs * cs));
                    sound_speed[[i, j, k]] = cp;
                    shear_sound_speed[[i, j, k]] = cs;
                    density_arr[[i, j, k]] = rho;
                }
            }
        }

        Ok(HeterogeneousMedium {
            use_trilinear_interpolation: true,
            density: density_arr,
            sound_speed,
            viscosity: Array3::zeros((nx, ny, nz)),
            surface_tension: Array3::zeros((nx, ny, nz)),
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            vapor_pressure: Array3::zeros((nx, ny, nz)),
            polytropic_index: Array3::zeros((nx, ny, nz)),
            specific_heat: Array3::zeros((nx, ny, nz)),
            thermal_conductivity: Array3::zeros((nx, ny, nz)),
            thermal_expansion: Array3::zeros((nx, ny, nz)),
            gas_diffusion_coeff: Array3::zeros((nx, ny, nz)),
            thermal_diffusivity: Array3::zeros((nx, ny, nz)),
            mu_a: Array3::zeros((nx, ny, nz)),
            mu_s_prime: Array3::zeros((nx, ny, nz)),
            temperature: Array3::from_elem((nx, ny, nz), ROOM_TEMPERATURE_K),
            bubble_radius: Array3::zeros((nx, ny, nz)),
            bubble_velocity: Array3::zeros((nx, ny, nz)),
            alpha0: Array3::zeros((nx, ny, nz)),
            delta: Array3::zeros((nx, ny, nz)),
            b_a: Array3::zeros((nx, ny, nz)),
            absorption: Array3::zeros((nx, ny, nz)),
            alpha_power: Array3::from_elem((nx, ny, nz), 1.0),
            nonlinearity: Array3::zeros((nx, ny, nz)),
            shear_sound_speed,
            shear_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            bulk_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            lame_lambda,
            lame_mu,
            reference_frequency,
        })
    }
}
