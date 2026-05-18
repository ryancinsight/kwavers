use crate::core::error::KwaversResult;
use crate::physics::acoustics::skull::AcousticSkullProperties;
use ndarray::Array3;

use super::constants::{ALPHA_WATER, C_WATER, RHO_WATER};
use super::model::HeterogeneousSkull;

impl HeterogeneousSkull {
    /// Create heterogeneous skull from CT data using the legacy CTImageLoader
    /// pipeline. The binary threshold HU > 700 selects bone; attenuation uses
    /// the provided props value for bone voxels.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn from_ct(ct_data: &Array3<f64>, props: &AcousticSkullProperties) -> KwaversResult<Self> {
        use crate::domain::imaging::medical::CTImageLoader;

        let sound_speed = ct_data.mapv(CTImageLoader::hu_to_sound_speed);
        let density = ct_data.mapv(CTImageLoader::hu_to_density);
        let attenuation = ct_data.mapv(|hu| {
            if hu > 700.0 {
                props.attenuation_coeff
            } else {
                ALPHA_WATER
            }
        });

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Create heterogeneous skull from CT data using the Hill-averaged BVF
    /// mixing model.
    ///
    /// ## Algorithm
    /// 1. Compute BVF φ(HU).
    /// 2. Compute Voigt density: ρ_eff = φ·ρ_bone + (1−φ)·ρ_water.
    /// 3. Compute Hill modulus: K_H = (K_V + K_R) / 2.
    /// 4. Effective speed: c_eff = sqrt(K_H / ρ_eff).
    /// 5. Attenuation: α_eff = φ·α_bone + (1−φ)·α_water.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn from_ct_hill(
        ct_data: &Array3<f64>,
        c_bone: f64,
        rho_bone: f64,
        alpha_bone: f64,
    ) -> KwaversResult<Self> {
        let k_bone = rho_bone * c_bone * c_bone;
        let k_water = RHO_WATER * C_WATER * C_WATER;

        let sound_speed = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            if phi <= 0.0 {
                return C_WATER;
            }
            if phi >= 1.0 {
                return c_bone;
            }
            let rho_eff = phi.mul_add(rho_bone, (1.0 - phi) * RHO_WATER);
            let k_voigt = phi.mul_add(k_bone, (1.0 - phi) * k_water);
            let k_reuss = 1.0 / (phi / k_bone + (1.0 - phi) / k_water);
            let k_hill = 0.5 * (k_voigt + k_reuss);
            (k_hill / rho_eff).sqrt()
        });

        let density = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            phi.mul_add(rho_bone, (1.0 - phi) * RHO_WATER)
        });

        let attenuation = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            phi.mul_add(alpha_bone, (1.0 - phi) * ALPHA_WATER)
        });

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }
}
