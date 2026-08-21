use crate::acoustics::skull::AcousticSkullProperties;
use kwavers_core::error::KwaversResult;
use leto::Array3;

use super::constants::ALPHA_WATER;
use super::model::HeterogeneousSkull;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversError;

impl HeterogeneousSkull {
    /// Create a heterogeneous skull from CT data using the continuous
    /// tissue-varying `HuAcousticModel` (Schneider 1996) for density and sound
    /// speed. Attenuation blends linearly from water (`ALPHA_WATER`) to the
    /// provided bone attenuation at 1 MHz by bone volume fraction, so every
    /// tissue type maps to distinct properties (no binary bone/soft split).
    /// # Errors
    /// - Returns [`Err`] when the CT volume is empty or contains a non-finite
    ///   Hounsfield-unit value.
    ///
    pub fn from_ct(ct_data: &Array3<f64>, props: &AcousticSkullProperties) -> KwaversResult<Self> {
        use kwavers_core::constants::hu_mapping::HuAcousticModel;

        validate_ct_volume(ct_data)?;
        let model = HuAcousticModel::default();
        let bone_attenuation = props.attenuation_at_one_megahertz().into_base();
        let sound_speed = ct_data.mapv(|hu| model.sound_speed(hu));
        let density = ct_data.mapv(|hu| model.density(hu));
        let attenuation = ct_data.mapv(|hu| {
            let phi = model.bone_fraction(hu);
            (1.0 - phi).mul_add(ALPHA_WATER, phi * bone_attenuation)
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
    /// 5. Attenuation at 1 MHz: α_eff = φ·α_bone + (1−φ)·α_water.
    /// # Errors
    /// - Returns [`Err`] when the CT volume is empty or contains a non-finite
    ///   Hounsfield-unit value.
    ///
    pub fn from_ct_hill(
        ct_data: &Array3<f64>,
        bone: &AcousticSkullProperties,
    ) -> KwaversResult<Self> {
        validate_ct_volume(ct_data)?;

        let c_bone = bone.sound_speed().into_base();
        let rho_bone = bone.density().into_base();
        let alpha_bone = bone.attenuation_at_one_megahertz().into_base();
        let k_bone = rho_bone * c_bone * c_bone;
        let k_water = DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM * SOUND_SPEED_WATER_SIM;

        let sound_speed = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            if phi <= 0.0 {
                return SOUND_SPEED_WATER_SIM;
            }
            if phi >= 1.0 {
                return c_bone;
            }
            let rho_eff = phi.mul_add(rho_bone, (1.0 - phi) * DENSITY_WATER_NOMINAL);
            let k_voigt = phi.mul_add(k_bone, (1.0 - phi) * k_water);
            let k_reuss = 1.0 / (phi / k_bone + (1.0 - phi) / k_water);
            let k_hill = 0.5 * (k_voigt + k_reuss);
            (k_hill / rho_eff).sqrt()
        });

        let density = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            phi.mul_add(rho_bone, (1.0 - phi) * DENSITY_WATER_NOMINAL)
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

fn validate_ct_volume(ct_data: &Array3<f64>) -> KwaversResult<()> {
    if ct_data.is_empty() {
        return Err(KwaversError::InvalidInput(
            "CT volume must contain at least one voxel".to_owned(),
        ));
    }
    if let Some((index, hu)) = ct_data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, hu)| !hu.is_finite())
    {
        return Err(KwaversError::InvalidInput(format!(
            "CT Hounsfield units must be finite; voxel {index} contains {hu}"
        )));
    }
    Ok(())
}
