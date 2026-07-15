use crate::acoustics::skull::AcousticSkullProperties;
use crate::parallel::zip_mut_ref;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;

use super::constants::ALPHA_WATER;
use super::model::HeterogeneousSkull;

impl HeterogeneousSkull {
    /// Create heterogeneous skull from binary mask and scalar skull properties.
    ///
    /// All voxels with `mask > 0.5` receive the provided skull properties;
    /// remaining voxels are assigned water properties.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn from_mask(
        _grid: &Grid,
        mask: &Array3<f64>,
        props: &AcousticSkullProperties,
    ) -> KwaversResult<Self> {
        use kwavers_core::constants::thermodynamic::ROOM_TEMPERATURE_C;
        use kwavers_core::constants::water::WaterProperties;

        let water_c = WaterProperties::sound_speed(ROOM_TEMPERATURE_C);
        let water_rho = WaterProperties::density(ROOM_TEMPERATURE_C);

        let mut sound_speed = Array3::from_elem(mask.shape(), water_c);
        let mut density = Array3::from_elem(mask.shape(), water_rho);
        let mut attenuation = Array3::from_elem(mask.shape(), ALPHA_WATER);

        zip_mut_ref(sound_speed.view_mut(), mask.view(), |c, &m| {
            if m > 0.5 {
                *c = props.sound_speed;
            }
        });

        zip_mut_ref(density.view_mut(), mask.view(), |rho, &m| {
            if m > 0.5 {
                *rho = props.density;
            }
        });

        zip_mut_ref(attenuation.view_mut(), mask.view(), |atten, &m| {
            if m > 0.5 {
                *atten = props.attenuation_coeff;
            }
        });

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Generate a binary mask from CT data (1.0 = bone, 0.0 = tissue).
    #[must_use]
    pub fn generate_mask_from_ct(ct_data: &Array3<f64>) -> Array3<f64> {
        ct_data.mapv(|hu| if hu > 700.0 { 1.0 } else { 0.0 })
    }
}
