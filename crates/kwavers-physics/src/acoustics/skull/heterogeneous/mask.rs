use crate::parallel::zip_mut_two_refs;
use crate::acoustics::skull::AcousticSkullProperties;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use ndarray::Array3;

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

        let mut sound_speed = Array3::from_elem(mask.dim(), water_c);
        let mut density = Array3::from_elem(mask.dim(), water_rho);
        let mut attenuation = Array3::from_elem(mask.dim(), ALPHA_WATER);

        zip_mut_two_refs(
            sound_speed.view_mut(),
            mask.view(),
            mask.view(),
            |c, &m, _unused| {
                if m > 0.5 {
                    *c = props.sound_speed;
                }
            },
        );

        zip_mut_two_refs(
            density.view_mut(),
            mask.view(),
            mask.view(),
            |rho, _unused, &m| {
                if m > 0.5 {
                    *rho = props.density;
                }
            },
        );

        zip_mut_two_refs(
            attenuation.view_mut(),
            mask.view(),
            mask.view(),
            |atten, _unused, &m| {
                if m > 0.5 {
                    *atten = props.attenuation_coeff;
                }
            },
        );

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
