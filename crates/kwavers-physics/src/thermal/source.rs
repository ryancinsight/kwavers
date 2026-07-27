//! Typed volumetric heat-source field for the deposition boundary.
//!
//! Every energy-transport modality — acoustic absorption, optical deposition,
//! radiofrequency dissipation — deposits a volumetric power density `Q`. The
//! bioheat equation consumes that one quantity and divides by the *local*
//! `ρ c_p`, so no modality re-derives the conversion and none of them can
//! disagree about which `ρ c_p` applies in a heterogeneous medium.
//!
//! See Atlas ADR 0032 §5 for the deposition spine this type anchors.

use aequitas::systems::si::quantities::VolumetricPowerDensity;
use aequitas::systems::si::units::WattPerCubicMeter;
use core::fmt;
use leto::ArrayView3;

/// Borrowed volumetric heat-source field `Q`, in watts per cubic metre.
///
/// The wrapper carries the unit that a bare `ArrayView3<f64>` cannot. It is
/// `#[repr(transparent)]` over the view, so passing one costs exactly what
/// passing the view costs.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct VolumetricHeatSource<'a>(ArrayView3<'a, f64>);

impl fmt::Debug for VolumetricHeatSource<'_> {
    /// `leto::ArrayView3` has no `Debug`, so report the shape rather than the
    /// samples — a voxel dump is not diagnostic anyway.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VolumetricHeatSource")
            .field("unit", &"W/m³")
            .field("shape", &self.0.shape())
            .finish()
    }
}

impl<'a> VolumetricHeatSource<'a> {
    /// Adopt a field whose samples are already coherent SI `W/m³`.
    #[must_use]
    pub const fn from_watts_per_cubic_meter(field: ArrayView3<'a, f64>) -> Self {
        Self(field)
    }

    /// Borrow the underlying field.
    #[must_use]
    pub const fn as_view(&self) -> &ArrayView3<'a, f64> {
        &self.0
    }

    /// Sample the deposition at one voxel as a dimensional quantity.
    #[must_use]
    pub fn quantity_at(&self, index: [usize; 3]) -> VolumetricPowerDensity {
        VolumetricPowerDensity::from_unit::<WattPerCubicMeter>(self.0[index])
    }
}

#[cfg(test)]
mod tests {
    use super::VolumetricHeatSource;
    use aequitas::systems::si::units::WattPerCubicMeter;
    use leto::Array3;

    #[test]
    fn samples_carry_their_volumetric_power_density_unit() {
        let field = Array3::from_elem((2, 2, 2), 1_500.0_f64);
        let source = VolumetricHeatSource::from_watts_per_cubic_meter(field.view());

        assert_eq!(
            source
                .quantity_at([1, 1, 1])
                .in_unit::<WattPerCubicMeter>()
                .to_bits(),
            1_500.0_f64.to_bits()
        );
    }

    #[test]
    fn the_wrapper_adds_no_storage_over_the_borrowed_view() {
        assert_eq!(
            core::mem::size_of::<VolumetricHeatSource<'_>>(),
            core::mem::size_of::<leto::ArrayView3<'_, f64>>()
        );
    }
}
