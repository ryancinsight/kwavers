//! Typed volumetric heat-source field for the deposition boundary.
//!
//! Every energy-transport modality — acoustic absorption, optical deposition,
//! radiofrequency dissipation — deposits a volumetric power density `Q`. The
//! bioheat equation consumes that one quantity and divides by the *local*
//! `ρ c_p`, so no modality re-derives the conversion and none of them can
//! disagree about which `ρ c_p` applies in a heterogeneous medium.
//!
//! See Atlas ADR 0032 §5 for the deposition spine this type anchors.

use aequitas::systems::si::dimensions;
use kwavers_core::units::DimensionedField;
use leto::ArrayView3;

/// Borrowed volumetric heat-source field `Q`, in watts per cubic metre.
///
/// A [`DimensionedField`] over a borrowed view, so passing one costs exactly
/// what passing the view costs and the unit lives in the type rather than a
/// doc comment.
pub type VolumetricHeatSource<'a> =
    DimensionedField<ArrayView3<'a, f64>, dimensions::VolumetricPowerDensity>;

#[cfg(test)]
mod tests {
    use super::VolumetricHeatSource;
    use aequitas::systems::si::units::WattPerCubicMeter;
    use leto::Array3;

    #[test]
    fn samples_carry_their_volumetric_power_density_unit() {
        let field = Array3::from_elem((2, 2, 2), 1_500.0_f64);
        let source = VolumetricHeatSource::from_base(field.view());

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
