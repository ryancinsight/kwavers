//! Electromagnetic specific-absorption-rate and deposition fields.
//!
//! For an isotropic conductive medium, the local Joule power density is
//! `q = σ |E|²` and the specific absorption rate is `SAR = q / ρ`. The field
//! storage remains a dense real Leto array; Aequitas tags the container so a
//! SAR field cannot be passed as a volumetric power-density field without an
//! explicit conversion. If a future solver supplies Eunomia complex phasors,
//! it must evaluate the Hermitian magnitude at that numerical boundary. SAR
//! and volumetric deposition remain real power metrics, so no imaginary unit is
//! introduced.

use aequitas::systems::si::dimensions;
use kwavers_core::units::DimensionedField;
use kwavers_field::EMFields;
use leto::{ArrayD, VecStorage};
use thiserror::Error;

use super::equations::EMMaterialDistribution;

/// A spatial field of specific absorption rate in `W/kg`.
pub type SpecificAbsorptionRateField =
    DimensionedField<ArrayD<f64, VecStorage<f64>>, dimensions::SpecificAbsorptionRate>;

/// A spatial field of volumetric electromagnetic power deposition in `W/m³`.
pub type VolumetricPowerDensityField =
    DimensionedField<ArrayD<f64, VecStorage<f64>>, dimensions::VolumetricPowerDensity>;

/// A typed electromagnetic deposition result.
#[derive(Debug, Clone)]
pub struct ElectromagneticDeposition {
    /// Specific absorption rate `σ·|E|²/ρ` in `W/kg`.
    pub specific_absorption_rate: SpecificAbsorptionRateField,
    /// Volumetric power density `σ·|E|²` in `W/m³`.
    pub volumetric_power_density: VolumetricPowerDensityField,
}

/// Failure while constructing an electromagnetic deposition result.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SpecificAbsorptionRateError {
    /// An input array shape does not match the expected spatial shape.
    #[error("{field} shape {actual:?} does not match expected {expected:?}")]
    ShapeMismatch {
        /// Input name.
        field: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Supplied shape.
        actual: Vec<usize>,
    },
    /// The electric field has no spatial/component split or has an unsupported
    /// number of vector components.
    #[error("electric field shape must end in 2 or 3 components, got {0:?}")]
    InvalidElectricFieldShape(Vec<usize>),
    /// A finite non-negative conductivity was required.
    #[error("conductivity sample {index} must be finite and non-negative, got {value}")]
    InvalidConductivity {
        /// Spatial sample index.
        index: usize,
        /// Conductivity value in `S/m`.
        value: f64,
    },
    /// A finite positive mass density was required.
    #[error("density sample {index} must be finite and positive, got {value}")]
    InvalidDensity {
        /// Spatial sample index.
        index: usize,
        /// Density value in `kg/m³`.
        value: f64,
    },
    /// A finite electric-field component was required.
    #[error("electric-field sample {index} must be finite, got {value}")]
    InvalidElectricField {
        /// Spatial sample index.
        index: usize,
        /// Electric-field component in `V/m`.
        value: f64,
    },
    /// The shape product could not be represented by `usize`.
    #[error("electric-field spatial shape product overflows usize")]
    SpatialSizeOverflow,
    /// Output construction failed after validated inputs were traversed.
    #[error("failed to construct deposition output: {0}")]
    OutputConstruction(String),
    /// The electromagnetic field aggregate is internally inconsistent.
    #[error("invalid electromagnetic fields: {0}")]
    InvalidFields(String),
}

/// Compute electromagnetic Joule heating and specific absorption rate.
///
/// The electric field uses a final component axis of length two or three. The
/// conductivity comes from the Aequitas-tagged material distribution, and the
/// density field must be tagged as Aequitas mass density. All samples are
/// canonical SI values at this formula boundary.
///
/// # Errors
///
/// Returns [`SpecificAbsorptionRateError`] when field shapes, finite-value
/// contracts, or density positivity are violated.
pub fn compute_electromagnetic_deposition(
    fields: &EMFields,
    materials: &EMMaterialDistribution,
    density: &DimensionedField<ArrayD<f64, VecStorage<f64>>, dimensions::MassDensity>,
) -> Result<ElectromagneticDeposition, SpecificAbsorptionRateError> {
    fields
        .validate_shapes()
        .map_err(SpecificAbsorptionRateError::InvalidFields)?;

    let electric_shape = fields.electric.shape();
    let Some(&components) = electric_shape.last() else {
        return Err(SpecificAbsorptionRateError::InvalidElectricFieldShape(
            electric_shape.to_vec(),
        ));
    };
    if electric_shape.len() < 2 || (components != 2 && components != 3) {
        return Err(SpecificAbsorptionRateError::InvalidElectricFieldShape(
            electric_shape.to_vec(),
        ));
    }

    let spatial_shape = electric_shape[..electric_shape.len() - 1].to_vec();
    let spatial_size = spatial_shape.iter().try_fold(1usize, |size, &extent| {
        size.checked_mul(extent)
            .ok_or(SpecificAbsorptionRateError::SpatialSizeOverflow)
    })?;

    let conductivity = materials.conductivity.samples();
    if conductivity.shape() != spatial_shape.as_slice() {
        return Err(SpecificAbsorptionRateError::ShapeMismatch {
            field: "conductivity",
            expected: spatial_shape.clone(),
            actual: conductivity.shape().to_vec(),
        });
    }
    let density_samples = density.samples();
    if density_samples.shape() != spatial_shape.as_slice() {
        return Err(SpecificAbsorptionRateError::ShapeMismatch {
            field: "density",
            expected: spatial_shape.clone(),
            actual: density_samples.shape().to_vec(),
        });
    }

    let mut electric_samples = fields.electric.iter().copied();
    let mut sar_values = Vec::with_capacity(spatial_size);
    let mut volumetric_values = Vec::with_capacity(spatial_size);

    for (index, (&conductivity_sample, &density_sample)) in
        conductivity.iter().zip(density_samples.iter()).enumerate()
    {
        if !conductivity_sample.is_finite() || conductivity_sample < 0.0 {
            return Err(SpecificAbsorptionRateError::InvalidConductivity {
                index,
                value: conductivity_sample,
            });
        }
        if !density_sample.is_finite() || density_sample <= 0.0 {
            return Err(SpecificAbsorptionRateError::InvalidDensity {
                index,
                value: density_sample,
            });
        }

        let mut electric_magnitude_squared = 0.0;
        for _ in 0..components {
            let Some(component) = electric_samples.next() else {
                return Err(SpecificAbsorptionRateError::InvalidElectricFieldShape(
                    electric_shape.to_vec(),
                ));
            };
            if !component.is_finite() {
                return Err(SpecificAbsorptionRateError::InvalidElectricField {
                    index,
                    value: component,
                });
            }
            electric_magnitude_squared = component.mul_add(component, electric_magnitude_squared);
        }

        let volumetric_power = conductivity_sample * electric_magnitude_squared;
        volumetric_values.push(volumetric_power);
        sar_values.push(volumetric_power / density_sample);
    }

    let sar = ArrayD::from_shape_vec(&spatial_shape, sar_values)
        .map_err(|error| SpecificAbsorptionRateError::OutputConstruction(error.to_string()))?;
    let volumetric_power_density = ArrayD::from_shape_vec(&spatial_shape, volumetric_values)
        .map_err(|error| SpecificAbsorptionRateError::OutputConstruction(error.to_string()))?;

    Ok(ElectromagneticDeposition {
        specific_absorption_rate: DimensionedField::from_base(sar),
        volumetric_power_density: DimensionedField::from_base(volumetric_power_density),
    })
}

/// Compute only the Aequitas-tagged specific-absorption-rate field.
///
/// # Errors
///
/// Propagates the validation errors from
/// [`compute_electromagnetic_deposition`].
pub fn compute_specific_absorption_rate(
    fields: &EMFields,
    materials: &EMMaterialDistribution,
    density: &DimensionedField<ArrayD<f64, VecStorage<f64>>, dimensions::MassDensity>,
) -> Result<SpecificAbsorptionRateField, SpecificAbsorptionRateError> {
    compute_electromagnetic_deposition(fields, materials, density)
        .map(|deposition| deposition.specific_absorption_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electromagnetic::equations::EMMaterialUtils;
    use aequitas::systems::si::{
        quantities::ElectricalConductivity,
        units::{KilogramPerCubicMeter, SiemensPerMeter, WattPerCubicMeter, WattPerKilogram},
    };
    use kwavers_core::units::DimensionedField;
    use kwavers_field::{ArrayD, VecStorage};

    fn fields(electric_values: &[f64]) -> EMFields {
        let electric =
            ArrayD::<f64, VecStorage<f64>>::from_shape_vec(&[2, 1, 2], electric_values.to_vec())
                .expect("test electric shape is valid");
        let magnetic = ArrayD::<f64, VecStorage<f64>>::zeros(&[2, 1, 2])
            .expect("test magnetic shape is valid");
        EMFields::new(electric, magnetic)
    }

    fn materials(conductivity: f64) -> EMMaterialDistribution {
        EMMaterialUtils::create_uniform_distribution(
            &[2, 1],
            kwavers_medium::properties::ElectromagneticPropertyData::new(
                1.0,
                1.0,
                ElectricalConductivity::from_unit::<SiemensPerMeter>(conductivity),
                None,
            )
            .expect("test material is valid"),
        )
    }

    fn density(
        values: &[f64],
    ) -> DimensionedField<ArrayD<f64, VecStorage<f64>>, dimensions::MassDensity> {
        DimensionedField::from_base(
            ArrayD::<f64, VecStorage<f64>>::from_shape_vec(&[2, 1], values.to_vec())
                .expect("test density shape is valid"),
        )
    }

    #[test]
    fn deposition_matches_joule_and_sar_laws() {
        let result = compute_electromagnetic_deposition(
            &fields(&[3.0, 4.0, 1.0, 2.0]),
            &materials(0.5),
            &density(&[1_000.0, 2_000.0]),
        )
        .expect("valid deposition inputs");

        assert_eq!(
            result
                .volumetric_power_density
                .quantity_at(&[0, 0])
                .in_unit::<WattPerCubicMeter>()
                .to_bits(),
            12.5_f64.to_bits()
        );
        assert_eq!(
            result
                .specific_absorption_rate
                .quantity_at(&[0, 0])
                .in_unit::<WattPerKilogram>()
                .to_bits(),
            0.0125_f64.to_bits()
        );
        assert_eq!(
            result
                .specific_absorption_rate
                .quantity_at(&[1, 0])
                .in_unit::<WattPerKilogram>()
                .to_bits(),
            0.00125_f64.to_bits()
        );
    }

    #[test]
    fn zero_conductivity_has_zero_deposition() {
        let result = compute_specific_absorption_rate(
            &fields(&[3.0, 4.0, 1.0, 2.0]),
            &materials(0.0),
            &density(&[1_000.0, 1_000.0]),
        )
        .expect("zero conductivity is valid");

        assert!(result.samples().iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn invalid_density_is_rejected_before_division() {
        let error = compute_specific_absorption_rate(
            &fields(&[1.0, 0.0, 0.0, 1.0]),
            &materials(0.5),
            &density(&[0.0, 1_000.0]),
        )
        .expect_err("zero density is invalid");

        assert_eq!(
            error,
            SpecificAbsorptionRateError::InvalidDensity {
                index: 0,
                value: 0.0,
            }
        );
    }

    #[test]
    fn density_field_is_tagged_without_changing_storage() {
        assert_eq!(
            core::mem::size_of::<
                DimensionedField<ArrayD<f64, VecStorage<f64>>, dimensions::MassDensity>,
            >(),
            core::mem::size_of::<ArrayD<f64, VecStorage<f64>>>()
        );
        let value = density(&[1_000.0, 1_000.0])
            .quantity_at(&[0, 0])
            .in_unit::<KilogramPerCubicMeter>();
        assert_eq!(value.to_bits(), 1_000.0_f64.to_bits());
    }
}
