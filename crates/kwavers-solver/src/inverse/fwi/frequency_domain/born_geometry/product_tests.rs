//! Exact spatial-product representation and requested-allocation budget.

use super::super::acquisition::RingAcquisition;
use super::super::born_kernel::{
    fields::{FactoredProducts, StoredProducts},
    integrate,
};
use super::*;
use aequitas::systems::si::quantities::Length;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;

fn geometry() -> BornGeometry {
    let array =
        MultiRowRingArray::new(4, 1, Length::from_base(0.02), Length::from_base(0.001)).unwrap();
    let acquisition = RingAcquisition::new(&array);
    BornGeometry::new([3, 2, 1], 180_000.0, &acquisition, &Config::default(), 4)
}

fn assert_bits(actual: Complex64, expected: Complex64) {
    assert_eq!(actual.re.to_bits(), expected.re.to_bits());
    assert_eq!(actual.im.to_bits(), expected.im.to_bits());
}

#[test]
fn spatial_products_obey_exact_budget_and_preserve_fields() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("spatial_product_budget").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let mut geometry = geometry();
    let required = 4
        * (2 * 4 * 6 * std::mem::size_of::<Complex64>()
            + std::mem::size_of::<super::products::ProductRow>());
    let weights: [_; 6] = std::array::from_fn(|lane| (lane + 1) as f64 / 16.0);
    drop(arrange);
    for budget in [0, required - 1, required, required + 1] {
        let retained = {
            let _act = tracing::info_span!("act", budget).entered();
            geometry.prepare_products(budget)
        };
        let _assert = tracing::info_span!("assert", budget).entered();
        assert_eq!(retained, if budget < required { 0 } else { required });
        for (transmit, row) in geometry.transmits.iter().enumerate() {
            for (receiver, &field) in row.receivers.iter().enumerate() {
                let green = &geometry.fields[field];
                let mut expected = Complex64::new(0.0, 0.0);
                for voxel in 0..6 {
                    expected += row.incident[voxel] * green[voxel] * weights[voxel];
                }
                let actual = if let Some(products) = &geometry.products {
                    let unscaled = products[transmit].fields::<false>(receiver);
                    let unit_scaled = products[transmit].fields::<true>(receiver);
                    assert_eq!(unscaled.len(), 6);
                    assert_eq!(unit_scaled.len(), 6);
                    for voxel in 0..6 {
                        assert_bits(unscaled[voxel], row.incident[voxel] * green[voxel]);
                        assert_bits(
                            unit_scaled[voxel],
                            (Complex64::new(1.0, 0.0) * row.incident[voxel]) * green[voxel],
                        );
                    }
                    integrate::<_, _, 4>(&StoredProducts::new(unscaled), &weights)
                } else {
                    integrate::<_, _, 4>(&FactoredProducts::new(&row.incident, green), &weights)
                };
                assert_bits(actual, expected);
            }
        }
    }
}

#[test]
fn nonfinite_products_retain_factored_ieee_propagation() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("nonfinite_spatial_products").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let mut geometry = geometry();
    geometry.transmits[0].incident[0] = Complex64::new(f64::NAN, 0.5);
    drop(arrange);
    let (retained, actual) = {
        let _act = tracing::info_span!("act").entered();
        let retained = geometry.prepare_products(8 * 1024 * 1024);
        let row = &geometry.transmits[0];
        let actual = integrate::<_, _, 4>(
            &FactoredProducts::new(&row.incident, &geometry.fields[row.receivers[0]]),
            &[0.5; 6],
        );
        (retained, actual)
    };
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(retained, 0);
    assert_eq!(geometry.products.as_ref().map(Vec::len), None);
    assert_eq!(actual.re.classify(), std::num::FpCategory::Nan);
    assert_eq!(actual.im.classify(), std::num::FpCategory::Nan);
}
