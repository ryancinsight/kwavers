use super::material::{DamageParameters, MaterialProperties};
use super::model::CavitationDamage;
use crate::physics::bubble_dynamics::bubble_field::BubbleStateFields;

#[test]
fn test_damage_calculation() {
    let material = MaterialProperties::default();
    let params = DamageParameters::default();
    let mut damage = CavitationDamage::new((10, 10, 10), material, params);

    let mut states = BubbleStateFields::new((10, 10, 10));
    states.is_collapsing[[5, 5, 5]] = 1.0;
    states.radius[[5, 5, 5]] = 1e-6;
    states.velocity[[5, 5, 5]] = -100.0;
    states.pressure[[5, 5, 5]] = 1e9;

    damage.update_damage(&states, (1000.0, 1500.0), 1e-6);

    assert!(damage.damage_field[[5, 5, 5]] > 0.0);
    assert!(damage.impact_count[[5, 5, 5]] > 0);
}

#[test]
fn test_erosion_threshold() {
    let material = MaterialProperties::default();
    let params = DamageParameters::default();
    let damage = CavitationDamage::new((5, 5, 5), material.clone(), params);

    let rate = damage.calculate_erosion_rate(material.hardness * 0.5, 1e-6, 100.0);
    assert_eq!(rate, 0.0);

    let rate = damage.calculate_erosion_rate(material.hardness * 2.0, 1e-6, 100.0);
    assert!(rate > 0.0);
}
