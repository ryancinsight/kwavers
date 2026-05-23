//! Tests for stone fracture mechanics.

use super::material::StoneMaterial;
use super::model::StoneFractureModel;
use crate::core::constants::numerical::MPA_TO_PA;
use ndarray::Array3;

#[test]
fn test_calcium_oxalate_properties() {
    let stone = StoneMaterial::calcium_oxalate_monohydrate();

    assert_eq!(stone.density(), 2000.0);
    assert_eq!(stone.youngs_modulus(), 20e9);
    assert_eq!(stone.poisson_ratio(), 0.3);

    assert!(stone.elastic().p_wave_speed() > 0.0);
    assert!(stone.elastic().s_wave_speed() > 0.0);
    assert!(stone.elastic().s_wave_speed() < stone.elastic().p_wave_speed());

    assert_eq!(stone.tensile_strength(), 5e6);
    assert!(stone.strength().yield_strength < stone.strength().ultimate_strength);
    assert!(stone.strength().hardness > 0.0);
}

#[test]
fn test_material_composition() {
    let stone = StoneMaterial::calcium_oxalate_monohydrate();

    assert!(stone.elastic().density > 0.0);
    assert!(stone.strength().ultimate_strength > 0.0);

    assert_eq!(stone.density(), stone.elastic().density);
    assert_eq!(stone.youngs_modulus(), stone.elastic().youngs_modulus());
    assert_eq!(stone.tensile_strength(), stone.strength().ultimate_strength);
}

#[test]
fn test_stone_type_differences() {
    let oxalate = StoneMaterial::calcium_oxalate_monohydrate();
    let uric = StoneMaterial::uric_acid();
    let cystine = StoneMaterial::cystine();

    assert!(uric.youngs_modulus() < oxalate.youngs_modulus());
    assert!(uric.tensile_strength() < oxalate.tensile_strength());

    assert!(cystine.youngs_modulus() > oxalate.youngs_modulus());
    assert!(cystine.tensile_strength() > oxalate.tensile_strength());
}

#[test]
fn test_fracture_model_initialization() {
    let stone = StoneMaterial::calcium_oxalate_monohydrate();
    let model = StoneFractureModel::new(stone, (10, 10, 10));

    assert_eq!(model.damage_field().dim(), (10, 10, 10));
    assert_eq!(model.damage_field().sum(), 0.0);
    assert!(model.fragment_sizes().is_empty());
}

#[test]
fn test_damage_accumulation() {
    let stone = StoneMaterial::calcium_oxalate_monohydrate();
    let mut model = StoneFractureModel::new(stone.clone(), (5, 5, 5));

    let mut stress_field = Array3::zeros((5, 5, 5));
    let threshold = stone.tensile_strength();
    stress_field[[2, 2, 2]] = -2.0 * threshold;

    model.apply_stress_loading(&stress_field, 1e-6, MPA_TO_PA);

    assert!(model.damage_field()[[2, 2, 2]] > 0.0);
    assert!(model.damage_field()[[2, 2, 2]] <= 1.0);
    assert_eq!(model.damage_field()[[0, 0, 0]], 0.0);
}

#[test]
fn test_damage_saturation() {
    let stone = StoneMaterial::calcium_oxalate_monohydrate();
    let mut model = StoneFractureModel::new(stone.clone(), (3, 3, 3));

    let mut stress_field = Array3::zeros((3, 3, 3));
    stress_field[[1, 1, 1]] = -100.0 * stone.tensile_strength();

    for _ in 0..100 {
        model.apply_stress_loading(&stress_field, 1e-6, MPA_TO_PA);
    }

    assert_eq!(model.damage_field()[[1, 1, 1]], 1.0);
}

#[test]
fn test_no_damage_below_threshold() {
    let stone = StoneMaterial::calcium_oxalate_monohydrate();
    let mut model = StoneFractureModel::new(stone.clone(), (4, 4, 4));

    let mut stress_field = Array3::zeros((4, 4, 4));
    stress_field[[2, 2, 2]] = -0.5 * stone.tensile_strength();

    model.apply_stress_loading(&stress_field, 1e-6, MPA_TO_PA);

    assert_eq!(model.damage_field()[[2, 2, 2]], 0.0);
}

#[test]
fn test_material_accessor() {
    let stone = StoneMaterial::uric_acid();
    let model = StoneFractureModel::new(stone.clone(), (8, 8, 8));

    assert_eq!(model.material().density(), stone.density());
    assert_eq!(
        model.material().tensile_strength(),
        stone.tensile_strength()
    );
}
