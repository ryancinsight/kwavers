use super::field::AblationField;
use super::kinetics::AblationKinetics;
use super::state::AblationState;
use ndarray::Array3;

#[test]
fn test_kinetics_creation() {
    let kinetics = AblationKinetics::protein_denaturation();
    assert!(kinetics.frequency_factor > 0.0);
    assert!(kinetics.activation_energy > 0.0);
    assert_eq!(kinetics.damage_threshold, 1.0);
}

#[test]
fn test_damage_rate_temperature_dependence() {
    let kinetics = AblationKinetics::hifu_ablation();

    // Higher temperature should give higher damage rate
    let rate_43c = kinetics.damage_rate(273.15 + 43.0);
    let rate_50c = kinetics.damage_rate(273.15 + 50.0);
    let rate_70c = kinetics.damage_rate(273.15 + 70.0);

    assert!(rate_43c > 0.0);
    assert!(rate_50c > rate_43c); // 50°C > 43°C
    assert!(rate_70c > rate_50c); // 70°C > 50°C
}

#[test]
fn test_damage_accumulation() {
    let mut current_damage = 0.0;
    let damage_rate = 0.5; // 1/s
    let dt = 1.0; // 1 second

    for _ in 0..3 {
        current_damage = AblationKinetics::accumulated_damage(current_damage, damage_rate, dt);
    }

    // After 3 seconds at 0.5 damage/s: Ω = 1.5
    assert!((current_damage - 1.5).abs() < 1e-6);
}

#[test]
fn test_viability_from_damage() {
    let kinetics = AblationKinetics::default();

    // No damage
    assert!((kinetics.viability(0.0) - 1.0).abs() < 1e-6);

    // Ω = 1 (63.2% damage) -> e^(-1)
    assert!((kinetics.viability(1.0) - (-1.0_f64).exp()).abs() < 1e-12);

    // Ω = 5 (99.3% damage) -> e^(-5)
    assert!((kinetics.viability(5.0) - (-5.0_f64).exp()).abs() < 1e-12);
}

#[test]
fn test_ablation_threshold() {
    let kinetics = AblationKinetics::hifu_ablation();

    // Below threshold
    assert!(!kinetics.is_ablated(0.5));
    assert!(!kinetics.is_ablated(0.99));

    // At threshold
    assert!(kinetics.is_ablated(1.0));

    // Above threshold
    assert!(kinetics.is_ablated(1.5));
    assert!(kinetics.is_ablated(10.0));
}

#[test]
fn test_ablation_state_update() {
    let kinetics = AblationKinetics::hifu_ablation();
    let mut state = AblationState::new(37.0, &kinetics);

    // No damage at body temperature
    state.update(37.0, &kinetics, 1.0);
    assert_eq!(state.damage, 0.0);
    assert!(!state.ablated);

    // Damage accumulates at higher temperature
    state.update(60.0, &kinetics, 1.0);
    assert!(state.damage > 0.0);
    assert!(state.viability < 1.0);

    // Continuous heating
    for _ in 0..10 {
        state.update(65.0, &kinetics, 1.0);
    }
    assert!(state.ablated || state.damage > 0.0);
}

#[test]
fn test_kinetics_variants() {
    let protein = AblationKinetics::protein_denaturation();
    let collagen = AblationKinetics::collagen_denaturation();
    let hifu = AblationKinetics::hifu_ablation();

    // Different kinetics should give different damage rates at same temperature
    let t = 273.15 + 60.0;
    let rate_protein = protein.damage_rate(t);
    let rate_collagen = collagen.damage_rate(t);
    let rate_hifu = hifu.damage_rate(t);

    // HIFU should be most aggressive
    assert!(rate_hifu > rate_protein || rate_hifu > rate_collagen);
}

#[test]
fn test_ablation_field() {
    let kinetics = AblationKinetics::hifu_ablation();
    let mut field = AblationField::new((10, 10, 10), kinetics);

    // Create temperature field
    let mut temperature = Array3::from_elem((10, 10, 10), 37.0);
    temperature[[5, 5, 5]] = 70.0; // Hot spot at center

    // Update multiple times
    for _ in 0..5 {
        let _ = field.update(&temperature, 0.1);
    }

    // Center should have damage
    assert!(field.damage()[[5, 5, 5]] > 0.0);
    assert!(field.viability()[[5, 5, 5]] < 1.0);

    // Periphery should be unaffected
    assert_eq!(field.damage()[[0, 0, 0]], 0.0);
    assert_eq!(field.viability()[[0, 0, 0]], 1.0);
}

#[test]
fn test_ablation_volume_counting() {
    let kinetics = AblationKinetics::default();
    let mut field = AblationField::new((5, 5, 5), kinetics);

    // Create hotly heated field
    let mut temperature = Array3::from_elem((5, 5, 5), 37.0);
    for i in 1..4 {
        for j in 1..4 {
            for k in 1..4 {
                temperature[[i, j, k]] = 75.0;
            }
        }
    }

    // Update until some tissue is ablated
    for _ in 0..20 {
        let _ = field.update(&temperature, 0.1);
    }

    // Should have some ablated volume
    let ablated_vol = field.ablated_volume();
    assert!(ablated_vol > 0, "Expected ablated volume > 0");
}
