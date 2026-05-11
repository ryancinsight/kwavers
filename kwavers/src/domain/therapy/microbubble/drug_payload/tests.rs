use super::loading_mode::DrugLoadingMode;
use super::payload::DrugPayload;
use crate::domain::therapy::microbubble::shell::ShellState;

#[test]
fn test_create_drug_payload() {
    let volume = 1e-15;
    let concentration = 50.0;
    let payload =
        DrugPayload::new(concentration, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

    assert_eq!(payload.concentration, concentration);
    assert_eq!(payload.released_mass, 0.0);
    assert_eq!(payload.release_fraction(), 0.0);
    assert!(!payload.is_depleted());
    payload.validate().unwrap();
}

#[test]
fn test_doxorubicin_payload() {
    let volume = 1e-15;
    let payload = DrugPayload::doxorubicin(volume).unwrap();

    assert_eq!(payload.concentration, 50.0);
    assert_eq!(payload.loading_mode, DrugLoadingMode::ShellEmbedded);
    payload.validate().unwrap();
}

#[test]
fn test_permeability_ruptured() {
    let volume = 1e-15;
    let payload = DrugPayload::new(50.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

    let perm = payload.permeability_factor(ShellState::Ruptured, 0.5);
    assert_eq!(perm, 1.0);
}

#[test]
fn test_permeability_elastic_no_strain() {
    let volume = 1e-15;
    let payload = DrugPayload::new(50.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

    let perm = payload.permeability_factor(ShellState::Elastic, 0.0);
    assert_eq!(perm, payload.baseline_permeability);
}

#[test]
fn test_permeability_enhanced_by_strain() {
    let volume = 1e-15;
    let payload = DrugPayload::new(50.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

    let strain = 0.5;
    let perm_strained = payload.permeability_factor(ShellState::Elastic, strain);
    let perm_unstrained = payload.permeability_factor(ShellState::Elastic, 0.0);

    assert!(perm_strained > perm_unstrained);
}

#[test]
fn test_release_no_permeability() {
    let volume = 1e-15;
    let mut payload =
        DrugPayload::new(50.0, volume, DrugLoadingMode::CoreEncapsulated, 0.0).unwrap();

    let released = payload
        .update_release(volume, ShellState::Elastic, 0.0, 1.0)
        .unwrap();

    assert_eq!(released, 0.0);
    assert_eq!(payload.concentration, 50.0);
}

#[test]
fn test_release_exponential_decay() {
    let volume = 1e-15;
    let concentration = 100.0;
    let k = 0.1;
    let mut payload =
        DrugPayload::new(concentration, volume, DrugLoadingMode::SurfaceAttached, k).unwrap();

    let dt = 1.0;
    let permeability = payload.permeability_factor(ShellState::Elastic, 0.0);

    payload
        .update_release(volume, ShellState::Elastic, 0.0, dt)
        .unwrap();

    let expected = concentration * (-k * permeability * dt).exp();
    assert!((payload.concentration - expected).abs() < 1e-10);
}

#[test]
fn test_release_mass_conservation() {
    let volume = 1e-15;
    let concentration = 100.0;
    let mut payload =
        DrugPayload::new(concentration, volume, DrugLoadingMode::ShellEmbedded, 0.1).unwrap();

    let initial_mass = payload.initial_mass;

    for _ in 0..10 {
        payload
            .update_release(volume, ShellState::Elastic, 0.2, 0.1)
            .unwrap();
    }

    let remaining = payload.remaining_mass(volume);
    let total = remaining + payload.released_mass;

    assert!((total - initial_mass).abs() / initial_mass < 1e-10);
}

#[test]
fn test_release_complete_on_rupture() {
    let volume = 1e-15;
    let mut payload =
        DrugPayload::new(100.0, volume, DrugLoadingMode::CoreEncapsulated, 0.5).unwrap();

    for _ in 0..100 {
        payload
            .update_release(volume, ShellState::Ruptured, 1.0, 0.1)
            .unwrap();
    }

    assert!(
        payload.release_fraction() > 0.99,
        "Expected >99% release, got {:.2}%",
        payload.release_fraction() * 100.0
    );
}

#[test]
fn test_release_fraction() {
    let volume = 1e-15;
    let mut payload = DrugPayload::new(100.0, volume, DrugLoadingMode::ShellEmbedded, 0.1).unwrap();

    assert_eq!(payload.release_fraction(), 0.0);

    let half_life = (2.0_f64.ln()) / 0.1;
    let permeability = payload.permeability_factor(ShellState::Elastic, 0.0);
    let dt = half_life / permeability;

    payload
        .update_release(volume, ShellState::Elastic, 0.0, dt)
        .unwrap();

    assert!((payload.release_fraction() - 0.5).abs() < 0.01);
}

#[test]
fn test_is_depleted() {
    let volume = 1e-15;
    let mut payload = DrugPayload::new(1e-9, volume, DrugLoadingMode::ShellEmbedded, 1.0).unwrap();

    assert!(!payload.is_depleted());

    for _ in 0..100 {
        payload
            .update_release(volume, ShellState::Ruptured, 1.0, 0.1)
            .unwrap();
    }

    assert!(payload.is_depleted());
}

#[test]
fn test_validation_negative_concentration() {
    let volume = 1e-15;
    let result = DrugPayload::new(-10.0, volume, DrugLoadingMode::ShellEmbedded, 0.1);
    assert!(result.is_err());
}

#[test]
fn test_loading_mode_display() {
    assert_eq!(format!("{}", DrugLoadingMode::SurfaceAttached), "Surface");
    assert_eq!(format!("{}", DrugLoadingMode::ShellEmbedded), "Shell");
    assert_eq!(format!("{}", DrugLoadingMode::CoreEncapsulated), "Core");
}
