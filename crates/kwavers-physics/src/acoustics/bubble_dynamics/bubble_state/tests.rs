use super::gas_dynamics::GasSpecies;
use super::parameters::BubbleParameters;
use super::state::BubbleState;

#[test]
fn test_bubble_state_creation() {
    let params = BubbleParameters::default();
    let state = BubbleState::new(&params);

    assert_eq!(state.radius, params.r0);
    assert_eq!(state.wall_velocity, 0.0);
    assert!(state.n_gas > 0.0);
    assert_eq!(state.gas_species, GasSpecies::Air);
}

#[test]
fn test_gas_properties() {
    assert_eq!(GasSpecies::Argon.gamma(), 5.0 / 3.0);
    assert_eq!(GasSpecies::Air.gamma(), 1.4);
    assert!((GasSpecies::Xenon.molecular_weight() - 0.131).abs() < 1e-6);
}

#[test]
fn test_compression_tracking() {
    let params = BubbleParameters::default();
    let mut state = BubbleState::new(&params);

    state.radius = params.r0 / 10.0; // Compress to 1/10
    state.update_compression(params.r0);

    assert_eq!(state.compression_ratio, 10.0);
    assert_eq!(state.max_compression, 10.0);
}
