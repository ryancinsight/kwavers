use super::model::{ChemicalModel, ChemicalModelState};
use super::reactions::{
    ChemicalReactionConfig, LightDependence, PressureDependence, ReactionType, ThermalDependence,
};
use crate::domain::grid::Grid;

/// Default rate constant used in tests
const DEFAULT_REACTION_RATE: f64 = 1e-3;

#[test]
fn test_chemical_model_creation() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let model = ChemicalModel::new(&grid, true, true).unwrap();
    assert_eq!(model.state(), &ChemicalModelState::Initialized);
}

#[test]
fn test_reaction_config() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let mut model = ChemicalModel::new(&grid, true, false).unwrap();

    let config = ChemicalReactionConfig {
        reaction_type: ReactionType::Dissociation,
        thermal_dependence: ThermalDependence::Constant,
        pressure_dependence: PressureDependence::Constant,
        light_dependence: LightDependence::None,
        rate_constant: DEFAULT_REACTION_RATE,
        activation_energy: 0.0,
    };

    model.add_reaction_config("test_reaction".to_string(), config);
    assert_eq!(model.reactions.len(), 1);
}
