use super::readonly::PluginFieldAccess;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::physics::acoustics::state::PhysicsState;

#[test]
fn test_plugin_field_access() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let state = PhysicsState::new(grid);

    // Create accessor with specific permissions
    let required = vec![UnifiedFieldType::Pressure];
    let provided = vec![UnifiedFieldType::Temperature];
    let access = PluginFieldAccess::new(&state, &required, &provided);

    // Should be able to read pressure (required)
    assert!(access.can_read(UnifiedFieldType::Pressure));
    assert!(!access.can_write(UnifiedFieldType::Pressure));

    // Should be able to read and write temperature (provided)
    assert!(access.can_read(UnifiedFieldType::Temperature));
    assert!(access.can_write(UnifiedFieldType::Temperature));

    // Should not be able to access density (not declared)
    assert!(!access.can_read(UnifiedFieldType::Density));
    assert!(!access.can_write(UnifiedFieldType::Density));
}

#[test]
fn test_unauthorized_access() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let state = PhysicsState::new(grid);

    // Create accessor with limited permissions
    let required = vec![UnifiedFieldType::Pressure];
    let provided = vec![];
    let access = PluginFieldAccess::new(&state, &required, &provided);

    // get_field_mut is not available on immutable accessor
    // This would require PluginFieldAccessMut

    // Try to access an undeclared field
    let result = access.get_field(UnifiedFieldType::Density);
    assert!(result.is_err());
}
