//! Unified field mapping system - Single Source of Truth for field indices
//!
//! This module provides a centralized, type-safe way to map between field types
//! and their indices in the global fields array. This prevents data corruption
//! from incorrect field indexing.

pub use crate::UnifiedFieldType;
use leto::{Array4, ArrayView3, ArrayViewMut3};

/// Type-safe field accessor to prevent index confusion
#[derive(Debug)]
pub struct FieldAccessor<'a> {
    fields: &'a Array4<f64>,
}

impl<'a> FieldAccessor<'a> {
    #[must_use]
    pub fn new(fields: &'a Array4<f64>) -> Self {
        Self { fields }
    }

    /// Get a specific field by type
    #[must_use]
    pub fn get(&self, field_type: UnifiedFieldType) -> ArrayView3<'a, f64> {
        self.fields
            .index_axis::<3>(0, field_type.index())
            .expect("field index is always valid for a correctly shaped fields array")
    }

    /// Get pressure field
    #[must_use]
    pub fn pressure(&self) -> ArrayView3<'a, f64> {
        self.get(UnifiedFieldType::Pressure)
    }

    /// Get temperature field
    #[must_use]
    pub fn temperature(&self) -> ArrayView3<'a, f64> {
        self.get(UnifiedFieldType::Temperature)
    }

    /// Get density field
    #[must_use]
    pub fn density(&self) -> ArrayView3<'a, f64> {
        self.get(UnifiedFieldType::Density)
    }
}

/// Type-safe mutable field accessor
#[derive(Debug)]
pub struct FieldAccessorMut<'a> {
    fields: &'a mut Array4<f64>,
}

impl<'a> FieldAccessorMut<'a> {
    #[must_use]
    pub fn new(fields: &'a mut Array4<f64>) -> Self {
        Self { fields }
    }

    /// Get a specific field mutably by type
    pub fn get_mut(&mut self, field_type: UnifiedFieldType) -> ArrayViewMut3<'_, f64> {
        self.fields
            .index_axis_mut::<3>(0, field_type.index())
            .expect("field index is always valid for a correctly shaped fields array")
    }

    /// Get pressure field mutably
    pub fn pressure_mut(&mut self) -> ArrayViewMut3<'_, f64> {
        self.get_mut(UnifiedFieldType::Pressure)
    }

    /// Get temperature field mutably
    pub fn temperature_mut(&mut self) -> ArrayViewMut3<'_, f64> {
        self.get_mut(UnifiedFieldType::Temperature)
    }

    /// Get density field mutably
    pub fn density_mut(&mut self) -> ArrayViewMut3<'_, f64> {
        self.get_mut(UnifiedFieldType::Density)
    }
}

// Migration helper removed - composable module has been deprecated and removed

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_indices_are_unique() {
        let all_fields = UnifiedFieldType::all();
        let mut indices = std::collections::HashSet::new();

        for field in &all_fields {
            let idx = field.index();
            assert!(
                indices.insert(idx),
                "Duplicate index {} for field {:?}",
                idx,
                field
            );
        }
    }

    #[test]
    fn test_field_index_consistency() {
        // Ensure indices match the constants in state.rs
        assert_eq!(UnifiedFieldType::Pressure.index(), 0);
        assert_eq!(UnifiedFieldType::Temperature.index(), 1);
        assert_eq!(UnifiedFieldType::BubbleRadius.index(), 2);
        assert_eq!(UnifiedFieldType::BubbleVelocity.index(), 3);
        assert_eq!(UnifiedFieldType::Density.index(), 4);
        assert_eq!(UnifiedFieldType::SoundSpeed.index(), 5);
    }

    #[test]
    fn test_from_index() {
        assert_eq!(
            UnifiedFieldType::from_index(0),
            Some(UnifiedFieldType::Pressure)
        );
        assert_eq!(
            UnifiedFieldType::from_index(1),
            Some(UnifiedFieldType::Temperature)
        );
        assert_eq!(UnifiedFieldType::from_index(100), None);
    }

    #[test]
    fn test_display() {
        let test_cases = vec![
            (UnifiedFieldType::Pressure, "Pressure (Pa)"),
            (UnifiedFieldType::Temperature, "Temperature (K)"),
            (UnifiedFieldType::LightFluence, "Light Fluence (J/m²)"),
            (UnifiedFieldType::BubbleRadius, "Bubble Radius (m)"),
        ];

        for (field_type, expected_display) in test_cases {
            assert_eq!(field_type.to_string(), expected_display);
        }
    }
}
