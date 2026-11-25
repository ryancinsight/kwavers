//! Unified field mapping system - Single Source of Truth for field indices
//!
//! This module provides a centralized, type-safe way to map between field types
//! and their indices in the global fields array. This prevents data corruption
//! from incorrect field indexing.

pub use crate::UnifiedFieldType;
use leto::{Array4, ArrayView3, ArrayViewMut3, SliceArg};

/// Fix the field axis and retain the three spatial axes, including their strides.
fn field_slice(field: UnifiedFieldType) -> [SliceArg; 4] {
    let index = isize::try_from(field.index()).expect("field discriminants fit isize");
    [
        SliceArg::Index(index),
        SliceArg::All,
        SliceArg::All,
        SliceArg::All,
    ]
}

/// Type-safe field accessor to prevent index confusion
#[derive(Debug)]
pub struct FieldAccessor<'a> {
    fields: &'a Array4<f64>,
}

impl<'a> FieldAccessor<'a> {
    /// Create a read-only accessor over a unified field array.
    #[must_use]
    pub fn new(fields: &'a Array4<f64>) -> Self {
        Self { fields }
    }

    /// Get a specific field by type
    ///
    /// # Panics
    ///
    /// Panics if the first axis of `fields` is shorter than the index assigned
    /// to `field_type`. A correctly constructed unified field array contains
    /// every [`UnifiedFieldType`] slot.
    #[must_use]
    pub fn get(&self, field_type: UnifiedFieldType) -> ArrayView3<'a, f64> {
        self.fields
            .slice_with(&field_slice(field_type))
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
    /// Create a mutable accessor over a unified field array.
    #[must_use]
    pub fn new(fields: &'a mut Array4<f64>) -> Self {
        Self { fields }
    }

    /// Get a specific field mutably by type
    ///
    /// # Panics
    ///
    /// Panics if the first axis of `fields` is shorter than the index assigned
    /// to `field_type`. A correctly constructed unified field array contains
    /// every [`UnifiedFieldType`] slot.
    pub fn get_mut(&mut self, field_type: UnifiedFieldType) -> ArrayViewMut3<'_, f64> {
        self.fields
            .slice_with_mut(&field_slice(field_type))
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

    /// Field channels may be interleaved at each voxel in an owned array.
    #[test]
    fn accessors_select_the_same_channel_on_interleaved_storage() {
        use leto::{Layout, VecStorage};
        let count = UnifiedFieldType::COUNT;
        let channels = isize::try_from(count).unwrap();
        let layout = Layout::try_new(
            [count, 2, 1, 3],
            [1, 3 * channels, 3 * channels, channels],
            0,
        )
        .unwrap();
        let mut fields = Array4::new(layout, VecStorage::fill(count * 6, 0.0)).unwrap();
        for (n, field) in UnifiedFieldType::all().iter().copied().enumerate() {
            let value = f64::from(u16::try_from(n + 1).unwrap());
            FieldAccessorMut::new(&mut fields).get_mut(field)[[1, 0, 2]] = value;
            assert_eq!(fields[[field.index(), 1, 0, 2]], value);
        }
        let accessor = FieldAccessor::new(&fields);
        for (n, field) in UnifiedFieldType::all().iter().copied().enumerate() {
            let view = accessor.get(field);
            assert_eq!(view[[1, 0, 2]], f64::from(u16::try_from(n + 1).unwrap()));
            assert_eq!(view[[0, 0, 2]], 0.0);
        }
    }

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
