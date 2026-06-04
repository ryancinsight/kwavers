//! Plugin interfaces and contracts
//!
//! This module defines the core traits and types for the plugin system,
//! allowing loose coupling between the solver orchestration and physics implementations.

pub mod access;
pub mod fields;
pub mod metadata;

pub use access::DirectPluginFieldAccess;
pub use fields::PluginFields;
pub use metadata::PluginMetadata;

use kwavers_core::error::KwaversResult;
use kwavers_core::time::StabilityConstraints;
use crate::boundary::Boundary;
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use crate::source::Source;
use ndarray::Array4;
use std::any::Any;
use std::fmt::Debug;

/// State of a plugin
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is created but not initialized
    Created,
    /// Plugin is configured with parameters
    Configured,
    /// Plugin is initialized and ready
    Initialized,
    /// Plugin is actively processing
    Running,
    /// Plugin is paused
    Paused,
    /// Plugin encountered an error
    Error,
    /// Plugin has been finalized
    Finalized,
}

/// Priority levels for plugin execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PluginPriority {
    /// Lowest priority - executed last
    Low = 0,
    /// Normal priority - default
    Normal = 1,
    /// High priority - executed early
    High = 2,
    /// Critical priority - executed first
    Critical = 3,
}

/// Context passed to plugins during execution
#[derive(Debug)]
pub struct PluginContext<'a> {
    /// Additional fields for plugin communication
    pub extra_fields: &'a PluginFields,
    /// Acoustic sources in the simulation
    pub sources: &'a [Box<dyn Source>],
    /// Boundary conditions
    pub boundary: &'a mut dyn Boundary,
}

/// Core trait that all plugins must implement
pub trait Plugin: Debug + Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Get current plugin state
    fn state(&self) -> PluginState;

    /// Set plugin state
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn set_state(&mut self, state: PluginState);

    /// Get required fields for this plugin
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn required_fields(&self) -> Vec<UnifiedFieldType>;

    /// Get fields provided by this plugin
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn provided_fields(&self) -> Vec<UnifiedFieldType>;

    /// Update the plugin with current fields
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()>;

    /// Initialize the plugin
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        Ok(())
    }

    /// Finalize the plugin
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }

    /// Reset plugin state
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn reset(&mut self) -> KwaversResult<()> {
        Ok(())
    }

    /// Get diagnostic information
    fn diagnostics(&self) -> String {
        format!("Plugin: {:?}", self.metadata())
    }

    /// Get stability constraints for time stepping
    fn stability_constraints(&self) -> StabilityConstraints {
        StabilityConstraints::default()
    }

    /// Get plugin priority
    fn priority(&self) -> PluginPriority {
        PluginPriority::Normal
    }

    /// Check if plugin is compatible with another plugin
    fn is_compatible_with(&self, _other: &dyn Plugin) -> bool {
        true
    }

    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Convert to mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ── Shared test support ───────────────────────────────────────────────────────

/// Test infrastructure for plugin unit tests.
///
/// Every plugin test that calls [`Plugin::update`] must supply a
/// [`PluginContext`], which requires a concrete [`Boundary`] implementation.
/// Re-implementing the full four-method `Boundary` trait in every test file
/// would violate DRY.  This module provides:
///
/// - [`test_support::NullBoundary`] — a no-op boundary satisfying the trait
///   contract without affecting any field value.
/// - [`test_support::null_context`] — constructs a ready-to-use
///   `(PluginFields, NullBoundary)` pair that callers borrow to form a
///   [`PluginContext`].
///
/// ## Usage
///
/// ```rust,ignore
/// use crate::plugin::test_support::{NullBoundary, null_plugin_fields};
/// use crate::plugin::{PluginContext, PluginFields};
///
/// let extra = null_plugin_fields(&grid);
/// let mut boundary = NullBoundary;
/// let mut ctx = PluginContext { extra_fields: &extra, sources: &[], boundary: &mut boundary };
/// plugin.update(&mut fields, &grid, &medium, dt, t, &mut ctx)?;
/// ```
/// Plugin test helpers (`make_context`, `null_plugin_fields`, `NullBoundary`).
/// Exposed under the `test-util` feature so downstream crates' tests (e.g. the
/// solver plugin tests) can reuse them across the crate boundary.
#[cfg(any(test, feature = "test-util"))]
pub mod test_support {
    use super::fields::PluginFields;
    use super::PluginContext;
    use kwavers_core::error::KwaversResult;
    use crate::boundary::Boundary;
    use kwavers_grid::Grid;
    use ndarray::Array3;

    /// No-op boundary condition for plugin unit tests.
    ///
    /// Every required method is a no-op: no energy is added or removed from
    /// any field.  The implementation is intentionally minimal — test behaviour
    /// must not depend on boundary effects unless the test is specifically
    /// exercising boundary logic.
    #[derive(Debug)]
    pub struct NullBoundary;

    impl Boundary for NullBoundary {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn apply_acoustic(
            &mut self,
            _field: ndarray::ArrayViewMut3<f64>,
            _grid: &Grid,
            _time_step: usize,
        ) -> KwaversResult<()> {
            Ok(())
        }

        fn apply_acoustic_freq(
            &mut self,
            _field: &mut Array3<kwavers_math::fft::Complex64>,
            _grid: &Grid,
            _time_step: usize,
        ) -> KwaversResult<()> {
            Ok(())
        }

        fn apply_light(
            &mut self,
            _field: ndarray::ArrayViewMut3<f64>,
            _grid: &Grid,
            _time_step: usize,
        ) {
        }
    }

    /// Construct zero-filled [`PluginFields`] sized to `grid`.
    ///
    /// The fields are spatially zero which is correct for test contexts where
    /// the extra-fields channel is not exercised.
    #[must_use]
    pub fn null_plugin_fields(grid: &Grid) -> PluginFields {
        PluginFields::new(Array3::zeros((grid.nx, grid.ny, grid.nz)))
    }

    /// Build a [`PluginContext`] from the provided extra-fields and boundary.
    ///
    /// The returned struct borrows both arguments; lifetime elision works
    /// naturally when the caller holds `extra_fields` and `boundary` as local
    /// `let` bindings before calling this helper.
    #[must_use]
    pub fn make_context<'a>(
        extra_fields: &'a PluginFields,
        boundary: &'a mut NullBoundary,
    ) -> PluginContext<'a> {
        PluginContext {
            extra_fields,
            sources: &[],
            boundary,
        }
    }
}
