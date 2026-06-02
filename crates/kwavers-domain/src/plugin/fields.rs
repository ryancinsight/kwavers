//! Plugin field data container
//!
//! Holds field data for plugins.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

/// Plugin field data container
#[derive(Debug, Clone)]
pub struct PluginFields {
    data: Array3<f64>,
}

impl PluginFields {
    #[must_use]
    pub fn new(data: Array3<f64>) -> Self {
        Self { data }
    }

    #[must_use]
    pub fn view(&self) -> ArrayView3<'_, f64> {
        self.data.view()
    }

    #[must_use]
    pub fn view_mut(&mut self) -> ArrayViewMut3<'_, f64> {
        self.data.view_mut()
    }
}
