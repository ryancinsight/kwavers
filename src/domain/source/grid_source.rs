//! Grid Source Definition
//!
//! Data structures for defining sources on the computational grid.

use ndarray::{Array2, Array3};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SourceMode {
    #[default]
    Additive,
    Dirichlet, // Enforce value (hard source)
}

/// Container for grid-distributed source definitions
#[derive(Default, Clone, Debug)]
pub struct GridSource {
    // Initial conditions
    pub p0: Option<Array3<f64>>,
    pub u0: Option<(Array3<f64>, Array3<f64>, Array3<f64>)>,

    // Time varying pressure source
    // If p_mask is defined, p_signal must be defined
    pub p_mask: Option<Array3<f64>>,
    pub p_signal: Option<Array2<f64>>, // [num_sources, time_steps]
    pub p_mode: SourceMode,

    // Time varying velocity source
    pub u_mask: Option<Array3<f64>>,
    pub u_signal: Option<Array3<f64>>, // [3, num_sources, time_steps]
    pub u_mode: SourceMode,
}
