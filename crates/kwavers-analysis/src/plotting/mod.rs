//! Interactive HTML visualization of Kwavers acoustic fields and time series.

pub use kwavers_grid::Grid;
pub use kwavers_receiver::recorder::Recorder;

#[cfg(not(feature = "plotting"))]
mod disabled;
#[cfg(not(feature = "plotting"))]
pub use disabled::*;

#[cfg(feature = "plotting")]
mod field;
#[cfg(feature = "plotting")]
mod output;
#[cfg(feature = "plotting")]
mod series;
#[cfg(feature = "plotting")]
mod spatial;

#[cfg(feature = "plotting")]
pub use field::{plot_field_comparison, plot_pressure_field_2d};
#[cfg(feature = "plotting")]
pub use series::{plot_recorder_data, plot_temporal_evolution};
#[cfg(feature = "plotting")]
pub use spatial::{plot_positions, plot_pressure_field_3d};
