//! Interactive controls for real-time parameter adjustment during visualization

mod parameter;
mod state;
#[cfg(feature = "egui")]
mod ui;
mod validation;

pub use parameter::{ParameterDefinition, ParameterType, ParameterValue};
pub use state::{ControlState, StateSnapshot};
#[cfg(feature = "egui")]
pub use ui::{ControlPanel, ControlPanelConfig};
pub use validation::{ControlValidationResult, ParameterValidator};

// Re-export main control system
pub use state::InteractiveControls;
