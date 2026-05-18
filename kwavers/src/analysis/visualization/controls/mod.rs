//! Interactive controls for real-time parameter adjustment during visualization

mod parameter;
mod state;
mod ui;
mod validation;

pub use parameter::{ParameterDefinition, ParameterType, ParameterValue};
pub use state::{ControlState, StateSnapshot};
pub use ui::{ControlPanel, ControlPanelConfig};
pub use validation::{ControlValidationResult, ParameterValidator};

// Re-export main control system
pub use state::InteractiveControls;
