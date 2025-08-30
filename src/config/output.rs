// config/output.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize]
pub struct OutputConfig {
    #[serde(default = "default_pressure_file")]
    pub pressure_file: String,
    #[serde(default = "default_light_file")]
    pub light_file: String, // Added for light fluence rate
    #[serde(default = "default_summary_file")]
    pub summary_file: String,
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval: usize,
    #[serde(default = "default_visualization")]
    pub enable_visualization: bool,
}

fn default_pressure_file() -> String {
    "pressure_output.csv".to_string()
}
fn default_light_file() -> String {
    "light_output.csv".to_string()
}
fn default_summary_file() -> String {
    "summary.csv".to_string()
}
fn default_snapshot_interval() -> usize {
    10
}
fn default_visualization() -> bool {
    false
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            pressure_file: default_pressure_file(),
            light_file: default_light_file(),
            summary_file: default_summary_file(),
            snapshot_interval: default_snapshot_interval(),
            enable_visualization: default_visualization(),
        }
    }
}
