//! Wavefield modeling for FWI
//! Based on Virieux (1986): "P-SV wave propagation in heterogeneous media"

use ndarray::Array3;

mod adjoint;
mod forward;
mod modeler;
mod numerics;
#[cfg(test)]
mod tests;

/// Configuration for wavefield modeling
#[derive(Debug, Clone)]
pub struct WavefieldConfig {
    /// Grid spacing \[m\]
    pub dx: f64,
    /// Time step \[s\]
    pub dt: f64,
    /// Maximum simulation time \[s\]
    pub max_time: f64,
    /// Peak frequency for source wavelet \[Hz\]
    pub peak_frequency: f64,
    /// Source position (i, j, k)
    pub source_position: Option<(usize, usize, usize)>,
    /// Receiver positions
    pub receivers: Vec<(usize, usize, usize)>,
}

impl Default for WavefieldConfig {
    fn default() -> Self {
        Self {
            dx: 0.001,
            dt: 1e-6,
            max_time: 0.01,
            peak_frequency: 1e6,
            source_position: None,
            receivers: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct ForwardCheckpoint {
    previous: Array3<f64>,
    current: Array3<f64>,
}

#[derive(Debug)]
struct ForwardReplayCache {
    nt: usize,
    stride: usize,
    checkpoints: Vec<ForwardCheckpoint>,
}

/// Wavefield modeling for forward and adjoint problems
#[derive(Debug)]
pub struct WavefieldModeler {
    config: WavefieldConfig,
    /// Sparse forward checkpoints for exact replay-based adjoint accumulation
    forward_replay: Option<ForwardReplayCache>,
    /// Final forward wavefield snapshot for diagnostics
    last_forward_wavefield: Option<Array3<f64>>,
    /// PML boundary width
    pml_width: usize,
}

impl Default for WavefieldModeler {
    fn default() -> Self {
        Self::new()
    }
}
