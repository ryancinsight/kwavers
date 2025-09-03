//! k-Wave test case definitions
//!
//! Standard test cases for validation against k-Wave toolbox

/// k-Wave validation test case
#[derive(Debug, Clone)]
pub struct KWaveTestCase {
    /// Test case name
    pub name: String,
    /// Test description
    pub description: String,
    /// Expected error tolerance
    pub tolerance: f64,
    /// Reference solution source
    pub reference: ReferenceSource,
}

/// Source of reference solution
#[derive(Debug, Clone)]
pub enum ReferenceSource {
    /// Analytical solution
    Analytical,
    /// k-Wave MATLAB output
    KWaveMatlab,
    /// k-Wave C++ output
    KWaveCpp,
    /// Published paper results
    Literature(String),
}

impl KWaveTestCase {
    /// Create standard k-Wave test cases
    #[must_use]
    pub fn standard_test_cases() -> Vec<KWaveTestCase> {
        vec![
            KWaveTestCase {
                name: "homogeneous_propagation".to_string(),
                description: "Plane wave in homogeneous medium".to_string(),
                tolerance: 1e-3,
                reference: ReferenceSource::Analytical,
            },
            KWaveTestCase {
                name: "pml_absorption".to_string(),
                description: "PML boundary absorption test".to_string(),
                tolerance: 1e-4,
                reference: ReferenceSource::KWaveMatlab,
            },
            KWaveTestCase {
                name: "heterogeneous_medium".to_string(),
                description: "Wave propagation in layered medium".to_string(),
                tolerance: 5e-3,
                reference: ReferenceSource::KWaveMatlab,
            },
            KWaveTestCase {
                name: "nonlinear_propagation".to_string(),
                description: "Nonlinear wave with harmonic generation".to_string(),
                tolerance: 1e-2,
                reference: ReferenceSource::Literature("Treeby et al. 2012".to_string()),
            },
            KWaveTestCase {
                name: "focused_transducer".to_string(),
                description: "Focused bowl transducer field".to_string(),
                tolerance: 2e-3,
                reference: ReferenceSource::KWaveMatlab,
            },
        ]
    }
}
