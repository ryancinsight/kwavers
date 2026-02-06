/// Clinical workflow state
#[derive(Debug, Clone)]
pub enum WorkflowState {
    /// Initializing workflow components
    Initializing,
    /// Acquiring data from modalities
    Acquiring,
    /// Processing acquired data
    Processing,
    /// Performing multi-modal fusion
    Fusing,
    /// Running AI analysis
    Analyzing,
    /// Generating clinical report
    Reporting,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed with error
    Failed(String),
}

#[allow(clippy::derivable_impls)]
impl Default for WorkflowState {
    fn default() -> Self {
        Self::Initializing
    }
}
