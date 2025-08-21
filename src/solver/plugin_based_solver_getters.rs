// Temporary file with getter methods to be added to PluginBasedSolver

impl PluginBasedSolver {
    /// Get current time
    pub fn time(&self) -> f64 {
        self.time.current
    }
    
    /// Get medium
    pub fn medium(&self) -> &Arc<dyn Medium> {
        &self.medium
    }
    
    /// Get source
    pub fn source(&self) -> &Box<dyn Source> {
        &self.source
    }
}