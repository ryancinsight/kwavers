/// Risk management record
#[derive(Debug, Clone)]
pub struct RiskRecord {
    /// Risk ID
    pub risk_id: String,
    /// Risk description
    pub description: String,
    /// Severity level: Low, Medium, High, Critical
    pub severity: String,
    /// Probability of occurrence: Low, Medium, High
    pub probability: String,
    /// Risk priority number (RPN)
    pub rpn: u32,
    /// Mitigation measures
    pub mitigations: Vec<String>,
    /// Residual risk assessment
    pub residual_assessment: String,
}

impl RiskRecord {
    /// Create new risk record
    pub fn new(risk_id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            risk_id: risk_id.into(),
            description: description.into(),
            severity: "Medium".to_string(),
            probability: "Medium".to_string(),
            rpn: 9,
            mitigations: Vec::new(),
            residual_assessment: String::new(),
        }
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: impl Into<String>) -> Self {
        self.severity = severity.into();
        self
    }

    /// Set probability level
    pub fn with_probability(mut self, probability: impl Into<String>) -> Self {
        self.probability = probability.into();
        self
    }

    /// Calculate RPN based on severity and probability
    pub fn calculate_rpn(&mut self) -> u32 {
        let severity = match self.severity.as_str() {
            "Critical" => 10,
            "High" => 7,
            "Medium" => 5,
            "Low" => 2,
            _ => 3,
        };

        let probability = match self.probability.as_str() {
            "High" => 5,
            "Medium" => 3,
            "Low" => 1,
            _ => 2,
        };

        self.rpn = severity * probability;
        self.rpn
    }

    /// Add mitigation measure
    pub fn add_mitigation(&mut self, mitigation: impl Into<String>) {
        self.mitigations.push(mitigation.into());
    }

    /// Set residual risk assessment
    pub fn set_residual_assessment(&mut self, assessment: impl Into<String>) {
        self.residual_assessment = assessment.into();
    }
}
