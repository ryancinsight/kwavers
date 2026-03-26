use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

/// Device classification per FDA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// Class I: Low risk, general controls
    ClassI,
    /// Class II: Moderate risk, requires 510(k) submission
    ClassII,
    /// Class III: High risk, requires Premarket Approval (PMA)
    ClassIII,
}

impl DeviceClass {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ClassI => "Class I",
            Self::ClassII => "Class II",
            Self::ClassIII => "Class III",
        }
    }

    /// Get risk level description
    pub fn risk_level(&self) -> &'static str {
        match self {
            Self::ClassI => "Low Risk",
            Self::ClassII => "Moderate Risk",
            Self::ClassIII => "High Risk",
        }
    }
}

/// Device description section
#[derive(Debug, Clone)]
pub struct DeviceDescription {
    /// Device name and model
    pub name: String,
    /// Device classification
    pub classification: DeviceClass,
    /// Intended use statement
    pub intended_use: String,
    /// Indications for use
    pub indications: Vec<String>,
    /// Device specifications
    pub specifications: HashMap<String, String>,
    /// Key features
    pub features: Vec<String>,
    /// Contraindications
    pub contraindications: Vec<String>,
}

impl DeviceDescription {
    /// Create a new device description
    pub fn new(name: impl Into<String>, classification: DeviceClass) -> Self {
        Self {
            name: name.into(),
            classification,
            intended_use: String::new(),
            indications: Vec::new(),
            specifications: HashMap::new(),
            features: Vec::new(),
            contraindications: Vec::new(),
        }
    }

    /// Set intended use statement
    pub fn with_intended_use(mut self, use_statement: impl Into<String>) -> Self {
        self.intended_use = use_statement.into();
        self
    }

    /// Add indication for use
    pub fn add_indication(&mut self, indication: impl Into<String>) {
        self.indications.push(indication.into());
    }

    /// Add device specification
    pub fn add_specification(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.specifications.insert(key.into(), value.into());
    }

    /// Add feature
    pub fn add_feature(&mut self, feature: impl Into<String>) {
        self.features.push(feature.into());
    }

    /// Add contraindication
    pub fn add_contraindication(&mut self, contraindication: impl Into<String>) {
        self.contraindications.push(contraindication.into());
    }

    /// Validate device description completeness
    pub fn validate(&self) -> KwaversResult<()> {
        if self.name.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Device name cannot be empty".to_string(),
            ));
        }

        if self.intended_use.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Intended use statement required".to_string(),
            ));
        }

        if self.indications.is_empty() {
            return Err(KwaversError::InvalidInput(
                "At least one indication for use required".to_string(),
            ));
        }

        Ok(())
    }
}

/// Predicate device information for substantial equivalence
#[derive(Debug, Clone)]
pub struct PredicateDevice {
    /// Predicate device name
    pub name: String,
    /// 510(k) number of predicate
    pub k_number: String,
    /// Manufacturer name
    pub manufacturer: String,
    /// Year of predicate clearance
    pub clearance_year: u32,
    /// Similarities to predicate
    pub similarities: Vec<String>,
    /// Differences from predicate (and justification)
    pub differences: Vec<(String, String)>,
}

impl PredicateDevice {
    /// Create new predicate device record
    pub fn new(
        name: impl Into<String>,
        k_number: impl Into<String>,
        manufacturer: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            k_number: k_number.into(),
            manufacturer: manufacturer.into(),
            clearance_year: 2020,
            similarities: Vec::new(),
            differences: Vec::new(),
        }
    }

    /// Add similarity to predicate
    pub fn add_similarity(&mut self, similarity: impl Into<String>) {
        self.similarities.push(similarity.into());
    }

    /// Add difference with justification
    pub fn add_difference(
        &mut self,
        difference: impl Into<String>,
        justification: impl Into<String>,
    ) {
        self.differences
            .push((difference.into(), justification.into()));
    }

    /// Validate predicate device information
    pub fn validate(&self) -> KwaversResult<()> {
        if self.name.is_empty() || self.k_number.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Predicate device name and K number required".to_string(),
            ));
        }

        if self.similarities.is_empty() {
            return Err(KwaversError::InvalidInput(
                "At least one similarity to predicate required".to_string(),
            ));
        }

        Ok(())
    }
}
