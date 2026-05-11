use crate::core::error::{KwaversError, KwaversResult, ValidationError};

/// Field layout configuration for batch allocation.
#[derive(Debug, Clone, Copy)]
pub struct BatchFieldConfig {
    /// Number of elements per field.
    pub field_elements: usize,
    /// Number of fields to allocate.
    pub num_fields: usize,
    /// Alignment in bytes (default: `CACHE_LINE_SIZE`).
    pub alignment: usize,
    /// NUMA node for first-touch policy (`None` = OS default).
    pub numa_node: Option<u32>,
}

impl Default for BatchFieldConfig {
    fn default() -> Self {
        Self {
            field_elements: 0,
            num_fields: 0,
            alignment: super::CACHE_LINE_SIZE,
            numa_node: None,
        }
    }
}

impl BatchFieldConfig {
    /// Create configuration for a 3-D field batch.
    #[must_use]
    pub fn for_3d_fields(nx: usize, ny: usize, nz: usize, num_fields: usize) -> Self {
        Self {
            field_elements: nx * ny * nz,
            num_fields,
            alignment: super::CACHE_LINE_SIZE,
            numa_node: None,
        }
    }

    /// Create configuration for a 2-D field batch.
    #[must_use]
    pub fn for_2d_fields(nx: usize, ny: usize, num_fields: usize) -> Self {
        Self {
            field_elements: nx * ny,
            num_fields,
            alignment: super::CACHE_LINE_SIZE,
            numa_node: None,
        }
    }

    /// Set NUMA node preference.
    #[must_use]
    pub fn with_numa_node(mut self, node: u32) -> Self {
        self.numa_node = Some(node);
        self
    }

    /// Total memory required in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.field_elements
            .checked_mul(self.num_fields)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .unwrap_or(usize::MAX)
    }

    /// Validate configuration.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.field_elements == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "field_elements".to_owned(),
                value: 0.0,
                reason: "Field elements must be non-zero".to_owned(),
            }));
        }
        if self.num_fields == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "num_fields".to_owned(),
                value: 0.0,
                reason: "Number of fields must be non-zero".to_owned(),
            }));
        }
        if !self.alignment.is_power_of_two() {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "alignment".to_owned(),
                value: self.alignment as f64,
                reason: "Alignment must be a power of two".to_owned(),
            }));
        }
        Ok(())
    }
}
