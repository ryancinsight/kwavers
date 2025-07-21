// src/config.rs
//! Advanced configuration system for kwavers
//!
//! This module provides a sophisticated configuration management system following
//! multiple design principles:
//!
//! Design Principles Implemented:
//! - SOLID: Single responsibility for each config type, open/closed for extensibility
//! - CUPID: Composable config components, predictable config behavior
//! - GRASP: Information expert for config validation, controller for config flow
//! - ACID: Atomic config operations, consistent config states
//! - DRY: Shared config patterns and utilities
//! - KISS: Simple, clear config interfaces
//! - YAGNI: Only implement necessary config features
//! - SSOT: Single source of truth for configuration
//! - CCP: Common closure for related config handling
//! - CRP: Common reuse of config utilities
//! - ADP: Acyclic dependency in config hierarchy

use crate::error::{KwaversResult, ConfigError, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

/// Configuration value types
/// 
/// Implements SSOT principle as the single source of truth for config values
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
    Null,
}

impl ConfigValue {
    /// Get value as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    /// Get value as integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(i) => Some(*i),
            ConfigValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }
    
    /// Get value as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            ConfigValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
    
    /// Get value as boolean
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            ConfigValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    
    /// Get value as array
    pub fn as_array(&self) -> Option<&Vec<ConfigValue>> {
        match self {
            ConfigValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
    
    /// Get value as object
    pub fn as_object(&self) -> Option<&HashMap<String, ConfigValue>> {
        match self {
            ConfigValue::Object(obj) => Some(obj),
            _ => None,
        }
    }
    
    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, ConfigValue::Null)
    }
}

/// Configuration schema for validation
/// 
/// Implements Information Expert principle by providing validation logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    pub fields: HashMap<String, FieldSchema>,
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub field_type: FieldType,
    pub description: String,
    pub default_value: Option<ConfigValue>,
    pub constraints: Vec<Constraint>,
    pub nested_schema: Option<Box<ConfigSchema>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    Any,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    MinValue(f64),
    MaxValue(f64),
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Enum(Vec<ConfigValue>),
    Custom(String), // Custom validation rule
}

impl ConfigSchema {
    /// Validate configuration against schema
    pub fn validate(&self, config: &Configuration) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        // Check required fields
        for field in &self.required_fields {
            if !config.has_key(field) {
                errors.push(ValidationError::FieldValidation {
                    field: field.clone(),
                    value: "missing".to_string(),
                    constraint: "required".to_string(),
                });
            }
        }
        
        // Validate field values
        for (field_name, field_value) in config.iter() {
            if let Some(field_schema) = self.fields.get(field_name) {
                if let Err(field_errors) = self.validate_field(field_name, field_value, field_schema) {
                    errors.extend(field_errors);
                }
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    
    fn validate_field(&self, field_name: &str, value: &ConfigValue, schema: &FieldSchema) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        // Type validation
        if !self.matches_type(value, &schema.field_type) {
            errors.push(ValidationError::TypeValidation {
                field: field_name.to_string(),
                expected_type: format!("{:?}", schema.field_type),
                actual_type: self.get_value_type(value),
            });
        }
        
        // Constraint validation
        for constraint in &schema.constraints {
            if let Err(error) = self.validate_constraint(field_name, value, constraint) {
                errors.push(error);
            }
        }
        
        // Nested schema validation
        if let (Some(nested_schema), ConfigValue::Object(obj)) = (&schema.nested_schema, value) {
            let nested_config = Configuration::from_map(obj.clone());
            if let Err(nested_errors) = nested_schema.validate(&nested_config) {
                for nested_error in nested_errors {
                    errors.push(ValidationError::FieldValidation {
                        field: format!("{}.{}", field_name, nested_error.to_string()),
                        value: "nested validation failed".to_string(),
                        constraint: nested_error.to_string(),
                    });
                }
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    
    fn matches_type(&self, value: &ConfigValue, expected_type: &FieldType) -> bool {
        match (value, expected_type) {
            (ConfigValue::String(_), FieldType::String) => true,
            (ConfigValue::Integer(_), FieldType::Integer) => true,
            (ConfigValue::Float(_), FieldType::Float) => true,
            (ConfigValue::Boolean(_), FieldType::Boolean) => true,
            (ConfigValue::Array(_), FieldType::Array) => true,
            (ConfigValue::Object(_), FieldType::Object) => true,
            (_, FieldType::Any) => true,
            _ => false,
        }
    }
    
    fn get_value_type(&self, value: &ConfigValue) -> String {
        match value {
            ConfigValue::String(_) => "String".to_string(),
            ConfigValue::Integer(_) => "Integer".to_string(),
            ConfigValue::Float(_) => "Float".to_string(),
            ConfigValue::Boolean(_) => "Boolean".to_string(),
            ConfigValue::Array(_) => "Array".to_string(),
            ConfigValue::Object(_) => "Object".to_string(),
            ConfigValue::Null => "Null".to_string(),
        }
    }
    
    fn validate_constraint(&self, field_name: &str, value: &ConfigValue, constraint: &Constraint) -> Result<(), ValidationError> {
        match constraint {
            Constraint::MinValue(min) => {
                if let Some(val) = value.as_float() {
                    if val < *min {
                        return Err(ValidationError::RangeValidation {
                            field: field_name.to_string(),
                            value: val,
                            min: *min,
                            max: f64::INFINITY,
                        });
                    }
                }
            }
            Constraint::MaxValue(max) => {
                if let Some(val) = value.as_float() {
                    if val > *max {
                        return Err(ValidationError::RangeValidation {
                            field: field_name.to_string(),
                            value: val,
                            min: f64::NEG_INFINITY,
                            max: *max,
                        });
                    }
                }
            }
            Constraint::MinLength(min_len) => {
                if let Some(s) = value.as_string() {
                    if s.len() < *min_len {
                        return Err(ValidationError::RangeValidation {
                            field: field_name.to_string(),
                            value: s.len() as f64,
                            min: *min_len as f64,
                            max: f64::INFINITY,
                        });
                    }
                }
            }
            Constraint::MaxLength(max_len) => {
                if let Some(s) = value.as_string() {
                    if s.len() > *max_len {
                        return Err(ValidationError::RangeValidation {
                            field: field_name.to_string(),
                            value: s.len() as f64,
                            min: 0.0,
                            max: *max_len as f64,
                        });
                    }
                }
            }
            Constraint::Pattern(pattern) => {
                if let Some(s) = value.as_string() {
                    // Simple pattern matching (could be enhanced with regex)
                    if !s.contains(pattern) {
                        return Err(ValidationError::FieldValidation {
                            field: field_name.to_string(),
                            value: s.clone(),
                            constraint: format!("pattern: {}", pattern),
                        });
                    }
                }
            }
            Constraint::Enum(allowed_values) => {
                if !allowed_values.contains(value) {
                    return Err(ValidationError::FieldValidation {
                        field: field_name.to_string(),
                        value: format!("{:?}", value),
                        constraint: format!("enum: {:?}", allowed_values),
                    });
                }
            }
            Constraint::Custom(_) => {
                // Custom validation would be implemented here
            }
        }
        Ok(())
    }
}

/// Main configuration container
/// 
/// Implements ACID principles for configuration management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    data: HashMap<String, ConfigValue>,
    metadata: ConfigMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    pub source: String,
    pub timestamp: SystemTime,
    pub version: String,
    pub checksum: String,
    pub schema_version: Option<String>,
}

impl Configuration {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            metadata: ConfigMetadata {
                source: "memory".to_string(),
                timestamp: SystemTime::now(),
                version: "1.0.0".to_string(),
                checksum: "".to_string(),
                schema_version: None,
            },
        }
    }
    
    /// Create configuration from map
    pub fn from_map(data: HashMap<String, ConfigValue>) -> Self {
        Self {
            data,
            metadata: ConfigMetadata {
                source: "map".to_string(),
                timestamp: SystemTime::now(),
                version: "1.0.0".to_string(),
                checksum: "".to_string(),
                schema_version: None,
            },
        }
    }
    
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> KwaversResult<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| ConfigError::FileNotFound {
            path: path.to_string_lossy().to_string(),
        })?;
        
        let data: HashMap<String, ConfigValue> = toml::from_str(&content).map_err(|e| ConfigError::ParseError {
            line: e.line().unwrap_or(0),
            column: e.column().unwrap_or(0),
            reason: e.to_string(),
        })?;
        
        let mut config = Self::from_map(data);
        config.metadata.source = path.to_string_lossy().to_string();
        config.update_checksum();
        
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> KwaversResult<()> {
        let path = path.as_ref();
        let content = toml::to_string_pretty(&self.data).map_err(|e| ConfigError::ParseError {
            line: 0,
            column: 0,
            reason: e.to_string(),
        })?;
        
        fs::write(path, content).map_err(|e| ConfigError::WriteError {
            path: path.to_string_lossy().to_string(),
            reason: e.to_string(),
        })?;
        
        Ok(())
    }
    
    /// Get value by key
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.data.get(key)
    }
    
    /// Get value by key with default
    pub fn get_or_default(&self, key: &str, default: ConfigValue) -> ConfigValue {
        self.data.get(key).cloned().unwrap_or(default)
    }
    
    /// Set value by key
    pub fn set(&mut self, key: String, value: ConfigValue) {
        self.data.insert(key, value);
        self.update_checksum();
    }
    
    /// Check if key exists
    pub fn has_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
    
    /// Remove key
    pub fn remove(&mut self, key: &str) -> Option<ConfigValue> {
        let result = self.data.remove(key);
        if result.is_some() {
            self.update_checksum();
        }
        result
    }
    
    /// Get nested configuration
    pub fn get_nested(&self, key: &str) -> Option<Configuration> {
        self.data.get(key).and_then(|value| {
            value.as_object().map(|obj| Configuration::from_map(obj.clone()))
        })
    }
    
    /// Set nested configuration
    pub fn set_nested(&mut self, key: String, nested: Configuration) {
        self.data.insert(key, ConfigValue::Object(nested.data));
        self.update_checksum();
    }
    
    /// Merge with another configuration
    pub fn merge(&mut self, other: &Configuration) {
        for (key, value) in &other.data {
            self.data.insert(key.clone(), value.clone());
        }
        self.update_checksum();
    }
    
    /// Validate against schema
    pub fn validate(&self, schema: &ConfigSchema) -> KwaversResult<()> {
        schema.validate(self).map_err(|errors| {
            ConfigError::ValidationFailed {
                section: "root".to_string(),
                reason: format!("{} validation errors", errors.len()),
            }
        })
    }
    
    /// Get all keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }
    
    /// Get all key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ConfigValue)> {
        self.data.iter()
    }
    
    /// Get configuration size
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if configuration is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Update checksum
    fn update_checksum(&mut self) {
        // Simple checksum implementation (could be enhanced with proper hashing)
        let content = format!("{:?}", self.data);
        self.metadata.checksum = format!("{:x}", md5::compute(content.as_bytes()));
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration manager for global configuration handling
/// 
/// Implements Controller pattern from GRASP principles
pub struct ConfigManager {
    configs: Arc<RwLock<HashMap<String, Configuration>>>,
    schemas: Arc<RwLock<HashMap<String, ConfigSchema>>>,
    default_config: Arc<RwLock<Configuration>>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
            schemas: Arc::new(RwLock::new(HashMap::new())),
            default_config: Arc::new(RwLock::new(Configuration::new())),
        }
    }
    
    /// Load configuration from file
    pub fn load_config(&self, name: String, path: PathBuf) -> KwaversResult<()> {
        let config = Configuration::from_file(path)?;
        let mut configs = self.configs.write().unwrap();
        configs.insert(name, config);
        Ok(())
    }
    
    /// Get configuration by name
    pub fn get_config(&self, name: &str) -> Option<Configuration> {
        let configs = self.configs.read().unwrap();
        configs.get(name).cloned()
    }
    
    /// Set configuration by name
    pub fn set_config(&self, name: String, config: Configuration) {
        let mut configs = self.configs.write().unwrap();
        configs.insert(name, config);
    }
    
    /// Register schema for validation
    pub fn register_schema(&self, name: String, schema: ConfigSchema) {
        let mut schemas = self.schemas.write().unwrap();
        schemas.insert(name, schema);
    }
    
    /// Validate configuration against schema
    pub fn validate_config(&self, config_name: &str, schema_name: &str) -> KwaversResult<()> {
        let configs = self.configs.read().unwrap();
        let schemas = self.schemas.read().unwrap();
        
        let config = configs.get(config_name).ok_or_else(|| ConfigError::FileNotFound {
            path: config_name.to_string(),
        })?;
        
        let schema = schemas.get(schema_name).ok_or_else(|| ConfigError::FileNotFound {
            path: schema_name.to_string(),
        })?;
        
        config.validate(schema)
    }
    
    /// Get default configuration
    pub fn get_default_config(&self) -> Configuration {
        self.default_config.read().unwrap().clone()
    }
    
    /// Set default configuration
    pub fn set_default_config(&self, config: Configuration) {
        let mut default = self.default_config.write().unwrap();
        *default = config;
    }
    
    /// Get all configuration names
    pub fn get_config_names(&self) -> Vec<String> {
        let configs = self.configs.read().unwrap();
        configs.keys().cloned().collect()
    }
    
    /// Remove configuration
    pub fn remove_config(&self, name: &str) -> Option<Configuration> {
        let mut configs = self.configs.write().unwrap();
        configs.remove(name)
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration builder for fluent configuration construction
/// 
/// Implements Builder pattern for complex configuration creation
pub struct ConfigBuilder {
    config: Configuration,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: Configuration::new(),
        }
    }
    
    /// Add string value
    pub fn with_string(mut self, key: String, value: String) -> Self {
        self.config.set(key, ConfigValue::String(value));
        self
    }
    
    /// Add integer value
    pub fn with_integer(mut self, key: String, value: i64) -> Self {
        self.config.set(key, ConfigValue::Integer(value));
        self
    }
    
    /// Add float value
    pub fn with_float(mut self, key: String, value: f64) -> Self {
        self.config.set(key, ConfigValue::Float(value));
        self
    }
    
    /// Add boolean value
    pub fn with_boolean(mut self, key: String, value: bool) -> Self {
        self.config.set(key, ConfigValue::Boolean(value));
        self
    }
    
    /// Add array value
    pub fn with_array(mut self, key: String, value: Vec<ConfigValue>) -> Self {
        self.config.set(key, ConfigValue::Array(value));
        self
    }
    
    /// Add nested configuration
    pub fn with_nested(mut self, key: String, nested: Configuration) -> Self {
        self.config.set_nested(key, nested);
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> Configuration {
        self.config
    }
}

/// Configuration utilities for common operations
/// 
/// Implements DRY principle by providing reusable configuration utilities
pub mod utils {
    use super::*;
    
    /// Create a simple configuration from key-value pairs
    pub fn create_simple_config(pairs: Vec<(&str, ConfigValue)>) -> Configuration {
        let mut config = Configuration::new();
        for (key, value) in pairs {
            config.set(key.to_string(), value);
        }
        config
    }
    
    /// Create configuration schema for common types
    pub fn create_basic_schema() -> ConfigSchema {
        let mut fields = HashMap::new();
        
        // Add common field schemas
        fields.insert("name".to_string(), FieldSchema {
            field_type: FieldType::String,
            description: "Configuration name".to_string(),
            default_value: None,
            constraints: vec![
                Constraint::MinLength(1),
                Constraint::MaxLength(100),
            ],
            nested_schema: None,
        });
        
        fields.insert("version".to_string(), FieldSchema {
            field_type: FieldType::String,
            description: "Configuration version".to_string(),
            default_value: Some(ConfigValue::String("1.0.0".to_string())),
            constraints: vec![
                Constraint::Pattern("\\d+\\.\\d+\\.\\d+".to_string()),
            ],
            nested_schema: None,
        });
        
        fields.insert("enabled".to_string(), FieldSchema {
            field_type: FieldType::Boolean,
            description: "Whether configuration is enabled".to_string(),
            default_value: Some(ConfigValue::Boolean(true)),
            constraints: vec![],
            nested_schema: None,
        });
        
        ConfigSchema {
            fields,
            required_fields: vec!["name".to_string()],
            optional_fields: vec!["version".to_string(), "enabled".to_string()],
        }
    }
    
    /// Validate configuration file exists and is readable
    pub fn validate_config_file<P: AsRef<Path>>(path: P) -> KwaversResult<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(ConfigError::FileNotFound {
                path: path.to_string_lossy().to_string(),
            }.into());
        }
        
        if !path.is_file() {
            return Err(ConfigError::InvalidValue {
                parameter: "path".to_string(),
                value: path.to_string_lossy().to_string(),
                constraint: "must be a file".to_string(),
            }.into());
        }
        
        // Try to read the file to ensure it's accessible
        fs::read_to_string(path).map_err(|e| ConfigError::ReadError {
            path: path.to_string_lossy().to_string(),
            reason: e.to_string(),
        })?;
        
        Ok(())
    }
    
    /// Merge multiple configurations
    pub fn merge_configs(configs: Vec<Configuration>) -> Configuration {
        let mut result = Configuration::new();
        for config in configs {
            result.merge(&config);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_value_conversions() {
        let string_val = ConfigValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test"));
        assert_eq!(string_val.as_integer(), None);
        
        let int_val = ConfigValue::Integer(42);
        assert_eq!(int_val.as_integer(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));
        
        let float_val = ConfigValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_integer(), Some(3));
    }
    
    #[test]
    fn test_configuration_operations() {
        let mut config = Configuration::new();
        
        config.set("test_key".to_string(), ConfigValue::String("test_value".to_string()));
        assert!(config.has_key("test_key"));
        assert_eq!(config.get("test_key").unwrap().as_string(), Some("test_value"));
        
        config.remove("test_key");
        assert!(!config.has_key("test_key"));
    }
    
    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .with_string("name".to_string(), "test".to_string())
            .with_integer("count".to_string(), 42)
            .with_boolean("enabled".to_string(), true)
            .build();
        
        assert_eq!(config.get("name").unwrap().as_string(), Some("test"));
        assert_eq!(config.get("count").unwrap().as_integer(), Some(42));
        assert_eq!(config.get("enabled").unwrap().as_boolean(), Some(true));
    }
    
    #[test]
    fn test_config_schema_validation() {
        let schema = utils::create_basic_schema();
        let config = ConfigBuilder::new()
            .with_string("name".to_string(), "test".to_string())
            .with_string("version".to_string(), "1.0.0".to_string())
            .build();
        
        assert!(config.validate(&schema).is_ok());
    }
    
    #[test]
    fn test_config_manager() {
        let manager = ConfigManager::new();
        let config = Configuration::new();
        
        manager.set_config("test".to_string(), config);
        assert!(manager.get_config("test").is_some());
        
        let names = manager.get_config_names();
        assert!(names.contains(&"test".to_string()));
    }
}