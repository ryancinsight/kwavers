//! Parameter types and definitions for the control system

use std::fmt;

/// Parameter types supported by the interactive control system
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterType {
    /// Floating point parameter with range
    Float { min: f64, max: f64, step: f64 },
    /// Integer parameter with range
    Integer { min: i64, max: i64, step: i64 },
    /// Boolean toggle parameter
    Boolean,
    /// Enumeration with predefined values
    Enum { options: Vec<String> },
    /// 3D vector parameter
    Vector3 { min: f64, max: f64, step: f64 },
    /// Color parameter (RGB)
    Color,
}

/// Parameter definition for the control system
#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub parameter_type: ParameterType,
    pub default_value: ParameterValue,
    pub group: String,
    pub is_realtime: bool,
}

/// Parameter value storage
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    Enum(String),
    Vector3([f64; 3]),
    Color([f32; 3]),
}

impl ParameterValue {
    /// Convert to float if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Convert to integer if possible
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Convert to boolean if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    /// Convert to vector3 if possible
    pub fn as_vector3(&self) -> Option<[f64; 3]> {
        match self {
            Self::Vector3(v) => Some(*v),
            _ => None,
        }
    }

    /// Convert to color if possible
    pub fn as_color(&self) -> Option<[f32; 3]> {
        match self {
            Self::Color(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for ParameterValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{:.3}", v),
            Self::Integer(v) => write!(f, "{}", v),
            Self::Boolean(v) => write!(f, "{}", v),
            Self::Enum(v) => write!(f, "{}", v),
            Self::Vector3(v) => write!(f, "[{:.2}, {:.2}, {:.2}]", v[0], v[1], v[2]),
            Self::Color(v) => write!(
                f,
                "RGB({:.0}, {:.0}, {:.0})",
                v[0] * 255.0,
                v[1] * 255.0,
                v[2] * 255.0
            ),
        }
    }
}

impl ParameterDefinition {
    /// Create a float parameter definition
    pub fn float(
        name: impl Into<String>,
        display_name: impl Into<String>,
        min: f64,
        max: f64,
        default: f64,
    ) -> Self {
        Self {
            name: name.into(),
            display_name: display_name.into(),
            description: String::new(),
            parameter_type: ParameterType::Float {
                min,
                max,
                step: (max - min) / 100.0,
            },
            default_value: ParameterValue::Float(default),
            group: "General".to_string(),
            is_realtime: true,
        }
    }

    /// Create a boolean parameter definition
    pub fn boolean(
        name: impl Into<String>,
        display_name: impl Into<String>,
        default: bool,
    ) -> Self {
        Self {
            name: name.into(),
            display_name: display_name.into(),
            description: String::new(),
            parameter_type: ParameterType::Boolean,
            default_value: ParameterValue::Boolean(default),
            group: "General".to_string(),
            is_realtime: true,
        }
    }

    /// Set the description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the group
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = group.into();
        self
    }

    /// Set whether the parameter updates in realtime
    pub fn with_realtime(mut self, realtime: bool) -> Self {
        self.is_realtime = realtime;
        self
    }
}
