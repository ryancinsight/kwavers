//! Formatting utilities for general use across the codebase
//! 
//! This module contains general-purpose formatting functions that are not
//! specific to any particular domain, following the separation of concerns principle.

use std::time::Duration;

/// Format a duration into a human-readable string
/// 
/// # Examples
/// ```
/// use std::time::Duration;
/// use kwavers::utils::format::format_duration;
/// 
/// assert_eq!(format_duration(Duration::from_secs(45)), "45s");
/// assert_eq!(format_duration(Duration::from_secs(125)), "2m 5s");
/// assert_eq!(format_duration(Duration::from_secs(3725)), "1h 2m 5s");
/// ```
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = duration.subsec_millis();
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else if seconds > 0 {
        format!("{}s", seconds)
    } else {
        format!("{}ms", millis)
    }
}

/// Format a large number with SI prefixes for readability
/// 
/// # Examples
/// ```
/// use kwavers::utils::format::format_si_number;
/// 
/// assert_eq!(format_si_number(1234.0), "1.23k");
/// assert_eq!(format_si_number(1234567.0), "1.23M");
/// assert_eq!(format_si_number(0.001234), "1.23m");
/// ```
pub fn format_si_number(value: f64) -> String {
    let abs_value = value.abs();
    
    let (prefix, scale) = if abs_value >= 1e12 {
        ("T", 1e12)
    } else if abs_value >= 1e9 {
        ("G", 1e9)
    } else if abs_value >= 1e6 {
        ("M", 1e6)
    } else if abs_value >= 1e3 {
        ("k", 1e3)
    } else if abs_value >= 1.0 {
        ("", 1.0)
    } else if abs_value >= 1e-3 {
        ("m", 1e-3)
    } else if abs_value >= 1e-6 {
        ("µ", 1e-6)
    } else if abs_value >= 1e-9 {
        ("n", 1e-9)
    } else {
        ("p", 1e-12)
    };
    
    format!("{:.2}{}", value / scale, prefix)
}

/// Format bytes into human-readable format
/// 
/// # Examples
/// ```
/// use kwavers::utils::format::format_bytes;
/// 
/// assert_eq!(format_bytes(1024), "1.00 KiB");
/// assert_eq!(format_bytes(1048576), "1.00 MiB");
/// ```
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let bytes = bytes as f64;
    let i = (bytes.ln() / 1024_f64.ln()).floor() as usize;
    let i = i.min(UNITS.len() - 1);
    let value = bytes / 1024_f64.powi(i as i32);
    
    if i == 0 {
        format!("{} {}", bytes as usize, UNITS[i])
    } else {
        format!("{:.2} {}", value, UNITS[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0ms");
        assert_eq!(format_duration(Duration::from_millis(500)), "0ms");
        assert_eq!(format_duration(Duration::from_secs(45)), "45s");
        assert_eq!(format_duration(Duration::from_secs(125)), "2m 5s");
        assert_eq!(format_duration(Duration::from_secs(3725)), "1h 2m 5s");
        assert_eq!(format_duration(Duration::from_secs(7325)), "2h 2m 5s");
    }
    
    #[test]
    fn test_format_si_number() {
        assert_eq!(format_si_number(1234.0), "1.23k");
        assert_eq!(format_si_number(1234567.0), "1.23M");
        assert_eq!(format_si_number(1234567890.0), "1.23G");
        assert_eq!(format_si_number(0.001234), "1.23");
        assert_eq!(format_si_number(0.000001234), "1.23µ");
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1536), "1.50 KiB");
        assert_eq!(format_bytes(1048576), "1.00 MiB");
        assert_eq!(format_bytes(1073741824), "1.00 GiB");
    }
}