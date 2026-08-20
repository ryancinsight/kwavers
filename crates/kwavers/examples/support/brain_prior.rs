//! Explicit brain-prior selections shared by the 2-D and 3-D seismic workflows.

use std::path::PathBuf;
use std::str::FromStr;

/// Selects the brain prior used after skull reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrainPriorMode {
    /// Use the deterministic homogeneous brain prior.
    Uniform,
    /// Load a MNI tissue-probability directory.
    Mni(PathBuf),
    /// Load a co-registered T1 MRI volume.
    T1(PathBuf),
    /// Load both MNI tissue probabilities and a T1 MRI volume.
    MniT1 { mni: PathBuf, t1: PathBuf },
}

impl BrainPriorMode {
    /// Read a brain-prior mode from an environment variable, defaulting to uniform.
    pub fn from_env(variable: &str) -> Result<Self, String> {
        let value = std::env::var(variable).unwrap_or_else(|_| "uniform".to_owned());
        Self::from_str(&value)
    }
}

impl FromStr for BrainPriorMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("uniform") {
            return Ok(Self::Uniform);
        }
        if let Some(path) = value.strip_prefix("mni:") {
            return non_empty_path(path, "mni").map(Self::Mni);
        }
        if let Some(path) = value.strip_prefix("t1:") {
            return non_empty_path(path, "t1").map(Self::T1);
        }
        if let Some(paths) = value.strip_prefix("mni-t1:") {
            let (mni, t1) = paths
                .split_once('|')
                .ok_or_else(|| "mni-t1 prior must use mni-t1:<mni-dir>|<t1-path>".to_owned())?;
            return Ok(Self::MniT1 {
                mni: non_empty_path(mni, "mni")?,
                t1: non_empty_path(t1, "t1")?,
            });
        }
        Err(format!(
            "unsupported {value:?} brain prior; expected uniform, mni:<dir>, t1:<path>, or mni-t1:<dir>|<path>"
        ))
    }
}

fn non_empty_path(value: &str, label: &str) -> Result<PathBuf, String> {
    (!value.is_empty())
        .then(|| PathBuf::from(value))
        .ok_or_else(|| format!("{label} prior path must not be empty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_brain_priors() {
        assert_eq!("uniform".parse(), Ok(BrainPriorMode::Uniform));
        assert_eq!(
            "mni:atlas".parse(),
            Ok(BrainPriorMode::Mni(PathBuf::from("atlas")))
        );
        assert_eq!(
            "mni-t1:atlas|brain.nii.gz".parse(),
            Ok(BrainPriorMode::MniT1 {
                mni: PathBuf::from("atlas"),
                t1: PathBuf::from("brain.nii.gz"),
            })
        );
        assert!("t1".parse::<BrainPriorMode>().is_err());
    }
}
