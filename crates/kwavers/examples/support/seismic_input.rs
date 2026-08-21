//! Explicit input selections shared by the seismic example workflows.

use std::path::PathBuf;
use std::str::FromStr;

/// Selects the source data for a seismic workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeismicInputMode {
    /// Use the deterministic analytical phantom shipped with the example.
    Synthetic,
    /// Load a CT volume from the supplied path.
    Ct(PathBuf),
    /// Load co-registered CT and T1 MRI volumes from the supplied paths.
    CtMri { ct: PathBuf, mri: PathBuf },
}

impl SeismicInputMode {
    /// Read a mode from an environment variable, defaulting to the synthetic mode.
    pub fn from_env(variable: &str) -> Result<Self, String> {
        let value = match std::env::var(variable) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "synthetic".to_owned(),
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(format!("{variable} contains invalid Unicode"));
            }
        };
        Self::from_str(&value)
    }
}

impl FromStr for SeismicInputMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("synthetic") {
            return Ok(Self::Synthetic);
        }

        let Some(path) = value.strip_prefix("ct:") else {
            if let Some(paths) = value.strip_prefix("ct-mri:") {
                let (ct, mri) = paths.split_once('|').ok_or_else(|| {
                    "ct-mri input must use ct-mri:<ct-path>|<mri-path>".to_owned()
                })?;
                if ct.is_empty() || mri.is_empty() {
                    return Err("ct-mri input paths must not be empty".to_owned());
                }
                return Ok(Self::CtMri {
                    ct: PathBuf::from(ct),
                    mri: PathBuf::from(mri),
                });
            }
            return Err(format!(
                "unsupported {value:?} input; expected synthetic, ct:<path>, or ct-mri:<ct>|<mri>"
            ));
        };
        if path.is_empty() {
            return Err("ct input path must not be empty".to_owned());
        }
        Ok(Self::Ct(PathBuf::from(path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_input_modes() {
        assert_eq!("synthetic".parse(), Ok(SeismicInputMode::Synthetic));
        assert_eq!(
            "ct:head.nii.gz".parse(),
            Ok(SeismicInputMode::Ct(PathBuf::from("head.nii.gz")))
        );
        assert_eq!(
            "ct-mri:head.nii.gz|brain.nii.gz".parse(),
            Ok(SeismicInputMode::CtMri {
                ct: PathBuf::from("head.nii.gz"),
                mri: PathBuf::from("brain.nii.gz"),
            })
        );
    }

    #[test]
    fn rejects_ambiguous_input_modes() {
        assert!("ct".parse::<SeismicInputMode>().is_err());
        assert!("ct-mri:head.nii.gz".parse::<SeismicInputMode>().is_err());
        assert!("unknown".parse::<SeismicInputMode>().is_err());
    }
}
