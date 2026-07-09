use super::{CheckpointManager, DistributedTrainingConfig};
use kwavers_core::error::KwaversResult;

const CHECKPOINT_PREFIX: &str = "checkpoint_epoch_";
const CHECKPOINT_SUFFIX: &str = ".json";

pub(crate) fn checkpoint_filename(epoch: usize) -> String {
    format!("{CHECKPOINT_PREFIX}{epoch}{CHECKPOINT_SUFFIX}")
}

impl CheckpointManager {
    /// From config.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn from_config(config: &DistributedTrainingConfig) -> Self {
        Self {
            checkpoint_dir: std::path::PathBuf::from(&config.checkpoint_config.directory),
            max_checkpoints: config.checkpoint_config.max_checkpoints,
        }
    }
    /// Ensure checkpoint dir.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn ensure_checkpoint_dir(&self) -> KwaversResult<()> {
        if !self.checkpoint_dir.exists() {
            std::fs::create_dir_all(&self.checkpoint_dir)?;
        }
        Ok(())
    }
    /// List checkpoints.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn list_checkpoints(&self) -> KwaversResult<Vec<usize>> {
        let mut checkpoints = Vec::new();

        if self.checkpoint_dir.exists() {
            for entry in std::fs::read_dir(&self.checkpoint_dir)? {
                let entry = entry?;
                let filename_owned = entry.file_name().to_string_lossy().to_string();

                if filename_owned.starts_with(CHECKPOINT_PREFIX)
                    && filename_owned.ends_with(CHECKPOINT_SUFFIX)
                {
                    if let Some(epoch_str) = filename_owned
                        .strip_prefix(CHECKPOINT_PREFIX)
                        .and_then(|s| s.strip_suffix(CHECKPOINT_SUFFIX))
                    {
                        if let Ok(epoch) = epoch_str.parse::<usize>() {
                            checkpoints.push(epoch);
                        }
                    }
                }
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }
    /// Cleanup old checkpoints.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn cleanup_old_checkpoints(&self) -> KwaversResult<()> {
        let checkpoints = self.list_checkpoints()?;
        let to_remove = (checkpoints.shape()[0] * checkpoints.shape()[1] * checkpoints.shape()[2]).saturating_sub(self.max_checkpoints);

        for &epoch in checkpoints.iter().take(to_remove) {
            let path = self.checkpoint_dir.join(checkpoint_filename(epoch));
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }

        Ok(())
    }
}
