use super::{CheckpointManager, DistributedTrainingConfig};
use crate::core::error::KwaversResult;

impl CheckpointManager {
    pub(super) fn from_config(config: &DistributedTrainingConfig) -> Self {
        Self {
            checkpoint_dir: std::path::PathBuf::from(&config.checkpoint_config.directory),
            max_checkpoints: config.checkpoint_config.max_checkpoints,
            checkpoint_interval: config.checkpoint_config.interval,
            auto_save: config.checkpoint_config.auto_save,
        }
    }

    pub fn ensure_checkpoint_dir(&self) -> KwaversResult<()> {
        if !self.checkpoint_dir.exists() {
            std::fs::create_dir_all(&self.checkpoint_dir)?;
        }
        Ok(())
    }

    pub fn list_checkpoints(&self) -> KwaversResult<Vec<usize>> {
        let mut checkpoints = Vec::new();

        if self.checkpoint_dir.exists() {
            for entry in std::fs::read_dir(&self.checkpoint_dir)? {
                let entry = entry?;
                let filename_owned = entry.file_name().to_string_lossy().to_string();

                if filename_owned.starts_with("checkpoint_epoch_")
                    && filename_owned.ends_with(".bin")
                {
                    if let Some(epoch_str) = filename_owned
                        .strip_prefix("checkpoint_epoch_")
                        .and_then(|s| s.strip_suffix(".bin"))
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

    pub fn cleanup_old_checkpoints(&self) -> KwaversResult<()> {
        let checkpoints = self.list_checkpoints()?;
        let to_remove = checkpoints.len().saturating_sub(self.max_checkpoints);

        for &epoch in checkpoints.iter().take(to_remove) {
            let filename = format!("checkpoint_epoch_{}.bin", epoch);
            let path = self.checkpoint_dir.join(filename);
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }

        Ok(())
    }
}
