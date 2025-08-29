//! Transfer statistics tracking

use std::time::Instant;

/// Data transfer statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics {
    pub total_bytes_transferred: usize,
    pub transfer_time_ms: f32,
    pub bandwidth_gb_per_sec: f32,
    pub num_transfers: usize,
    pub last_transfer_time: Instant,
}

impl Default for TransferStatistics {
    fn default() -> Self {
        Self {
            total_bytes_transferred: 0,
            transfer_time_ms: 0.0,
            bandwidth_gb_per_sec: 0.0,
            num_transfers: 0,
            last_transfer_time: Instant::now(),
        }
    }
}

impl TransferStatistics {
    /// Update statistics with a new transfer
    pub fn record_transfer(&mut self, bytes: usize, duration_ms: f32) {
        self.total_bytes_transferred += bytes;
        self.transfer_time_ms += duration_ms;
        self.num_transfers += 1;
        self.last_transfer_time = Instant::now();
        
        // Calculate bandwidth in GB/s
        if duration_ms > 0.0 {
            let gb = bytes as f32 / 1_073_741_824.0; // 1024^3
            let seconds = duration_ms / 1000.0;
            self.bandwidth_gb_per_sec = gb / seconds;
        }
    }
    
    /// Get average transfer size
    pub fn average_transfer_size(&self) -> usize {
        if self.num_transfers > 0 {
            self.total_bytes_transferred / self.num_transfers
        } else {
            0
        }
    }
    
    /// Get average transfer time
    pub fn average_transfer_time_ms(&self) -> f32 {
        if self.num_transfers > 0 {
            self.transfer_time_ms / self.num_transfers as f32
        } else {
            0.0
        }
    }
}