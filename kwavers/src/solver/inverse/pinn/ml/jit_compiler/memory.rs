use crate::core::error::KwaversResult;

use super::MemoryPool;

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            buffer_sizes: vec![64, 128, 256, 512, 1024, 2048, 4096],
            _current_index: 0,
        }
    }

    pub fn allocate_for_kernel(&mut self, _kernel_id: &str) -> KwaversResult<()> {
        for &size in &self.buffer_sizes {
            self.buffers.push(vec![0.0; size]);
        }
        Ok(())
    }

    pub fn allocate_output_buffer(&self, size: usize) -> KwaversResult<Vec<f32>> {
        let buffer_size = self
            .buffer_sizes
            .iter()
            .find(|&&s| s >= size)
            .copied()
            .unwrap_or(size);

        Ok(vec![0.0; buffer_size])
    }

    pub fn get_total_allocated(&self) -> usize {
        self.buffers
            .iter()
            .map(|b| b.len() * std::mem::size_of::<f32>())
            .sum()
    }
}
