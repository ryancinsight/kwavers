use coeus_autograd::Var;
use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::loss::CollocationData;

/// Iterator over mini-batches of collocation point indices.
#[derive(Debug)]
pub struct BatchIterator {
    indices: Vec<usize>,
    batch_size: usize,
    position: usize,
}

impl BatchIterator {
    pub(super) fn new(indices: Vec<usize>, batch_size: usize) -> Self {
        Self {
            indices,
            batch_size,
            position: 0,
        }
    }
}

impl Iterator for BatchIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch = self.indices[self.position..end].to_vec();
        self.position = end;
        Some(batch)
    }
}

/// Extract subset of collocation data by indices.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
pub fn extract_batch<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    data: &CollocationData<B>,
    indices: &[usize],
) -> KwaversResult<CollocationData<B>>
where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    if indices.is_empty() {
        return Err(KwaversError::InvalidInput("Empty batch indices".into()));
    }
    let backend = B::default();
    let idx_values: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
    let idx_var = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![idx_values.len()], &idx_values, &backend),
        false,
    );

    let x = coeus_autograd::index_select(&data.x, 0, &idx_var);
    let y = coeus_autograd::index_select(&data.y, 0, &idx_var);
    let t = coeus_autograd::index_select(&data.t, 0, &idx_var);
    let n = indices.len();
    let source_x = Var::new(coeus_tensor::Tensor::zeros_on(vec![n, 1], &backend), false);
    let source_y = Var::new(coeus_tensor::Tensor::zeros_on(vec![n, 1], &backend), false);
    Ok(CollocationData {
        x,
        y,
        t,
        source_x: Some(source_x),
        source_y: Some(source_y),
    })
}
