#[cfg(feature = "pinn")]
use super::super::loss::CollocationData;
#[cfg(feature = "pinn")]
use burn::tensor::Tensor;
#[cfg(feature = "pinn")]
use kwavers_core::error::{KwaversError, KwaversResult};

/// Iterator over mini-batches of collocation point indices.
#[cfg(feature = "pinn")]
#[derive(Debug)]
pub struct BatchIterator {
    indices: Vec<usize>,
    batch_size: usize,
    position: usize,
}

#[cfg(feature = "pinn")]
impl BatchIterator {
    pub(super) fn new(indices: Vec<usize>, batch_size: usize) -> Self {
        Self {
            indices,
            batch_size,
            position: 0,
        }
    }
}

#[cfg(feature = "pinn")]
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
///
#[cfg(feature = "pinn")]
pub fn extract_batch<B: burn::tensor::backend::AutodiffBackend>(
    data: &CollocationData<B>,
    indices: &[usize],
) -> KwaversResult<CollocationData<B>> {
    if indices.is_empty() {
        return Err(KwaversError::InvalidInput("Empty batch indices".into()));
    }
    let device = data.x.device();
    let idx_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(
        indices
            .iter()
            .map(|&i| i as i64)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );
    let x = data.x.clone().select(0, idx_tensor.clone());
    let y = data.y.clone().select(0, idx_tensor.clone());
    let t = data.t.clone().select(0, idx_tensor.clone());
    let source_x = Tensor::zeros_like(&x);
    let source_y = Tensor::zeros_like(&y);
    Ok(CollocationData {
        x,
        y,
        t,
        source_x: Some(source_x),
        source_y: Some(source_y),
    })
}
