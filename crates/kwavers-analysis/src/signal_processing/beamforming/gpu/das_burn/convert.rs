use burn::tensor::{backend::Backend, Tensor};
use ndarray::Array3;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::BurnDasBeamformer;

impl<B: Backend> BurnDasBeamformer<B> {
    /// Array to tensor 3d.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn array_to_tensor_3d(&self, array: &Array3<f64>) -> KwaversResult<Tensor<B, 3>> {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
        Ok(Tensor::<B, 1>::from_data(data.as_slice(), &self.device)
            .reshape([shape[0], shape[1], shape[2]]))
    }
    /// Array to tensor 2d.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn array_to_tensor_2d(
        &self,
        array: &ndarray::Array2<f64>,
    ) -> KwaversResult<Tensor<B, 2>> {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
        Ok(Tensor::<B, 1>::from_data(data.as_slice(), &self.device).reshape([shape[0], shape[1]]))
    }
    /// Tensor to array 3d.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn tensor_to_array_3d(&self, tensor: &Tensor<B, 3>) -> KwaversResult<Array3<f64>> {
        let shape = tensor.shape();
        let data = tensor.clone().into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::InvalidInput(format!("Tensor conversion failed: {:?}", e))
        })?;

        Array3::from_shape_vec(
            (shape.dims[0], shape.dims[1], shape.dims[2]),
            slice.iter().map(|&x| x as f64).collect(),
        )
        .map_err(|e| KwaversError::InvalidInput(format!("Array reshape failed: {}", e)))
    }
}
