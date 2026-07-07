//! `FourierFeatures`: frequency-domain input embedding for PINNs.

use coeus_autograd::Var;

/// Fourier feature embedding for improved frequency representation.
///
/// Fourier features provide better representation of oscillatory physics by mapping
/// input coordinates to higher-frequency sinusoidal functions.
pub struct FourierFeatures<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Learned frequency parameters, shape `[input_dim, num_features]`.
    frequencies: Var<f32, B>,
    /// Feature scaling factor.
    scale: f32,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for FourierFeatures<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FourierFeatures")
            .field("scale", &self.scale)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> FourierFeatures<B> {
    /// Create Fourier feature embedding.
    pub fn new(input_dim: usize, num_features: usize, scale: f32) -> Self {
        let backend = B::default();
        let freq_data: Vec<f32> = (0..input_dim * num_features)
            .map(|_| rand::random::<f32>() * scale)
            .collect();

        let frequencies = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![input_dim, num_features], &freq_data, &backend),
            true,
        );

        Self { frequencies, scale }
    }

    /// Flatten the learnable frequency parameters.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![self.frequencies.clone()]
    }

    /// Write updated frequency values back (optimizer round-trip).
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        self.frequencies = params[0].clone();
    }

    /// Apply Fourier feature transformation.
    ///
    /// Maps `x` ∈ ℝ^{batch × input_dim} to
    /// `[cos(scale·x·f), sin(scale·x·f)]` ∈ ℝ^{batch × 2·num_features}.
    pub fn forward(&self, x: &Var<f32, B>) -> Var<f32, B>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let scaled_x = coeus_autograd::scalar_mul(x, self.scale);
        let features = coeus_autograd::matmul(&scaled_x, &self.frequencies);

        let cos_features = coeus_autograd::cos(&features);
        let sin_features = coeus_autograd::sin(&features);

        coeus_autograd::cat(&[&cos_features, &sin_features], 1)
    }
}
