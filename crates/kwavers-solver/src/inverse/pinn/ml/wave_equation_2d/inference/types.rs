/// Activation Function Types
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    /// Tanh activation (standard for PINNs)
    Tanh,
    /// ReLU activation (faster alternative)
    Relu,
    /// Linear activation (output layer)
    Linear,
}

/// Quantized Neural Network for Real-Time Inference
#[derive(Debug)]
pub struct QuantizedNetwork {
    /// Quantized weights for each layer [layer_idx][weight_idx]
    pub weights: Vec<Vec<i8>>,
    /// Quantization scales for weights [layer_idx]
    pub weight_scales: Vec<f32>,
    /// Quantized biases [layer_idx][bias_idx]
    pub biases: Vec<Vec<i8>>,
    /// Bias quantization scales [layer_idx]
    pub bias_scales: Vec<f32>,
    /// Layer sizes [input_size, hidden_sizes..., output_size]
    pub layer_sizes: Vec<usize>,
    /// Activation function type per layer
    pub activations: Vec<ActivationType>,
}

/// Memory Pool for Zero-Allocation Inference
#[derive(Debug)]
pub struct WaveInferenceMemoryPool2D {
    /// Pre-allocated buffers for intermediate activations
    pub buffers: Vec<Vec<f32>>,
    /// Buffer sizes for each layer
    pub _buffer_sizes: Vec<usize>,
}

/// SIMD Processor for CPU Vectorization
#[cfg(feature = "simd")]
#[derive(Debug)]
pub struct SIMDProcessor {
    /// SIMD lanes available (typically 16 for f32x16)
    pub lanes: usize,
}

/// Neural network state for GPU inference over a pre-quantized network.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PinnNeuralNetwork<B: coeus_ops::BackendOps<f32> + Default> {
    pub weights: Vec<coeus_tensor::Tensor<f32, B>>,
    pub biases: Vec<coeus_tensor::Tensor<f32, B>>,
    pub activation: String,
}
