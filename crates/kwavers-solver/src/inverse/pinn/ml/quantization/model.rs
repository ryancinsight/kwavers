use super::{QuantizedData, QuantizedModel};

impl QuantizedModel {
    pub fn memory_usage(&self) -> usize {
        self.quantized_weights
            .iter()
            .map(|tensor| match &tensor.data {
                QuantizedData::F32(v) => (v.shape()[0] * v.shape()[1] * v.shape()[2]) * 4,
                QuantizedData::I8(v) => (v.shape()[0] * v.shape()[1] * v.shape()[2]),
            })
            .sum::<usize>()
    }

    pub fn compression_ratio(&self) -> f32 {
        let original_size: usize = self
            .original_layers
            .iter()
            .map(|l| l.input_size * l.output_size + l.output_size)
            .sum::<usize>()
            * 4;

        let quantized_size = self.memory_usage();
        original_size as f32 / quantized_size as f32
    }

    pub fn dequantize_layer(&self, _layer_name: &str) -> Option<Vec<f32>> {
        self.quantized_weights
            .iter()
            .find(|tensor| (tensor.shape.shape()[0] * tensor.shape.shape()[1] * tensor.shape.shape()[2]) > 1)
            .map(|tensor| tensor.dequantize())
    }
}
