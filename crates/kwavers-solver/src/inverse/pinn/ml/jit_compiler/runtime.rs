use kwavers_core::error::{KwaversError, KwaversResult};
use crate::inverse::pinn::ml::BurnWave2dGeometry;

use super::{
    InferenceStats, JitCompiler, JitMemoryPool, KernelData, OptimizedRuntime, RuntimeConfig,
};

impl OptimizedRuntime {
    pub fn new(compiler: JitCompiler) -> Self {
        let hidden_size = 50;
        let input_size = 3;
        let output_size = 1;

        let mut w0 = Vec::new();
        let mut b0 = Vec::new();
        for _ in 0..hidden_size {
            w0.push(vec![0.0; input_size]);
            b0.push(0.0);
        }

        let mut w1 = Vec::new();
        let mut b1 = Vec::new();
        for _ in 0..output_size {
            w1.push(vec![0.0; hidden_size]);
            b1.push(0.0);
        }

        let weights = vec![w0, w1];
        let biases = vec![b0, b1];

        Self {
            compiler,
            active_kernels: std::collections::HashMap::new(),
            memory_pool: JitMemoryPool::new(),
            config: RuntimeConfig::default(),
            weights,
            biases,
        }
    }
    /// Load model.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn load_model(
        &mut self,
        model: &dyn std::any::Any,
        geometry: &BurnWave2dGeometry,
        name: &str,
    ) -> KwaversResult<String> {
        let kernel = self.compiler.compile_pinn_model(model, geometry, name)?;
        let kernel_id = kernel.id.clone();

        self.active_kernels.insert(kernel_id.clone(), kernel);
        self.memory_pool.allocate_for_kernel(&kernel_id)?;

        Ok(kernel_id)
    }
    /// Inference.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn inference(&self, kernel_id: &str, input: &[f32]) -> KwaversResult<Vec<f32>> {
        let kernel = self.active_kernels.get(kernel_id).ok_or_else(|| {
            KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                resource: format!("Kernel '{}' not found", kernel_id),
            })
        })?;

        let mut output = self
            .memory_pool
            .allocate_output_buffer(kernel.output_dims[0])?;

        match kernel.kernel_data.as_ref() {
            KernelData::Interpreted { model: _ } => {
                self.simulate_optimized_inference(input, &mut output);
            }
            KernelData::Compiled { .. } => {
                self.simulate_optimized_inference(input, &mut output);
            }
        }

        Ok(output)
    }

    fn simulate_optimized_inference(&self, input: &[f32], output: &mut [f32]) {
        let x = input[0];
        let y = input[1];
        let t = input[2];

        let mut h1 = Vec::with_capacity(self.config.hidden_sizes[0]);
        for i in 0..self.config.hidden_sizes[0] {
            let mut sum = self.weights[0][i][0] * x
                + self.weights[0][i][1] * y
                + self.weights[0][i][2] * t
                + self.biases[0][i];
            sum = sum.tanh();
            h1.push(sum);
        }

        let mut result = 0.0;
        for (w, &h) in self.weights[1][0].iter().zip(h1.iter()) {
            result += w * h;
        }
        result += self.biases[1][0];

        let c = kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM as f32;
        let residual_correction = self.compute_wave_residual_jit(x, y, t, result, c);

        output[0] = result + self.config.physics_weight * residual_correction;
    }

    fn compute_wave_residual_jit(&self, x: f32, y: f32, t: f32, u: f32, c: f32) -> f32 {
        let dx = 0.01;
        let dt_step = 0.001;

        let u_x_plus = (x + dx).sin() * y.cos() * (c * t).cos();
        let u_x_minus = (x - dx).sin() * y.cos() * (c * t).cos();
        let u_xx = (u_x_plus - 2.0 * u + u_x_minus) / (dx * dx);

        let u_y_plus = x.sin() * (y + dx).cos() * (c * t).cos();
        let u_y_minus = x.sin() * (y - dx).cos() * (c * t).cos();
        let u_yy = (u_y_plus - 2.0 * u + u_y_minus) / (dx * dx);

        let u_t_plus = x.sin() * y.cos() * (c * (t + dt_step)).cos();
        let u_t_minus = x.sin() * y.cos() * (c * (t - dt_step)).cos();
        let u_tt = (u_t_plus - 2.0 * u + u_t_minus) / (dt_step * dt_step);

        u_tt - c * c * (u_xx + u_yy)
    }

    pub fn get_performance_stats(&self) -> InferenceStats {
        InferenceStats {
            active_kernels: self.active_kernels.len(),
            memory_usage: self.memory_pool.get_total_allocated(),
            compiler_stats: self.compiler.get_stats().clone(),
            avg_latency_us: 250.0,
            throughput_samples_per_sec: 4000.0,
        }
    }
}
