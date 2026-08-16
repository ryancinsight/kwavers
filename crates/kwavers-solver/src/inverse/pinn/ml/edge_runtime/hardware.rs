use super::{Architecture, EdgeRuntime, HardwareCapabilities};
use crate::inverse::pinn::ml::quantization::LayerInfo;

impl EdgeRuntime {
    pub(super) fn detect_hardware_capabilities() -> HardwareCapabilities {
        let architecture = if cfg!(target_arch = "aarch64") {
            Architecture::ARM64
        } else if cfg!(target_arch = "arm") {
            Architecture::ARM
        } else if cfg!(target_arch = "riscv64") {
            Architecture::RISCV
        } else if cfg!(target_arch = "x86") {
            Architecture::X86
        } else if cfg!(target_arch = "x86_64") {
            Architecture::X86_64
        } else {
            Architecture::Other(std::env::consts::ARCH.to_string())
        };

        // SIMD capability reporting delegates to hermes's runtime host detection
        // (the Atlas ISA-dispatch SSOT). `cfg!(target_feature)` reflects only the
        // build's baseline flags — a portable binary under-reports the host's real
        // vector units — while hermes probes the live CPU with its OS-enabled
        // register-state checks.
        let mut instruction_sets = Vec::new();
        if hermes_simd::TargetId::Avx512.is_supported() {
            instruction_sets.push("AVX512".to_string());
        }
        if hermes_simd::TargetId::Avx2.is_supported() {
            instruction_sets.push("AVX2".to_string());
        }
        if hermes_simd::TargetId::Neon.is_supported() {
            instruction_sets.push("NEON".to_string());
        }

        let simd_width: usize = if hermes_simd::TargetId::Avx512.is_supported() {
            512
        } else if hermes_simd::TargetId::Avx2.is_supported() {
            256
        } else if hermes_simd::TargetId::Neon.is_supported() {
            128
        } else {
            64
        };

        HardwareCapabilities {
            architecture,
            instruction_sets,
            total_memory_mb: 512,
            // FPU presence is an architecture-level concern, not SIMD dispatch,
            // so it stays a compile-time check rather than a hermes probe.
            has_fpu: cfg!(target_arch = "x86_64")
                || cfg!(target_arch = "aarch64")
                || cfg!(target_feature = "neon")
                || cfg!(target_feature = "sse2"),
            simd_width,
            cache_line_size: 64,
        }
    }

    pub fn get_hardware_caps(&self) -> &HardwareCapabilities {
        &self.hardware_caps
    }

    pub(super) fn estimate_kernel_time(&self, layer: &LayerInfo) -> f64 {
        let operations = layer.input_size * layer.output_size;

        let base_time_per_op = match self.hardware_caps.architecture {
            Architecture::ARM | Architecture::ARM64 => 0.01,
            Architecture::RISCV => 0.05,
            Architecture::X86 | Architecture::X86_64 => 0.005,
            Architecture::Other(_) => 0.02,
        };

        let simd_factor = self.hardware_caps.simd_width as f64 / 8.0;
        let adjusted_time = base_time_per_op / simd_factor;

        operations as f64 * adjusted_time
    }

    pub(super) fn estimate_kernel_memory(&self, layer: &LayerInfo) -> usize {
        let weight_memory = layer.input_size * layer.output_size * std::mem::size_of::<i8>();
        let bias_memory = layer.output_size * std::mem::size_of::<f32>();
        let intermediate_memory = layer.output_size * std::mem::size_of::<f32>() * 2;

        weight_memory + bias_memory + intermediate_memory
    }
}
