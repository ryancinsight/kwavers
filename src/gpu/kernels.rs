//! GPU compute kernels

/// Collection of GPU compute kernels
#[derive(Debug)]
pub struct Kernels;

impl Kernels {
    /// Get FDTD kernel source
    pub fn fdtd() -> &'static str {
        include_str!("shaders/fdtd.wgsl")
    }

    /// Get k-space propagation kernel source
    pub fn kspace_propagate() -> &'static str {
        include_str!("shaders/kspace_propagate.wgsl")
    }

    /// Get FFT kernel source
    pub fn fft() -> &'static str {
        include_str!("shaders/fft.wgsl")
    }

    /// Get absorption kernel source
    pub fn absorption() -> &'static str {
        include_str!("shaders/absorption.wgsl")
    }

    /// Get PML boundary kernel source
    pub fn pml_boundary() -> &'static str {
        include_str!("shaders/pml.wgsl")
    }

    /// Get nonlinear propagation kernel source
    pub fn nonlinear() -> &'static str {
        include_str!("shaders/nonlinear.wgsl")
    }
}
