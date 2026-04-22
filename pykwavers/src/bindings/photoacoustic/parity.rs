/// Canonical parity references exposed by the Python photoacoustic binding surface.
pub const PHOTOACOUSTIC_PARITY_REFERENCES: &[&str] = &[
    "external/k-wave-python/examples/ivp_photoacoustic_waveforms",
    "external/k-wave-python/examples/pr_2D_FFT_line_sensor",
    "external/k-wave-python/examples/pr_3D_FFT_planar_sensor",
];

#[must_use]
pub fn parity_references() -> &'static [&'static str] {
    PHOTOACOUSTIC_PARITY_REFERENCES
}
