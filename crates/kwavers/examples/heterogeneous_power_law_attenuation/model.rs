#[derive(Debug, Clone, Copy)]
pub(crate) struct SweepRow {
    pub(crate) alpha0_db: f64,
    pub(crate) gamma: f64,
    pub(crate) frequency_hz: f64,
    pub(crate) prescribed_np_m: f64,
    pub(crate) measured_np_m: f64,
}

impl SweepRow {
    pub(crate) fn relative_error(&self) -> f64 {
        (self.measured_np_m - self.prescribed_np_m).abs() / self.prescribed_np_m
    }
}

/// One material layer on the propagation path.
pub(crate) struct Layer {
    pub(crate) name: &'static str,
    pub(crate) cells: usize,
    pub(crate) alpha0_db: f64,
    pub(crate) gamma: f64,
}
