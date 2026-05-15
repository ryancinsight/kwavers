use std::f64::consts::PI;

/// Configuration for the 2D linearised-FDTD standing-wave suppression solver.
///
/// Grid geometry (default):
/// ```text
/// x=0  PML  source(11)       focus(68)  layer(90-96)  PML  x=127
///      |     [≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡|////|            ]
///      |     12-element linear array     bone-like slab    |
/// ```
///
/// Reflection coefficient at default layer: R ≈ 0.322.
#[derive(Debug, Clone)]
pub struct StandingWaveOptConfig {
    // Grid
    pub nx: usize,
    pub ny: usize,
    pub dx_m: f64,
    pub pml_cells: usize,

    // Medium
    pub c_ref_m_s: f64,
    pub rho_ref_kg_m3: f64,
    pub c_layer_m_s: f64,
    pub rho_layer_kg_m3: f64,
    pub layer_x_start: usize,
    pub layer_x_end: usize,

    // Acoustics
    pub frequency_hz: f64,
    pub cfl: f64,

    // Array / focus
    pub source_x: usize,
    pub focus_x: usize,
    pub focus_y: usize,
    pub n_elements: usize,
    pub element_y_min: usize,
    pub element_y_max: usize,
    pub focal_radius_cells: usize,

    // FDTD drive
    pub burst_cycles: f64,
    pub accum_skip_cycles: f64,
    pub swi_axis_half_width: usize,

    // Optimization
    pub n_opt_iter: usize,
    pub swi_weight: f64,
    pub focal_weight: f64,
    pub grad_delta_rad: f64,
    pub armijo_c1: f64,
    pub line_search_alpha0: f64,
    pub line_search_beta: f64,
    pub line_search_max: usize,

    // Output
    pub n_snapshots: usize,
}

impl Default for StandingWaveOptConfig {
    fn default() -> Self {
        Self {
            nx: 128,
            ny: 64,
            dx_m: 7.5e-4,
            pml_cells: 10,
            c_ref_m_s: 1540.0,
            rho_ref_kg_m3: 1000.0,
            c_layer_m_s: 2000.0,
            rho_layer_kg_m3: 1500.0,
            layer_x_start: 90,
            layer_x_end: 96,
            frequency_hz: 250_000.0,
            cfl: 0.25,
            source_x: 11,
            focus_x: 68,
            focus_y: 32,
            n_elements: 12,
            element_y_min: 12,
            element_y_max: 52,
            focal_radius_cells: 3,
            burst_cycles: 5.0,
            accum_skip_cycles: 2.0,
            swi_axis_half_width: 2,
            n_opt_iter: 25,
            swi_weight: 0.70,
            focal_weight: 0.30,
            grad_delta_rad: 0.05,
            armijo_c1: 0.01,
            line_search_alpha0: 1.0,
            line_search_beta: 0.5,
            line_search_max: 12,
            n_snapshots: 5,
        }
    }
}

impl StandingWaveOptConfig {
    pub fn omega(&self) -> f64 {
        2.0 * PI * self.frequency_hz
    }

    pub fn dt(&self) -> f64 {
        self.cfl * self.dx_m / (self.c_layer_m_s.max(self.c_ref_m_s) * 2.0_f64.sqrt())
    }

    pub fn element_ys(&self) -> Vec<usize> {
        let n = self.n_elements;
        (0..n)
            .map(|i| {
                let t = i as f64 / (n.saturating_sub(1)).max(1) as f64;
                (self.element_y_min as f64 + t * (self.element_y_max - self.element_y_min) as f64)
                    .round() as usize
            })
            .collect()
    }

    /// Delay-and-sum focusing phases in radians.
    pub fn das_phases(&self, element_ys: &[usize]) -> Vec<f64> {
        let omega = self.omega();
        let dx = self.dx_m;
        let c = self.c_ref_m_s;
        let fx = self.focus_x as f64;
        let fy = self.focus_y as f64;
        let sx = self.source_x as f64;
        let distances: Vec<f64> = element_ys
            .iter()
            .map(|&ey| {
                let dy = (ey as f64 - fy) * dx;
                let ddx = (fx - sx) * dx;
                (dy * dy + ddx * ddx).sqrt()
            })
            .collect();
        let t_max = distances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        distances
            .iter()
            .map(|&d| -omega * (t_max / c - d / c))
            .collect()
    }
}
