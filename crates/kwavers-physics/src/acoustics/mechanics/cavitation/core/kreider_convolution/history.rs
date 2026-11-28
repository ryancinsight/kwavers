/// Eller–Flynn convolution state: segments `(s_lo, s_hi, F)` in
/// transformed time `s = ∫R⁴dt` with product integration,
/// `J = ∫ F(s′) (s_now − s′)^(−1/2) ds′` evaluated analytically
/// per segment. Old segments are compressed into s-weighted
/// blocks so the sum stays O(cycles + steps in current cycle).
#[derive(Debug, Clone)]
pub struct EllerFlynnHistory {
    diffusivity: f64,
    s_lo: Vec<f64>,
    s_hi: Vec<f64>,
    f_seg: Vec<f64>,
    /// Current transformed time `s_now` [m⁴·s].
    pub s_now: f64,
}

impl EllerFlynnHistory {
    /// New empty history for gas diffusivity `d` [m²/s].
    pub fn new(diffusivity: f64) -> Self {
        Self {
            diffusivity,
            s_lo: Vec::new(),
            s_hi: Vec::new(),
            f_seg: Vec::new(),
            s_now: 0.0,
        }
    }

    /// Append one integrator step: radius goes `r → r_next` over
    /// `dt` [s] with wall-minus-far-field concentration
    /// difference `f_val` [mol/m³] held constant over the step.
    pub fn add_step(&mut self, r: f64, r_next: f64, dt: f64, f_val: f64) {
        let mut ds = 0.5 * (r.powi(4) + r_next.powi(4)) * dt;
        if ds <= 0.0 {
            ds = r.powi(4) * dt;
        }
        self.s_lo.push(self.s_now);
        self.s_now += ds;
        self.s_hi.push(self.s_now);
        self.f_seg.push(f_val);
    }

    /// Compress all but the last `keep_last_steps` segments into
    /// s-weighted blocks of ~200 (kernel is smooth for old
    /// segments).
    pub fn compress(&mut self, keep_last_steps: usize) {
        let n = self.f_seg.len();
        if n <= keep_last_steps + 200 {
            return;
        }
        let cut = n - keep_last_steps;
        let (mut lo, mut hi, mut f) = (Vec::new(), Vec::new(), Vec::new());
        let mut j = 0;
        while j < cut {
            let k = (j + 200).min(cut);
            let mut w_sum = 0.0;
            let mut fw_sum = 0.0;
            for m in j..k {
                let w = self.s_hi[m] - self.s_lo[m];
                w_sum += w;
                fw_sum += self.f_seg[m] * w;
            }
            lo.push(self.s_lo[j]);
            hi.push(self.s_hi[k - 1]);
            f.push(fw_sum / w_sum.max(1e-300));
            j = k;
        }
        lo.extend_from_slice(&self.s_lo[cut..]);
        hi.extend_from_slice(&self.s_hi[cut..]);
        f.extend_from_slice(&self.f_seg[cut..]);
        self.s_lo = lo;
        self.s_hi = hi;
        self.f_seg = f;
    }

    /// The convolution integral `J` [mol/m³ · m²·s^½] at `s_now`.
    pub fn integral_j(&self) -> f64 {
        let mut j = 0.0;
        for m in 0..self.f_seg.len() {
            let w = 2.0
                * ((self.s_now - self.s_lo[m]).max(0.0).sqrt()
                    - (self.s_now - self.s_hi[m]).max(0.0).sqrt());
            j += self.f_seg[m] * w;
        }
        j
    }

    /// `(J_pos, J_neg)` restricted to segments starting at or
    /// after `s_from` — gross outflux/influx bookkeeping for the
    /// current cycle (diagnostic; older segments' kernel weights
    /// also evolve, so this is approximate by construction).
    pub fn integral_j_split(&self, s_from: f64) -> (f64, f64) {
        let (mut jp, mut jn) = (0.0, 0.0);
        for m in 0..self.f_seg.len() {
            if self.s_lo[m] < s_from - 1e-12 {
                continue;
            }
            let w = 2.0
                * ((self.s_now - self.s_lo[m]).max(0.0).sqrt()
                    - (self.s_now - self.s_hi[m]).max(0.0).sqrt());
            let c = self.f_seg[m] * w;
            if c > 0.0 {
                jp += c;
            } else {
                jn += c;
            }
        }
        (jp, jn)
    }

    /// Gas content [mol] given the initial content `n_g0` [mol]:
    /// `n_g = n_g0 − 4√(πD)·J` (Kreider Eq. (14)).
    pub fn gas_content(&self, n_g0: f64) -> f64 {
        n_g0 - 4.0 * (std::f64::consts::PI * self.diffusivity).sqrt() * self.integral_j()
    }
}
