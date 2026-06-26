/// Relative electromigration mean-time-to-failure (Black's equation) at operating `(j, temp_k)`
/// versus a reference `(j_ref, t_ref_k)`: `MTTF ∝ J^-n · exp(Ea/(k·T))`. Returns the ratio
/// `MTTF/MTTF_ref` (>1 is better). `n≈2`, `Ea≈0.9 eV` for copper.
#[must_use]
pub fn black_mttf_relative(j_ref: f64, t_ref_k: f64, j: f64, t_k: f64, n: f64, ea_ev: f64) -> f64 {
    if j <= 0.0 || t_k <= 0.0 || j_ref <= 0.0 || t_ref_k <= 0.0 {
        return f64::INFINITY;
    }
    let kb = 8.617_333e-5; // eV/K
    (j_ref / j).powf(n) * ((ea_ev / kb) * (1.0 / t_k - 1.0 / t_ref_k)).exp()
}
