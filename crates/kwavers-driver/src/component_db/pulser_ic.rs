//! The [`PulserIc`] datasheet record, the [`StockStatus`] availability enum, and the per-IC derived
//! property accessors (pin capacitance, supply/signal pin counts, decoupling, package and board
//! area). All fields are `&'static str` / scalar / `&'static [f64]`, so `PulserIc` is fully
//! const-constructible — the catalog table is a compile-time `static`.

/// Ultrasound pulser IC parametric model.
#[derive(Debug, Clone)]
pub struct PulserIc {
    /// Manufacturer part number.
    pub part_number: &'static str,
    /// Manufacturer name.
    pub vendor: &'static str,
    /// Number of independent output channels.
    pub channels: usize,
    /// Maximum per-channel voltage swing (V).
    pub v_max_v: f64,
    /// Maximum per-channel current (A).
    pub i_max_a: f64,
    /// Output on-resistance per channel, charge path (Ω).
    pub r_on_ohm: f64,
    /// Output on-resistance per channel, discharge path (Ω).
    pub r_on_disch_ohm: f64,
    /// Number of output voltage levels (2, 3, or 5).
    pub n_levels: usize,
    /// Integrated T/R switch for receive mode.
    pub tr_switch: bool,
    /// Integrated beamforming delay memory (channels × words).
    pub beamforming_mem: Option<usize>,
    /// Maximum switching frequency (MHz).
    pub f_max_mhz: f64,
    /// Gate charge per channel (nC).
    pub q_g_nc: f64,
    /// Supply voltage for gate drive (V).
    pub v_gate_v: f64,
    /// Package name.
    pub package: &'static str,
    /// Package body size (mm × mm).
    pub package_size_mm: (f64, f64),
    /// Package pitch (mm).
    pub pitch_mm: f64,
    /// Thermal resistance junction-to-ambient (K/W) — 0 if unknown.
    pub r_th_ja_k_per_w: f64,
    /// Typical cost per channel at quantity 1k ($).
    pub cost_per_ch_usd: f64,
    /// DigiKey stock status.
    pub stock_status: StockStatus,
    /// Typical supply voltages (V).
    pub supplies: &'static [f64],
    /// Supply current per channel at max frequency (mA).
    pub i_supply_per_ch_ma: f64,
    /// Whether charge-recycling is supported between levels.
    pub charge_recycling: bool,
    /// Whether integrated pattern memory exists.
    pub pattern_memory: bool,
}

/// DigiKey stock availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StockStatus {
    /// In stock, active part.
    Active,
    /// Evaluation sample or last-time buy.
    Limited,
    /// Not recommended for new designs.
    Nrfnd,
}

/// Per-channel output pin capacitance including ESD and PCB parasitics (pF).
/// Used for load-aware routing and loss computation.
#[must_use]
pub fn output_pin_capacitance_pf(part: &PulserIc) -> f64 {
    match part.part_number {
        "HV7355K6-G" => 15.0,
        "STHVUP32" => 8.0,
        "MAX14815" => 6.0,
        "STHV748S" => 10.0,
        "MD1715" => 20.0,
        "HV7360" => 5.0,
        _ => 12.0,
    }
}

/// Number of supply pins per channel — drives PDN complexity.
#[must_use]
pub fn supply_pins_per_ch(part: &PulserIc) -> usize {
    part.supplies.len() * 2
}

/// Recommended decoupling capacitance per channel (µF).
#[must_use]
pub fn decoupling_per_ch_uf(part: &PulserIc) -> f64 {
    match part.part_number {
        "HV7355K6-G" => 1.0,
        "STHVUP32" => 0.47,
        "MAX14815" => 0.33,
        "STHV748S" => 0.47,
        "MD1715" => 2.2,
        "HV7360" => 0.22,
        _ => 1.0,
    }
}

/// Per-channel pin count for high-level topology / fanout estimation.
#[must_use]
pub fn signal_pins_per_ch(part: &PulserIc) -> usize {
    let base = if part.beamforming_mem.is_some() { 4 } else { 3 };
    base + if part.tr_switch { 1 } else { 0 } + part.supplies.len()
}

/// Area (mm²) of one instance in the given package.
#[must_use]
pub fn pkg_area_mm2(part: &PulserIc) -> f64 {
    part.package_size_mm.0 * part.package_size_mm.1
}

/// Estimate board area (mm²) needed for `n` channels of the given pulser IC,
/// including decoupling capacitors, bypass caps, and routing fanout.
///
/// Conservative multipliers by channel density:
/// - ≥32 channels: 1.5× — BGA integrates decoupling internally; per-channel routing overhead
///   is low because many signal balls share the same power domain.
/// - ≥8 channels: 3× — QFN/MLF parts need external decoupling per supply.
/// - ≥4 channels: 5× — fewer channels per IC means more support components proportionally.
/// - 1–3 channels: 8× — single/dual-channel parts have many external components per channel.
#[must_use]
pub fn board_area_per_n_channels_mm2(part: &PulserIc, n: usize) -> f64 {
    let multiplier = if part.channels >= 32 {
        1.5
    } else if part.channels >= 8 {
        3.0
    } else if part.channels >= 4 {
        5.0
    } else {
        8.0
    };
    let ics = n.div_ceil(part.channels);
    pkg_area_mm2(part) * multiplier * ics as f64
}
