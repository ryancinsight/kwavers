//! The [`DcDcModule`] HV-bias-rail converter record and its static catalog. Like the pulser table,
//! every field is const-constructible, so [`available_dcdc_modules`] returns `&'static [DcDcModule]`
//! with no per-call allocation.

use super::pulser_ic::StockStatus;

/// Power-supply module parametric model.
#[derive(Debug, Clone)]
pub struct DcDcModule {
    /// Manufacturer part number.
    pub part_number: &'static str,
    /// Manufacturer name.
    pub vendor: &'static str,
    /// Minimum input voltage (V).
    pub vin_min_v: f64,
    /// Maximum input voltage (V).
    pub vin_max_v: f64,
    /// Regulated output voltage (V).
    pub vout_v: f64,
    /// Maximum continuous output current (A).
    pub iout_max_a: f64,
    /// Input-to-output isolation rating (V).
    pub isolation_v: f64,
    /// Typical conversion efficiency as a fraction (0–1).
    pub efficiency: f64,
    /// Package body dimensions (width × height, mm).
    pub package_size_mm: (f64, f64),
    /// Unit cost at quantity 1k (USD).
    pub cost_usd: f64,
    /// DigiKey stock status.
    pub stock_status: StockStatus,
}

/// Available DC-DC converter modules for HV bias rails.
///
/// Returns a borrow of the compile-time catalog; no per-call allocation.
#[must_use]
pub fn available_dcdc_modules() -> &'static [DcDcModule] {
    DCDC_MODULES
}

/// The DC-DC HV-bias-rail converter catalog, fixed at compile time.
static DCDC_MODULES: &[DcDcModule] = &[
    DcDcModule {
        part_number: "ROE-0515S",
        vendor: "Recom Power",
        vin_min_v: 5.0,
        vin_max_v: 5.0,
        vout_v: 15.0,
        iout_max_a: 0.066,
        isolation_v: 1000.0,
        efficiency: 0.80,
        package_size_mm: (19.6, 7.1),
        cost_usd: 4.50,
        stock_status: StockStatus::Active,
    },
    DcDcModule {
        part_number: "AM1S-0512SH30",
        vendor: "aimtec",
        vin_min_v: 4.5,
        vin_max_v: 5.5,
        vout_v: 12.0,
        iout_max_a: 0.083,
        isolation_v: 3000.0,
        efficiency: 0.82,
        package_size_mm: (12.7, 7.5),
        cost_usd: 3.80,
        stock_status: StockStatus::Active,
    },
    DcDcModule {
        part_number: "AE10-15S2V5",
        vendor: "Artesyn",
        vin_min_v: 4.5,
        vin_max_v: 5.5,
        vout_v: 150.0,
        iout_max_a: 0.067,
        isolation_v: 3000.0,
        efficiency: 0.87,
        package_size_mm: (50.8, 25.4),
        cost_usd: 32.00,
        stock_status: StockStatus::Active,
    },
];
