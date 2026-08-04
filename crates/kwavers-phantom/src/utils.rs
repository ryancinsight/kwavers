use super::types::PhantomTissueType;
use hyperion::coefficient::hemoglobin_absorption;
use kwavers_core::constants::optical::REFRACTIVE_INDEX_SOFT_TISSUE;
use kwavers_medium::properties::OpticalPropertyData;

/// Compute blood optical properties from hemoglobin database
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[must_use]
pub fn compute_blood_properties(wavelength_nm: f64, so2: f64) -> OpticalPropertyData {
    // Whole-blood hemoglobin: ~150 g/L, about 2.3 mM as tetramer.
    let c_total = 0.0023; // mol/L
    let c_hbo2 = c_total * so2;
    let c_hb = c_total * (1.0 - so2);

    // Hyperion owns the spectra and the Beer-Lambert conversion, so the
    // extinction lookup and the cm-to-m factor no longer live here.
    let mu_a = hemoglobin_absorption::<f64>(wavelength_nm, c_hbo2, c_hb)
        .expect("invariant: caller wavelength lies in the tabulated range")
        .in_unit::<aequitas::systems::si::units::PerMeter>();

    // Blood scattering properties (weakly wavelength-dependent)
    let mu_s = 200.0;
    let g = 0.95;
    let n = REFRACTIVE_INDEX_SOFT_TISSUE;

    OpticalPropertyData::new(mu_a, mu_s, g, n).unwrap()
}

/// Compute tumor optical properties
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[must_use]
pub fn compute_tumor_properties(wavelength_nm: f64, so2: f64) -> OpticalPropertyData {
    // Tumors have enhanced blood content (2-3x normal tissue)
    let blood_props = compute_blood_properties(wavelength_nm, so2);

    // Scale absorption by blood volume fraction (~10% for tumors vs 2% for normal)
    let mu_a = blood_props.absorption_coefficient().mul_add(0.1, 0.5); // Background tissue absorption

    // Tumor scattering is slightly higher due to disorganized structure
    let mu_s = 120.0;
    let g = 0.85;
    let n = REFRACTIVE_INDEX_SOFT_TISSUE;

    OpticalPropertyData::new(mu_a, mu_s, g, n).unwrap()
}

/// Get tissue optical properties by type
#[must_use]
pub fn get_tissue_properties(tissue_type: PhantomTissueType) -> OpticalPropertyData {
    match tissue_type {
        PhantomTissueType::SkinEpidermis => OpticalPropertyData::skin_epidermis(),
        PhantomTissueType::SkinDermis => OpticalPropertyData::skin_dermis(),
        PhantomTissueType::Fat => OpticalPropertyData::fat(),
        PhantomTissueType::Muscle => OpticalPropertyData::muscle(),
        PhantomTissueType::Liver => OpticalPropertyData::liver(),
        PhantomTissueType::Brain => OpticalPropertyData::brain_gray_matter(),
        PhantomTissueType::Bone => OpticalPropertyData::bone_cortical(),
        PhantomTissueType::Custom(_) => OpticalPropertyData::soft_tissue(),
    }
}
