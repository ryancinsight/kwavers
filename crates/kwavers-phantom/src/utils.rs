use super::types::PhantomTissueType;
use kwavers_core::constants::optical::REFRACTIVE_INDEX_SOFT_TISSUE;
use kwavers_medium::properties::OpticalPropertyData;
use kwavers_optics::chromophores::HemoglobinDatabase;

/// Compute blood optical properties from hemoglobin database
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[must_use]
pub fn compute_blood_properties(
    hb_db: &HemoglobinDatabase,
    wavelength_nm: f64,
    so2: f64,
) -> OpticalPropertyData {
    // Get extinction coefficients
    let eps_hbo2 = hb_db
        .hbo2_spectrum()
        .at_wavelength(wavelength_nm)
        .unwrap_or(0.0);
    let eps_hb = hb_db
        .hb_spectrum()
        .at_wavelength(wavelength_nm)
        .unwrap_or(0.0);

    // Blood hemoglobin concentration: ~150 g/L = 2.3 mM total
    let c_total = 0.0023; // mol/L
    let c_hbo2 = c_total * so2;
    let c_hb = c_total * (1.0 - so2);

    // Compute absorption coefficient (convert from cm⁻¹ to m⁻¹)
    let mu_a = (eps_hbo2 * c_hbo2 + eps_hb * c_hb) * 100.0 * 2.303; // ln(10) factor

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
pub fn compute_tumor_properties(
    hb_db: &HemoglobinDatabase,
    wavelength_nm: f64,
    so2: f64,
) -> OpticalPropertyData {
    // Tumors have enhanced blood content (2-3x normal tissue)
    let blood_props = compute_blood_properties(hb_db, wavelength_nm, so2);

    // Scale absorption by blood volume fraction (~10% for tumors vs 2% for normal)
    let mu_a = blood_props.absorption_coefficient.mul_add(0.1, 0.5); // Background tissue absorption

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
