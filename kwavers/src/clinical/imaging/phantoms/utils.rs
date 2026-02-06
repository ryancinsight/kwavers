use super::types::TissueType;
use crate::clinical::imaging::chromophores::HemoglobinDatabase;
use crate::domain::medium::properties::OpticalPropertyData;

/// Compute blood optical properties from hemoglobin database
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
    let n = 1.4;

    OpticalPropertyData::new(mu_a, mu_s, g, n).unwrap()
}

/// Compute tumor optical properties
pub fn compute_tumor_properties(
    hb_db: &HemoglobinDatabase,
    wavelength_nm: f64,
    so2: f64,
) -> OpticalPropertyData {
    // Tumors have enhanced blood content (2-3x normal tissue)
    let blood_props = compute_blood_properties(hb_db, wavelength_nm, so2);

    // Scale absorption by blood volume fraction (~10% for tumors vs 2% for normal)
    let mu_a = blood_props.absorption_coefficient * 0.1 + 0.5; // Background tissue absorption

    // Tumor scattering is slightly higher due to disorganized structure
    let mu_s = 120.0;
    let g = 0.85;
    let n = 1.4;

    OpticalPropertyData::new(mu_a, mu_s, g, n).unwrap()
}

/// Get tissue optical properties by type
pub fn get_tissue_properties(tissue_type: TissueType) -> OpticalPropertyData {
    match tissue_type {
        TissueType::SkinEpidermis => OpticalPropertyData::skin_epidermis(),
        TissueType::SkinDermis => OpticalPropertyData::skin_dermis(),
        TissueType::Fat => OpticalPropertyData::fat(),
        TissueType::Muscle => OpticalPropertyData::muscle(),
        TissueType::Liver => OpticalPropertyData::liver(),
        TissueType::Brain => OpticalPropertyData::brain_gray_matter(),
        TissueType::Bone => OpticalPropertyData::bone_cortical(),
        TissueType::Custom(_) => OpticalPropertyData::soft_tissue(),
    }
}
