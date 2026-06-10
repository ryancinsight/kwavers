//! CMUT-vs-PMUT figure of merit for intravascular ultrasound (IVUS).
//!
//! IVUS images coronary plaque from a catheter at 20–60 MHz. The dominant design
//! drivers are: **axial resolution** (∝ fractional bandwidth), **thermal safety**
//! inside the vessel (∝ 1/self-heating), **drive-voltage feasibility** for catheter
//! electronics, and **monolithic CMOS integration** for the tiny tip. This module
//! scores both technologies on those weighted criteria so the verdict is computed,
//! not asserted.
//!
//! The weighting reflects IVUS priorities (Stanford/Khuri-Yakub CMUT-on-CMOS and
//! the PMUT IVUS literature): bandwidth 0.40, thermal 0.30, drive 0.15,
//! integration 0.15.

use super::cmut::CmutCell;
use super::pmut::PmutCell;

/// Which technology a comparison recommends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutKind {
    /// Capacitive (CMUT).
    Cmut,
    /// Piezoelectric (PMUT).
    Pmut,
}

/// IVUS weighting of the four design criteria (sums to 1).
#[derive(Debug, Clone, Copy)]
pub struct IvusWeights {
    /// Axial resolution / fractional bandwidth.
    pub bandwidth: f64,
    /// Thermal safety (inverse self-heating).
    pub thermal: f64,
    /// Drive-voltage feasibility.
    pub drive: f64,
    /// Monolithic CMOS integration.
    pub integration: f64,
}

impl Default for IvusWeights {
    fn default() -> Self {
        Self {
            bandwidth: 0.40,
            thermal: 0.30,
            drive: 0.15,
            integration: 0.15,
        }
    }
}

/// Computed IVUS comparison between a CMUT and a PMUT design.
#[derive(Debug, Clone, Copy)]
pub struct IvusVerdict {
    /// CMUT fractional bandwidth.
    pub cmut_fbw: f64,
    /// PMUT fractional bandwidth.
    pub pmut_fbw: f64,
    /// CMUT self-heating power \[W].
    pub cmut_heating: f64,
    /// PMUT self-heating power \[W].
    pub pmut_heating: f64,
    /// CMUT drive voltage (≈0.8·collapse bias) \[V].
    pub cmut_drive_voltage: f64,
    /// PMUT drive voltage \[V].
    pub pmut_drive_voltage: f64,
    /// Weighted CMUT figure of merit ∈ [0, 1].
    pub cmut_fom: f64,
    /// Weighted PMUT figure of merit ∈ [0, 1].
    pub pmut_fom: f64,
    /// Recommended technology for the given IVUS weighting.
    pub recommended: MutKind,
}

/// Score CMUT vs PMUT for IVUS at a given fluid (blood) density and PMUT AC drive.
///
/// CMUT drive is taken as `0.8·V_collapse` (typical pre-collapse bias);
/// integration is categorical (CMUT 1.0 monolithic CMOS, PMUT 0.7). Each metric
/// is normalised to the better of the two before weighting.
#[must_use]
pub fn evaluate_ivus(
    cmut: &CmutCell,
    pmut: &PmutCell,
    fluid_density: f64,
    pmut_drive_voltage: f64,
    weights: IvusWeights,
) -> IvusVerdict {
    let cmut_fbw = cmut.fractional_bandwidth(fluid_density);
    let pmut_fbw = pmut.fractional_bandwidth(fluid_density);

    let cmut_drive = 0.8 * cmut.collapse_voltage();
    let cmut_heating =
        cmut.self_heating_power(0.1 * cmut_drive, cmut.immersion_resonance(fluid_density));
    let pmut_heating =
        pmut.self_heating_power(pmut_drive_voltage, pmut.immersion_resonance(fluid_density));

    // higher-is-better normalisation against the pair maximum
    let nb = |x: f64, y: f64| if x.max(y) > 0.0 { x / x.max(y) } else { 0.0 };
    // lower-is-better → invert (best gets 1.0)
    let nl = |x: f64, y: f64| {
        let m = x.min(y);
        if x > 0.0 {
            m / x
        } else {
            1.0
        }
    };

    let cmut_integration = 1.0; // monolithic CMUT-on-CMOS
    let pmut_integration = 0.7; // PMUT integrable but PZT needs a dedicated process

    let cmut_fom = weights.bandwidth * nb(cmut_fbw, pmut_fbw)
        + weights.thermal * nl(cmut_heating, pmut_heating)
        + weights.drive * nl(cmut_drive, pmut_drive_voltage)
        + weights.integration * nb(cmut_integration, pmut_integration);
    let pmut_fom = weights.bandwidth * nb(pmut_fbw, cmut_fbw)
        + weights.thermal * nl(pmut_heating, cmut_heating)
        + weights.drive * nl(pmut_drive_voltage, cmut_drive)
        + weights.integration * nb(pmut_integration, cmut_integration);

    IvusVerdict {
        cmut_fbw,
        pmut_fbw,
        cmut_heating,
        pmut_heating,
        cmut_drive_voltage: cmut_drive,
        pmut_drive_voltage,
        cmut_fom,
        pmut_fom,
        recommended: if cmut_fom >= pmut_fom {
            MutKind::Cmut
        } else {
            MutKind::Pmut
        },
    }
}

/// Computed therapy comparison (output-pressure governed) between a CMUT and PMUT.
#[derive(Debug, Clone, Copy)]
pub struct TherapyVerdict {
    /// CMUT peak output pressure after flex + substrate derating \[Pa].
    pub cmut_output_pa: f64,
    /// PMUT peak output pressure after substrate derating \[Pa].
    pub pmut_output_pa: f64,
    /// CMUT flex-gap derating factor applied (∈ (0, 1]).
    pub cmut_flex_derating: f64,
    /// CMUT self-heating \[W].
    pub cmut_heating: f64,
    /// PMUT self-heating \[W].
    pub pmut_heating: f64,
    /// Recommended technology for high-pressure therapy (governed by output).
    pub recommended: MutKind,
}

/// Score CMUT vs PMUT for **therapeutic** ultrasound (high acoustic pressure, low
/// MHz). Therapy is governed by deliverable output pressure, so the verdict is
/// taken on peak output. The CMUT output is gap-limited (`cmut_swing_fraction` of
/// the gap, ~1/3 conventional) and further derated by curvature (`curvature`,
/// `1/m`); the PMUT output scales with `pmut_drive_voltage`. A shared
/// `substrate_output_factor ∈ (0,1]` models the flexible-carrier recoil loss.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_therapy(
    cmut: &CmutCell,
    pmut: &PmutCell,
    fluid_density: f64,
    fluid_sound_speed: f64,
    cmut_swing_fraction: f64,
    pmut_drive_voltage: f64,
    curvature: f64,
    substrate_output_factor: f64,
) -> TherapyVerdict {
    let cmut_flex = cmut.flex_gap_derating(curvature);
    let cmut_output =
        cmut.max_output_pressure(fluid_density, fluid_sound_speed, cmut_swing_fraction)
            * cmut_flex
            * substrate_output_factor;
    let pmut_output =
        pmut.max_output_pressure(pmut_drive_voltage, fluid_density, fluid_sound_speed)
            * substrate_output_factor;

    let cmut_f = cmut.immersion_resonance(fluid_density);
    let pmut_f = pmut.immersion_resonance(fluid_density);
    let cmut_heating = cmut.self_heating_power(0.1 * cmut.collapse_voltage(), cmut_f);
    let pmut_heating = pmut.self_heating_power(pmut_drive_voltage, pmut_f);

    TherapyVerdict {
        cmut_output_pa: cmut_output,
        pmut_output_pa: pmut_output,
        cmut_flex_derating: cmut_flex,
        cmut_heating,
        pmut_heating,
        recommended: if pmut_output >= cmut_output {
            MutKind::Pmut
        } else {
            MutKind::Cmut
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mems::pmut::PiezoFilm;

    // Blood density.
    const BLOOD_RHO: f64 = 1060.0;
    // Representative IVUS CMUT: a=14 µm, h=0.4 µm membrane, 0.25 µm gap (→ tens-of-V collapse).
    fn ivus_cmut() -> CmutCell {
        CmutCell::silicon(14e-6, 0.4e-6, 0.25e-6).unwrap()
    }

    #[test]
    fn cmut_wins_ivus_on_bandwidth_and_thermal() {
        let cmut = ivus_cmut();
        let pmut = PmutCell::new(20e-6, 1e-6, 2e-6, PiezoFilm::Pzt).unwrap();
        let rho = BLOOD_RHO;
        let v = evaluate_ivus(&cmut, &pmut, rho, 5.0, IvusWeights::default());

        // CMUT's light membrane → wider fractional bandwidth (better axial resolution)
        assert!(
            v.cmut_fbw > v.pmut_fbw,
            "CMUT FBW {} vs PMUT {}",
            v.cmut_fbw,
            v.pmut_fbw
        );
        // CMUT (tanδ≈1e-3) self-heats less than PZT PMUT (tanδ≈0.02)
        assert!(
            v.cmut_heating < v.pmut_heating,
            "heating {} vs {}",
            v.cmut_heating,
            v.pmut_heating
        );
        // for the IVUS weighting, CMUT is recommended
        assert_eq!(v.recommended, MutKind::Cmut);
        assert!(v.cmut_fom > v.pmut_fom);
    }

    #[test]
    fn pmut_keeps_its_drive_voltage_advantage() {
        let cmut = ivus_cmut();
        let pmut = PmutCell::new(20e-6, 1e-6, 2e-6, PiezoFilm::Aln).unwrap();
        let v = evaluate_ivus(&cmut, &pmut, BLOOD_RHO, 5.0, IvusWeights::default());
        // PMUT operates well below the CMUT collapse bias — the documented trade-off
        assert!(
            v.pmut_drive_voltage < v.cmut_drive_voltage,
            "PMUT {} should drive below CMUT bias {}",
            v.pmut_drive_voltage,
            v.cmut_drive_voltage
        );
    }

    #[test]
    fn pmut_wins_high_pressure_therapy_and_flexing_hurts_cmut() {
        // therapy-scale (~2–5 MHz) designs in water; PMUT driven hard (PZT)
        let cmut = CmutCell::silicon(60e-6, 2.0e-6, 0.2e-6).unwrap();
        let pmut = PmutCell::new(60e-6, 2e-6, 4e-6, PiezoFilm::Pzt).unwrap();
        let (rho, c) = (1000.0, 1500.0);

        // flat, rigid backing
        let flat = evaluate_therapy(&cmut, &pmut, rho, c, 1.0 / 3.0, 40.0, 0.0, 1.0);
        // PMUT delivers more pressure for therapy (CMUT is gap-limited)
        assert!(
            flat.pmut_output_pa > flat.cmut_output_pa,
            "PMUT {} > CMUT {}",
            flat.pmut_output_pa,
            flat.cmut_output_pa
        );
        assert_eq!(flat.recommended, MutKind::Pmut);

        // wrap onto a flexible catheter (κ = 1/2 mm) → CMUT output falls further
        let wrapped = evaluate_therapy(&cmut, &pmut, rho, c, 1.0 / 3.0, 40.0, 1.0 / 2.0e-3, 1.0);
        assert!(
            wrapped.cmut_flex_derating < 1.0,
            "flexing must derate the CMUT"
        );
        assert!(
            wrapped.cmut_output_pa < flat.cmut_output_pa,
            "flexed CMUT output {} < flat {}",
            wrapped.cmut_output_pa,
            flat.cmut_output_pa
        );
        assert_eq!(wrapped.recommended, MutKind::Pmut);
    }

    #[test]
    fn drive_weighted_priority_can_favour_pmut() {
        // a drive-voltage-dominated weighting flips the verdict toward PMUT
        let cmut = ivus_cmut();
        let pmut = PmutCell::new(20e-6, 1e-6, 2e-6, PiezoFilm::Aln).unwrap();
        let w = IvusWeights {
            bandwidth: 0.1,
            thermal: 0.1,
            drive: 0.7,
            integration: 0.1,
        };
        let v = evaluate_ivus(&cmut, &pmut, BLOOD_RHO, 3.0, w);
        assert_eq!(v.recommended, MutKind::Pmut);
    }
}
