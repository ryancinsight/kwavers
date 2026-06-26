//! PDN parallel-bank impedance kernel.
//!
//! Three free functions that quantify the impedance of a parallel capacitor bank (and the
//! anti-resonance peak it produces).
//!
//! * [`self_resonant_freq_hz`] — single cap's SRF, `1/(2π√(LC))`.
//! * [`pdn_impedance_at_freq`] — parallel bank's |Z(f)| over an `(C_f, ESR_ohm, ESL_h)` slice.
//! * [`anti_resonance_hz`] — bulk↔local LC anti-resonance peak frequency.

/// Self-resonant frequency (Hz) of a capacitor with parasitic series inductance: `1/(2π√(LC))`.
/// A decoupling cap only lowers impedance below its SRF.
#[must_use]
pub fn self_resonant_freq_hz(esl_h: f64, c_f: f64) -> f64 {
    if esl_h <= 0.0 || c_f <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (2.0 * std::f64::consts::PI * (esl_h * c_f).sqrt())
}

/// PDN impedance magnitude |Z(f)| (Ω) of a parallel capacitor bank at frequency `f_hz`.
///
/// Each capacitor is modelled as series `ESR + ESL + 1/(jωC)`. The bank is the parallel
/// combination of all caps. This is the standard single-port PDN model used for target-impedance
/// analysis: the designer specifies a target impedance `Z_target = V_ripple/I_transient` and
/// verifies that `|Z(f)|` stays below it across the supply bandwidth.
///
/// `caps` — slice of `(C_f, ESR_ohm, ESL_h)` tuples, one per capacitor in the bank.
/// Returns `+∞` if the bank is empty.
///
/// # Example
///
/// A 100 nF cap with 50 mΩ ESR and 0.5 nH ESL self-resonates near 22 MHz; below SRF the cap
/// looks capacitive (|Z| falls), above it inductive (|Z| rises). At SRF |Z| = ESR.
#[must_use]
pub fn pdn_impedance_at_freq(caps: &[(f64, f64, f64)], f_hz: f64) -> f64 {
    if caps.is_empty() || f_hz < 0.0 {
        return f64::INFINITY;
    }
    let omega = 2.0 * std::f64::consts::PI * f_hz;
    // Each cap branch: Z_branch = ESR + j·(omega·ESL - 1/(omega·C)).
    // Admittance Y = 1/Z_branch (complex). Sum admittances, invert.
    let mut y_re = 0.0f64;
    let mut y_im = 0.0f64;
    for &(c_f, esr, esl) in caps {
        if c_f <= 0.0 {
            continue;
        }
        let x = if omega > 0.0 {
            omega * esl - 1.0 / (omega * c_f)
        } else {
            f64::NEG_INFINITY
        };
        // Z = esr + j·x; Y = Z* / |Z|² = (esr - j·x) / (esr² + x²)
        let z_mag2 = esr * esr + x * x;
        if z_mag2 <= 0.0 {
            continue;
        }
        y_re += esr / z_mag2;
        y_im += -x / z_mag2;
    }
    // |Y_total|, then |Z_total| = 1/|Y_total|.
    let y_mag = (y_re * y_re + y_im * y_im).sqrt();
    if y_mag <= 0.0 {
        f64::INFINITY
    } else {
        1.0 / y_mag
    }
}

/// Frequency (Hz) of the anti-resonant impedance peak between two capacitor stages.
///
/// When a low-frequency bulk capacitor (`C_bulk` with high ESL `L_bulk`) is paralleled with a
/// high-frequency local capacitor (`C_local` with low ESL `L_local`), the combination has a
/// parallel LC tank at the frequency where the bulk cap's inductive branch resonates with the
/// local cap's capacitive branch:
///
/// `f_ar ≈ 1 / (2π · √(L_bulk · C_local))` (dominant approximation for `L_local ≪ L_bulk`).
///
/// At this frequency |Z(f_ar)| peaks (can exceed the target impedance), so strategic mid-band
/// caps or resistive damping are needed. Returns `+∞` if either value is non-positive.
#[must_use]
pub fn anti_resonance_hz(l_bulk_h: f64, c_local_f: f64) -> f64 {
    if l_bulk_h <= 0.0 || c_local_f <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (2.0 * std::f64::consts::PI * (l_bulk_h * c_local_f).sqrt())
}
