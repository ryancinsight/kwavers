//! Source-encoded simultaneous-shot FWI (Krebs et al. 2009).
//!
//! # Specification
//!
//! Multi-source FWI sums per-shot adjoint gradients, costing one
//! forward+adjoint pair per shot per iteration. Source encoding collapses the
//! shot gather into a single *supershot*: each shot `i` is weighted by a code
//! `cᵢ`, the coded source signals are superposed into one excitation, and the
//! recorded data are combined as `d_enc = Σᵢ cᵢ dᵢ`. A single forward+adjoint
//! pair on the supershot then yields a stochastic estimate of the full
//! multi-shot gradient.
//!
//! ## Theorem (unbiasedness)
//!
//! Because the wave equation is linear in the source, the encoded forward field
//! is `u_enc = Σᵢ cᵢ uᵢ` and the encoded adjoint field is `λ_enc = Σⱼ cⱼ λⱼ`.
//! The zero-lag imaging condition is bilinear, so the encoded gradient is
//!
//! ```text
//! g(c) = Σᵢ Σⱼ cᵢ cⱼ Gᵢⱼ,     Gᵢⱼ = ⟨üᵢ, λⱼ⟩  (per-voxel correlation).
//! ```
//!
//! The diagonal `Σᵢ Gᵢᵢ` is exactly the summed per-shot gradient; the
//! off-diagonal terms `i ≠ j` are *crosstalk*. For random Rademacher codes
//! `E[cᵢcⱼ] = δᵢⱼ`, so `E[g(c)] = Σᵢ Gᵢᵢ` — `g(c)` is an unbiased estimator of
//! the full gradient (Krebs et al. 2009).
//!
//! ## Exact cancellation with orthogonal codes
//!
//! For a complete orthogonal code set (the rows `c⁽ᵏ⁾` of an `N×N` Hadamard
//! matrix), `Σₖ cₖᵢ cₖⱼ = N δᵢⱼ`, hence
//!
//! ```text
//! (1/N) Σₖ g(c⁽ᵏ⁾) = Σᵢ Gᵢᵢ
//! ```
//!
//! is the summed per-shot gradient *exactly* — the crosstalk cancels with no
//! statistical residual. This is the differential contract verified by the test
//! suite.
//!
//! # References
//! - Krebs, J. et al. (2009). *Fast full-wavefield seismic inversion using
//!   encoded sources.* Geophysics 74(6), WCC177–WCC188.
//! - Ben-Hadj-Ali, H., Operto, S., Virieux, J. (2011). *An efficient
//!   frequency-domain full waveform inversion method using simultaneous encoded
//!   sources.* Geophysics 76(4), R109–R124.
//! - Krebs (2009) extends to medical USCT via cross-correlation-adjusted and
//!   vortex-encoded variants (2024–2025 literature).

use super::{geometry::FwiGeometry, FwiProcessor};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_source::GridSource;
use leto::{Array2 as LetoArray2, Array3 as LetoArray3};
use leto::{
    Array2,
    Array3,
};

/// Generate the `n × n` Sylvester–Hadamard code matrix (`n` a power of two).
///
/// Returns `n` orthogonal Rademacher (±1) code vectors. The rows satisfy
/// `Σₖ cₖᵢ cₖⱼ = n δᵢⱼ`, which drives the exact crosstalk cancellation of the
/// averaged encoded gradient.
///
/// # Errors
/// Returns [`KwaversError::Validation`] if `n` is zero or not a power of two.
pub fn hadamard_codes(n: usize) -> KwaversResult<Vec<Vec<f64>>> {
    if n == 0 || !n.is_power_of_two() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!("Hadamard order must be a positive power of two; got {n}"),
            },
        ));
    }
    // Sylvester construction: H_1 = [1]; H_{2m} = [[H_m, H_m], [H_m, -H_m]].
    let mut h = vec![vec![1.0_f64]];
    while h.len() < n {
        let m = h.len();
        let mut next = vec![vec![0.0_f64; 2 * m]; 2 * m];
        for i in 0..m {
            for j in 0..m {
                let v = h[i][j];
                next[i][j] = v;
                next[i][j + m] = v;
                next[i + m][j] = v;
                next[i + m][j + m] = -v;
            }
        }
        h = next;
    }
    Ok(h)
}

/// Combine a shot gather into a single coded supershot.
///
/// All shots must share the same receiver layout and the same source mask
/// (the common simultaneous-source setting: one physical array, different
/// transmit excitations per shot). The supershot source signal is
/// `Σᵢ cᵢ sᵢ` and the encoded data is `Σᵢ cᵢ dᵢ`, both formed row-wise.
///
/// # Errors
/// Returns [`KwaversError::Validation`] if the gather is empty, the code count
/// differs from the shot count, a shot lacks a pressure source signal/mask, or
/// the shots do not share an identical receiver mask, source mask, source-signal
/// shape, and data shape.
pub fn encode_shots(
    shots: &[(FwiGeometry, Array2<f64>)],
    codes: &[f64],
) -> KwaversResult<(FwiGeometry, Array2<f64>)> {
    if shots.is_empty() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: "encode_shots requires at least one shot".to_owned(),
            },
        ));
    }
    if codes.len() != shots.len() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "encode_shots code count {} must equal shot count {}",
                    codes.len(),
                    shots.len()
                ),
            },
        ));
    }

    let (reference_geometry, reference_data) = &shots[0];
    let reference_signal = require_signal(&reference_geometry.source)?;
    let reference_mask = require_mask(&reference_geometry.source)?;

    for (index, (geometry, data)) in shots.iter().enumerate() {
        if geometry.sensor_mask != reference_geometry.sensor_mask {
            return Err(mismatch(index, "receiver mask"));
        }
        if require_mask(&geometry.source)? != reference_mask {
            return Err(mismatch(index, "source mask"));
        }
        if require_signal(&geometry.source)?.shape() != reference_signal.shape() {
            return Err(mismatch(index, "source-signal shape"));
        }
        if data.shape() != reference_data.shape() {
            return Err(mismatch(index, "data shape"));
        }
    }

    let [signal_rows, signal_cols] = reference_signal.shape();
    let mut encoded_signal = LetoArray2::zeros([signal_rows, signal_cols]);
    let mut encoded_data = Array2::zeros(reference_data.shape());
    for (code, (geometry, data)) in codes.iter().zip(shots.iter()) {
        let signal = require_signal(&geometry.source)?;
        for row in 0..signal_rows {
            for col in 0..signal_cols {
                encoded_signal[[row, col]] += code * signal[[row, col]];
            }
        }
        let [data_rows, data_cols] = data.shape();
        for row in 0..data_rows {
            for col in 0..data_cols {
                encoded_data[[row, col]] += code * data[[row, col]];
            }
        }
    }

    let source = GridSource {
        p0: reference_geometry.source.p0.clone(),
        u0: reference_geometry.source.u0.clone(),
        p_mask: Some(reference_mask.clone()),
        p_signal: Some(encoded_signal),
        p_mode: reference_geometry.source.p_mode,
        u_mask: reference_geometry.source.u_mask.clone(),
        u_signal: reference_geometry.source.u_signal.clone(),
        u_mode: reference_geometry.source.u_mode,
    };

    Ok((
        FwiGeometry::new(source, reference_geometry.sensor_mask.clone()),
        encoded_data,
    ))
}

fn require_signal(source: &GridSource) -> KwaversResult<&LetoArray2<f64>> {
    source.p_signal.as_ref().ok_or_else(|| {
        KwaversError::Validation(ValidationError::ConstraintViolation {
            message: "encode_shots requires every shot to carry a pressure source signal"
                .to_owned(),
        })
    })
}

fn require_mask(source: &GridSource) -> KwaversResult<&LetoArray3<f64>> {
    source.p_mask.as_ref().ok_or_else(|| {
        KwaversError::Validation(ValidationError::ConstraintViolation {
            message: "encode_shots requires every shot to carry a pressure source mask".to_owned(),
        })
    })
}

fn mismatch(index: usize, field: &str) -> KwaversError {
    KwaversError::Validation(ValidationError::ConstraintViolation {
        message: format!("encode_shots shot {index} {field} differs from shot 0"),
    })
}

impl FwiProcessor {
    /// Source-encoded simultaneous-shot FWI.
    ///
    /// Each entry of `code_schedule` is a Rademacher code vector (length = shot
    /// count) defining one stochastic iteration: the gather is encoded into a
    /// supershot and a single gradient-descent step is taken. Supplying the rows
    /// of a Hadamard matrix produces a deterministic, crosstalk-free sweep; a
    /// random ±1 schedule produces the classical stochastic encoded FWI.
    ///
    /// # Errors
    /// - [`KwaversError::Validation`] if the gather or schedule is empty, a code
    ///   vector length differs from the shot count, or a shot fails geometry
    ///   validation.
    /// - Propagates any [`KwaversError`] from encoding or the descent step.
    pub fn invert_encoded(
        &self,
        shots: &[(FwiGeometry, Array2<f64>)],
        initial_model: &Array3<f64>,
        grid: &Grid,
        code_schedule: &[Vec<f64>],
    ) -> KwaversResult<Array3<f64>> {
        if shots.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "invert_encoded requires at least one shot".to_owned(),
                },
            ));
        }
        if code_schedule.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "invert_encoded requires at least one code vector".to_owned(),
                },
            ));
        }
        for codes in code_schedule {
            if codes.len() != shots.len() {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "invert_encoded code vector length {} must equal shot count {}",
                            codes.len(),
                            shots.len()
                        ),
                    },
                ));
            }
        }
        for (geometry, _) in shots {
            geometry.validate(grid, self.parameters.nt)?;
        }

        let mut model = initial_model.clone();
        self.apply_model_constraints(&mut model);

        for (iteration, codes) in code_schedule.iter().enumerate() {
            let (geometry, encoded_data) = encode_shots(shots, codes)?;
            let (objective, updated, step_size) =
                self.descent_update(&model, &encoded_data, &geometry, grid, 1.0)?;
            log::info!("FWI encoded iter {iteration}: J={objective:.6e} step={step_size:.6e}");
            if step_size == 0.0 {
                log::info!(
                    "FWI encoded iter {iteration}: no descent for this code; retaining model"
                );
                continue;
            }
            model = updated;
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::seismic::parameters::FwiParameters;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_source::SourceMode;
    use leto::Array2;

    #[test]
    fn hadamard_codes_are_orthogonal() {
        for &n in &[1usize, 2, 4, 8] {
            let codes = hadamard_codes(n).expect("power of two");
            assert_eq!(codes.len(), n);
            for (i, row) in codes.iter().enumerate() {
                assert_eq!((row.len()), n);
                assert!(row.iter().all(|&v| v == 1.0 || v == -1.0));
                // Column-orthogonality: Σ_k c_ki c_kj = n δ_ij.
                for j in 0..n {
                    let dot: f64 = (0..n).map(|k| codes[k][i] * codes[k][j]).sum();
                    let expected = if i == j { n as f64 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-12,
                        "n={n} i={i} j={j} dot={dot}"
                    );
                }
            }
        }
    }

    #[test]
    fn hadamard_codes_reject_non_power_of_two() {
        assert!(hadamard_codes(0).is_err());
        assert!(hadamard_codes(3).is_err());
        assert!(hadamard_codes(6).is_err());
    }

    /// Build a single-element-source / 4-receiver geometry sharing layout across
    /// shots; only the source signal differs per shot.
    fn shot(signal_amp: f64, signal_phase: f64, observed_level: f64) -> (FwiGeometry, Array2<f64>) {
        let dims = (8usize, 8, 8);
        let nt = 48usize;
        let mut p_mask = Array3::from_elem(dims, 0.0_f64);
        p_mask[[2, 4, 4]] = 1.0;
        let mut p_signal = Array2::zeros((1, nt));
        for t in 0..16 {
            let phase = (t as f64) * 0.4 + signal_phase;
            p_signal[[0, t]] = signal_amp * (-phase * phase * 0.05).exp() * (2.0 * phase).sin();
        }
        let source = GridSource {
            p0: None,
            u0: None,
            p_mask: Some(p_mask),
            p_signal: Some(p_signal),
            p_mode: SourceMode::Additive,
            u_mask: None,
            u_signal: None,
            u_mode: SourceMode::default(),
        };
        let mut sensor_mask = Array3::from_elem(dims, false);
        for iy in 2..6 {
            sensor_mask[[6, iy, 4]] = true;
        }
        let geometry = FwiGeometry::new(source, sensor_mask);
        let observed = Array2::from_elem([4, nt], observed_level);
        (geometry, observed)
    }

    #[test]
    fn encode_shots_superposes_signals_and_data() {
        let shots = vec![shot(1.0, 0.0, 0.3), shot(1.5, 0.7, -0.2)];
        let codes = vec![1.0, -1.0];
        let (encoded, data) = encode_shots(&shots, &codes).expect("encode");

        let s0 = shots[0].0.source.p_signal.as_ref().unwrap();
        let s1 = shots[1].0.source.p_signal.as_ref().unwrap();
        let expected_signal = s0 - s1;
        let encoded_signal = encoded.source.p_signal.as_ref().unwrap();
        assert_eq!(encoded_signal, &expected_signal);

        let expected_data = &shots[0].1 - &shots[1].1;
        assert_eq!(data, expected_data);
    }

    #[test]
    fn encode_shots_rejects_mismatched_receiver_layout() {
        let (geom_a, data_a) = shot(1.0, 0.0, 0.0);
        let dims = (8usize, 8, 8);
        // A second shot with a different receiver mask.
        let (mut geom_b, _) = shot(1.0, 0.0, 0.0);
        let mut alt_mask = Array3::from_elem(dims, false);
        alt_mask[[6, 3, 4]] = true;
        geom_b.sensor_mask = alt_mask;
        let data_b = Array2::zeros((1, 48));
        let shots = vec![(geom_a, data_a), (geom_b, data_b)];
        let err = encode_shots(&shots, &[1.0, 1.0]).expect_err("layout mismatch");
        assert!(format!("{err:?}").contains("receiver mask"));
    }

    /// Exact differential contract (the unbiasedness theorem): averaging the
    /// encoded gradient over a complete Hadamard code set reproduces the summed
    /// per-shot gradient bit-for-bit up to FDTD round-off.
    #[test]
    fn hadamard_averaged_encoded_gradient_matches_summed_shot_gradient() {
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid");
        let dims = (8usize, 8, 8);
        let nt = 48usize;
        let model = Array3::from_elem(dims, SOUND_SPEED_WATER_SIM);
        let parameters = FwiParameters {
            nt,
            dt: 1e-7,
            frequency: 5e5,
            ..FwiParameters::default()
        };
        let processor = FwiProcessor::new(parameters);

        let shots = vec![shot(1.0, 0.0, 0.3), shot(1.3, 0.9, -0.15)];

        // Reference: Σ_i g_i (raw physics gradient per shot).
        let mut reference = Array3::<f64>::zeros(dims);
        for (geometry, observed) in &shots {
            let g = raw_gradient(&processor, &model, geometry, observed, &grid);
            leto_ops::zip_mut_with(&mut reference.view_mut(), &g.view(), |r, gv| *r += *gv)
                .expect("invariant: gradient shapes match");
        }

        // Encoded: average over the 2×2 Hadamard set.
        let codes = hadamard_codes(shots.len()).expect("hadamard");
        let mut encoded_avg = Array3::<f64>::zeros(dims);
        for code in &codes {
            let (geometry, encoded_data) = encode_shots(&shots, code).expect("encode");
            let g = raw_gradient(&processor, &model, &geometry, &encoded_data, &grid);
            leto_ops::zip_mut_with(&mut encoded_avg.view_mut(), &g.view(), |a, gv| *a += *gv)
                .expect("invariant: gradient shapes match");
        }
        let ncodes = codes.len() as f64;
        for v in encoded_avg.iter_mut() {
            *v /= ncodes;
        }

        let max_ref = reference.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        assert!(max_ref > 1e-18, "reference gradient must be non-trivial");
        let mut max_abs_diff = 0.0_f64;
        for (&a, &b) in reference.iter().zip(encoded_avg.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        let rel = max_abs_diff / max_ref;
        assert!(
            rel < 1e-6,
            "Hadamard-averaged encoded gradient must equal summed shot gradient: rel diff = {rel:e}"
        );
    }

    /// Raw physics gradient for one (geometry, observed) pair — forward, L2
    /// residual, time-reversed adjoint, adjoint-state correlation.
    fn raw_gradient(
        processor: &FwiProcessor,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        observed: &Array2<f64>,
        grid: &Grid,
    ) -> Array3<f64> {
        let (synthetic, history) = processor
            .forward_model(model, geometry, grid)
            .expect("forward");
        let residual = processor
            .compute_adjoint_source(observed, &synthetic)
            .expect("residual");
        let adjoint_source = processor
            .build_adjoint_source(&residual, geometry)
            .expect("adjoint source");
        processor
            .adjoint_model(&adjoint_source, model, grid, &history, None)
            .expect("gradient")
    }
}
