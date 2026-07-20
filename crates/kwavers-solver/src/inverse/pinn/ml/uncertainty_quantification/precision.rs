//! Precision boundary for the PINN model's legacy `f64` presentation.

pub(super) fn restore_model_precision(value: f64) -> f32 {
    let restored = value as f32;
    debug_assert_eq!(
        f64::from(restored).to_bits(),
        value.to_bits(),
        "invariant: PinnWave2D widens backend-native f32 outputs without arithmetic"
    );
    restored
}

#[cfg(test)]
mod tests {
    use super::restore_model_precision;

    #[test]
    fn widened_model_values_restore_bitwise() {
        for expected in [0.0_f32, -0.0, 1.0, f32::MIN_POSITIVE, f32::MAX] {
            assert_eq!(
                restore_model_precision(f64::from(expected)).to_bits(),
                expected.to_bits()
            );
        }
    }
}
