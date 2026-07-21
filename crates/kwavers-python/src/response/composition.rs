//! Independent-insult probability conversion.

use asclepius::{response::composition::IndependentInsults, BiologicalResponse, Probability};

pub(crate) fn independent_kill_probability(
    mechanical: &[f64],
    thermal: &[f64],
) -> Result<Vec<f64>, String> {
    if mechanical.len() != thermal.len() {
        return Err(format!(
            "mechanical probability length {} does not match thermal length {}",
            mechanical.len(),
            thermal.len()
        ));
    }
    let law = IndependentInsults::<2>;
    mechanical
        .iter()
        .copied()
        .zip(thermal.iter().copied())
        .map(|(mechanical, thermal)| {
            let insults = [
                Probability::new(mechanical).map_err(|source| source.to_string())?,
                Probability::new(thermal).map_err(|source| source.to_string())?,
            ];
            law.evaluate(&insults)
                .map(Probability::get)
                .map_err(|source| source.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_composition_matches_survival_product() {
        let combined = independent_kill_probability(&[0.2], &[0.5]).expect("valid probabilities");
        assert_eq!(combined, [0.6]);
    }

    #[test]
    fn independent_composition_rejects_shape_and_probability_errors() {
        assert!(independent_kill_probability(&[0.2], &[]).is_err());
        assert!(independent_kill_probability(&[1.1], &[0.5]).is_err());
    }
}
