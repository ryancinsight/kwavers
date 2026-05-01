use super::model::BubbleField;

/// Statistics about bubble field.
#[derive(Debug, Default)]
pub struct BubbleFieldStats {
    pub total_bubbles: usize,
    pub collapsing_bubbles: usize,
    pub max_temperature: f64,
    pub max_compression: f64,
    pub total_collapses: u32,
}

impl BubbleField {
    /// Get statistics about bubble field.
    #[must_use]
    pub fn get_statistics(&self) -> BubbleFieldStats {
        let mut stats = BubbleFieldStats::default();

        for state in self.bubbles.values() {
            stats.total_bubbles += 1;
            if state.is_collapsing {
                stats.collapsing_bubbles += 1;
            }
            stats.max_temperature = stats.max_temperature.max(state.temperature);
            stats.max_compression = stats.max_compression.max(state.compression_ratio);
            stats.total_collapses += state.collapse_count;
        }

        stats
    }
}
