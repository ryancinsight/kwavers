use super::model::BubbleField;

impl BubbleField {
    /// Get bubble state fields for physics modules.
    #[must_use]
    pub fn get_state_fields(&self) -> kwavers_field::BubbleStateFields {
        let shape = self.grid_shape;
        let mut fields = kwavers_field::BubbleStateFields::new(shape);

        for ((i, j, k), state) in &self.bubbles {
            fields.radius[[*i, *j, *k]] = state.radius;
            fields.temperature[[*i, *j, *k]] = state.temperature;
            fields.pressure[[*i, *j, *k]] = state.pressure_internal;
            fields.velocity[[*i, *j, *k]] = state.wall_velocity;
            fields.is_collapsing[[*i, *j, *k]] = f64::from(i32::from(state.is_collapsing));
            fields.compression_ratio[[*i, *j, *k]] = state.compression_ratio;
        }

        fields
    }
}
