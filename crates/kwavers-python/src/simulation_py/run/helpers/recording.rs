use kwavers_domain::sensor::recorder::fields::{SensorRecordField, SensorRecordSpec};

use super::super::super::Simulation;

impl Simulation {
    /// Map k-Wave-style `sensor.record` strings to a [`SensorRecordSpec`].
    pub(crate) fn record_modes_to_spec(modes: &[String]) -> SensorRecordSpec {
        let mut fields = vec![SensorRecordField::Pressure];
        for s in modes {
            match s.as_str() {
                "p" => {}
                "p_max" => fields.push(SensorRecordField::PressureMax),
                "p_min" => fields.push(SensorRecordField::PressureMin),
                "p_rms" => fields.push(SensorRecordField::PressureRms),
                "p_final" => fields.push(SensorRecordField::PressureFinal),
                "all" => {
                    fields.push(SensorRecordField::PressureMax);
                    fields.push(SensorRecordField::PressureMin);
                    fields.push(SensorRecordField::PressureRms);
                    fields.push(SensorRecordField::PressureFinal);
                }
                "ux" => fields.push(SensorRecordField::VelocityX),
                "uy" => fields.push(SensorRecordField::VelocityY),
                "uz" => fields.push(SensorRecordField::VelocityZ),
                "ux_max" => fields.push(SensorRecordField::VelocityMaxX),
                "uy_max" => fields.push(SensorRecordField::VelocityMaxY),
                "uz_max" => fields.push(SensorRecordField::VelocityMaxZ),
                "ux_min" => fields.push(SensorRecordField::VelocityMinX),
                "uy_min" => fields.push(SensorRecordField::VelocityMinY),
                "uz_min" => fields.push(SensorRecordField::VelocityMinZ),
                "ux_rms" => fields.push(SensorRecordField::VelocityRmsX),
                "uy_rms" => fields.push(SensorRecordField::VelocityRmsY),
                "uz_rms" => fields.push(SensorRecordField::VelocityRmsZ),
                "ux_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredX),
                "uy_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredY),
                "uz_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredZ),
                "Ix" => fields.push(SensorRecordField::IntensityX),
                "Iy" => fields.push(SensorRecordField::IntensityY),
                "Iz" => fields.push(SensorRecordField::IntensityZ),
                "I_avg_x" => fields.push(SensorRecordField::IntensityAvgX),
                "I_avg_y" => fields.push(SensorRecordField::IntensityAvgY),
                "I_avg_z" => fields.push(SensorRecordField::IntensityAvgZ),
                _ => {}
            }
        }
        SensorRecordSpec::from_fields(&fields)
    }

    /// Trim the recorder buffer to `Nt` columns aligned with k-Wave's time-axis convention.
    pub(crate) fn trim_initial_recorder_sample(
        recorded_data: ndarray::Array2<f64>,
        time_steps: usize,
        record_start_index: usize,
    ) -> ndarray::Array2<f64> {
        let start = record_start_index.max(1).min(time_steps);
        let skip = start.saturating_sub(1);
        if recorded_data.ncols() > time_steps {
            recorded_data
                .slice(ndarray::s![.., skip..time_steps])
                .to_owned()
        } else {
            recorded_data.slice(ndarray::s![.., skip..]).to_owned()
        }
    }

    /// Borrowed-view variant of [`trim_initial_recorder_sample`].
    pub(crate) fn trim_initial_recorder_view(
        recorded_data: ndarray::ArrayView2<'_, f64>,
        time_steps: usize,
        record_start_index: usize,
    ) -> ndarray::Array2<f64> {
        let start = record_start_index.max(1).min(time_steps);
        let skip = start.saturating_sub(1);
        if recorded_data.ncols() > time_steps {
            recorded_data
                .slice(ndarray::s![.., skip..time_steps])
                .to_owned()
        } else {
            recorded_data.slice(ndarray::s![.., skip..]).to_owned()
        }
    }

    /// Return the minimum active axis length and admissible CPML thickness.
    pub(crate) fn cpml_thickness_limits(nx: usize, ny: usize, nz: usize) -> (usize, usize) {
        let mut min_dim = usize::MAX;
        for dim in [nx, ny, nz] {
            if dim > 1 {
                min_dim = min_dim.min(dim);
            }
        }
        let min_dim = if min_dim == usize::MAX { 1 } else { min_dim };
        let max_allowed = (min_dim.saturating_sub(2)) / 2;
        let default_thickness = 20_usize.min(max_allowed).max(2);
        (default_thickness, max_allowed)
    }
}
