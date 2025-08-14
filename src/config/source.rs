use crate::grid::Grid;
use crate::medium::Medium;
use crate::signal::{chirp::ChirpSignal, sine_wave::SineWave, sweep::SweepSignal, Signal};
use crate::source::Source;
use crate::source::{apodization::HanningApodization, linear_array::LinearArray};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SourceConfig {
    pub num_elements: usize,
    pub signal_type: String,
    pub start_freq: Option<f64>,
    pub end_freq: Option<f64>,
    pub signal_duration: Option<f64>,
    pub phase: Option<f64>,
    pub focus_x: Option<f64>,
    pub focus_y: Option<f64>,
    pub focus_z: Option<f64>,
    pub frequency: Option<f64>, // Added for signal specific frequency
    pub amplitude: Option<f64>, // Added for signal specific amplitude
}

impl SourceConfig {
    pub fn initialize_source(
        &self,
        medium: &dyn Medium,
        grid: &Grid,
        // Parameters below might be needed if SourceConfig doesn't have its own Freq/Amp
        // default_frequency: f64,
        // default_amplitude: f64,
    ) -> Result<Box<dyn Source>, String> {
        let signal_frequency = self.frequency.ok_or_else(|| "frequency is required in SourceConfig for selected signal_type".to_string())?;
        let signal_amplitude = self.amplitude.ok_or_else(|| "amplitude is required in SourceConfig for selected signal_type".to_string())?;

        let signal: Box<dyn Signal> = match self.signal_type.as_str() {
            "sine" => Box::new(SineWave::new(
                signal_frequency,
                signal_amplitude,
                self.phase.unwrap_or(0.0),
            )),
            "sweep" => Box::new(SweepSignal::new(
                self.start_freq.ok_or("start_freq required")?,
                self.end_freq.ok_or("end_freq required")?,
                self.signal_duration.ok_or("signal_duration required")?,
                signal_amplitude, // Use common amplitude
                self.phase.unwrap_or(0.0),
            )),
            "chirp" => Box::new(ChirpSignal::new(
                self.start_freq.ok_or("start_freq required")?,
                self.end_freq.ok_or("end_freq required")?,
                self.signal_duration.ok_or("signal_duration required")?,
                signal_amplitude, // Use common amplitude
                self.phase.unwrap_or(0.0),
            )),
            _ => return Err(format!("Unknown signal_type: {}", self.signal_type)),
        };

        let source = if self.focus_x.is_some() && self.focus_y.is_some() && self.focus_z.is_some() {
            let mut array = LinearArray::new(
                0.1, // Default length
                self.num_elements,
                0.0,
                0.0,
                signal,
                medium,
                grid,
                self.frequency,
                crate::source::RectangularApodization,
            );
            array.adjust_focus(
                self.focus_x.unwrap(),
                self.focus_y.unwrap(),
                self.focus_z.unwrap(),
                medium,
                grid,
            );
            Box::new(array) as Box<dyn Source>
        } else {
            Box::new(LinearArray::new(
                0.1,
                self.num_elements,
                0.0,
                0.0,
                signal,
                medium,
                grid,
                signal_frequency, // Use signal's frequency
                HanningApodization, // Added default apodization
            )) as Box<dyn Source>
        };

        Ok(source)
    }
}
