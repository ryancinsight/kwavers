//! Aperture-driven array **design synthesis**.
//!
//! The rest of the crate (linear/matrix/2-D/phased arrays) takes a *fixed*
//! element count and computes spacing from it. This module solves the inverse,
//! clinical-design problem: given an **overall aperture size** and an **operating
//! frequency** (plus the medium sound speed and design constraints), derive a
//! complete element layout — element count per axis, centre-to-centre pitch, kerf
//! (inter-element gap), element size, and fill factor — with explicit
//! sub-wavelength (grating-lobe) classification, plus a **channel-wiring** map so
//! a physical 2-D matrix can be driven as fewer linear channels (the LeoNeuro fUS
//! device: an `el_x × el_y` matrix wired as `el_y` linear channels).
//!
//! The public contract uses Aequitas quantities; scalar `f64` values occur only
//! at Euclidean and design-formula boundaries. The result feeds the existing
//! array builders or a downstream geometry model without unit-bearing wrappers.

pub mod propagation;

pub use propagation::{
    propagate_focused_linear_array, FocusedLinearArrayPropagationSpec, FocusedPressureMap,
};

use aequitas::systems::si::quantities::{Dimensionless, Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond};
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// Pitch fraction of wavelength below which a steered array is grating-lobe-free
/// over its full steering range (the λ/2 spatial-Nyquist criterion).
pub const NYQUIST_PITCH_FRACTION: f64 = 0.5;
/// Default kerf as a fraction of pitch (10% gap is a common MEMS/PZT value).
pub const DEFAULT_KERF_FRACTION: f64 = 0.1;

/// How physical matrix elements are wired into independently-driven channels.
///
/// Axes follow the design convention below: `x` is the short / elevation /
/// curvature axis (`nx` columns), `y` is the long / steering axis (`ny`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelWiring {
    /// Every element is its own channel (a fully-populated 2-D phased array).
    PerElement,
    /// The `nx` elements sharing a long-axis position drive one channel — a
    /// matrix wired as `ny` **linear** channels (steering along `y`).
    ColumnsAsChannels,
    /// The `ny` elements sharing a short-axis position drive one channel — `nx`
    /// channels (steering along `x`).
    RowsAsChannels,
}

/// Constraints for synthesizing an array design from an aperture + frequency.
///
/// `aperture_x` is the short / elevation axis, `aperture_y` the long /
/// steering axis. Set an aperture to `0` to collapse that axis to a single
/// element (e.g. a 1-D linear array is `aperture_x = element height`,
/// `el_x = 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApertureDesignSpec {
    /// Overall aperture on the short / elevation axis.
    pub aperture_x: Length<f64>,
    /// Overall aperture on the long / steering axis.
    pub aperture_y: Length<f64>,
    /// Operating frequency.
    pub frequency: Frequency<f64>,
    /// Medium sound speed.
    pub sound_speed: Velocity<f64>,
    /// Maximum centre-to-centre pitch as a fraction of wavelength. `0.5` (λ/2)
    /// is grating-lobe-free over the full steering range; `1.0` (λ) is lobe-free
    /// only near broadside but halves the channel count.
    pub max_pitch_fraction: Dimensionless<f64>,
    /// Inter-element kerf as a fraction of pitch (`0.1` ⇒ 10% gap).
    pub kerf_fraction: Dimensionless<f64>,
    /// How elements are wired into driven channels.
    pub wiring: ChannelWiring,
}

impl ApertureDesignSpec {
    /// Wavelength in the medium.
    #[must_use]
    pub fn wavelength(&self) -> Length<f64> {
        Length::from_unit::<Meter>(
            self.sound_speed.in_unit::<MeterPerSecond>() / self.frequency.in_unit::<Hertz>(),
        )
    }
}

/// A fully-resolved array design (element layout + wiring + classification).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrayDesign {
    /// Element count on the short / elevation axis.
    pub nx: usize,
    /// Element count on the long / steering axis.
    pub ny: usize,
    /// Centre-to-centre pitch on `x` (`≤ max_pitch`).
    pub pitch_x: Length<f64>,
    /// Centre-to-centre pitch on `y` (`≤ max_pitch`).
    pub pitch_y: Length<f64>,
    /// Element extent on `x` (`pitch_x − kerf_x`).
    pub element_x: Length<f64>,
    /// Element extent on `y` (`pitch_y − kerf_y`).
    pub element_y: Length<f64>,
    /// Kerf (inter-element gap) on `x`.
    pub kerf_x: Length<f64>,
    /// Kerf (inter-element gap) on `y`.
    pub kerf_y: Length<f64>,
    /// Wavelength used for the design.
    pub wavelength: Length<f64>,
    /// Number of independently-driven channels after wiring.
    pub n_channels: usize,
    /// Wiring used.
    pub wiring: ChannelWiring,
    /// Whether the steered axis pitch satisfies λ/2 (grating-lobe-free steering).
    pub grating_lobe_free: bool,
}

impl ArrayDesign {
    /// Total physical element count (`nx · ny`).
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.nx * self.ny
    }

    /// Realized aperture on `x` (`nx · pitch_x`).
    #[must_use]
    pub fn aperture_x(&self) -> Length<f64> {
        Length::from_unit::<Meter>(self.nx as f64 * self.pitch_x.in_unit::<Meter>())
    }

    /// Realized aperture on `y` (`ny · pitch_y`).
    #[must_use]
    pub fn aperture_y(&self) -> Length<f64> {
        Length::from_unit::<Meter>(self.ny as f64 * self.pitch_y.in_unit::<Meter>())
    }

    /// Areal fill factor (active element area / pitch cell area).
    #[must_use]
    pub fn fill_factor(&self) -> Dimensionless<f64> {
        let cell = self.pitch_x.in_unit::<Meter>() * self.pitch_y.in_unit::<Meter>();
        let value = if cell > 0.0 {
            (self.element_x.in_unit::<Meter>() * self.element_y.in_unit::<Meter>()) / cell
        } else {
            0.0
        };
        Dimensionless::from_base(value)
    }

    /// Channel index of the element at flat index `idx`, for the column-major
    /// layout `idx = i·ny + j` (`i ∈ 0..nx`, `j ∈ 0..ny`). Matches the wiring.
    #[must_use]
    pub fn channel_of_index(&self, idx: usize) -> usize {
        match self.wiring {
            ChannelWiring::PerElement => idx,
            ChannelWiring::ColumnsAsChannels => idx % self.ny.max(1),
            ChannelWiring::RowsAsChannels => idx / self.ny.max(1),
        }
    }

    /// Element-centre coordinates for the synthesized layout, centred on
    /// `center`. The array lies in the `x`–`y` plane at constant `z = center\[2\]`;
    /// element `(i, j)` is at flat index `idx = i·ny + j` (`i ∈ 0..nx` short axis,
    /// `j ∈ 0..ny` long axis), matching [`Self::channel_of_index`]. This is the
    /// bridge from an abstract design to concrete geometry consumers. Scalar
    /// extraction belongs at those consumers' mesh or storage boundaries.
    #[must_use]
    pub fn element_positions(&self, center: [Length<f64>; 3]) -> Vec<[Length<f64>; 3]> {
        let center_m = center.map(|coordinate| coordinate.in_unit::<Meter>());
        let pitch_x_m = self.pitch_x.in_unit::<Meter>();
        let pitch_y_m = self.pitch_y.in_unit::<Meter>();
        let x0 = center_m[0] - 0.5 * (self.nx as f64 - 1.0) * pitch_x_m;
        let y0 = center_m[1] - 0.5 * (self.ny as f64 - 1.0) * pitch_y_m;
        let mut out = Vec::with_capacity(self.n_elements());
        for i in 0..self.nx {
            for j in 0..self.ny {
                out.push([
                    Length::from_unit::<Meter>((i as f64).mul_add(pitch_x_m, x0)),
                    Length::from_unit::<Meter>((j as f64).mul_add(pitch_y_m, y0)),
                    center[2],
                ]);
            }
        }
        out
    }

    /// Driven-channel centre coordinates (`n_channels` points): the centroid
    /// of the elements wired into each channel, indexed as [`Self::channel_of_index`].
    /// For `ColumnsAsChannels` this is a linear array of `ny` points along the long
    /// (`y`) steering axis at the short-axis centre; for `RowsAsChannels`, `nx`
    /// points along `x`; for `PerElement` it equals [`Self::element_positions`].
    #[must_use]
    pub fn channel_positions(&self, center: [Length<f64>; 3]) -> Vec<[Length<f64>; 3]> {
        let center_m = center.map(|coordinate| coordinate.in_unit::<Meter>());
        let pitch_x_m = self.pitch_x.in_unit::<Meter>();
        let pitch_y_m = self.pitch_y.in_unit::<Meter>();
        let x0 = center_m[0] - 0.5 * (self.nx as f64 - 1.0) * pitch_x_m;
        let y0 = center_m[1] - 0.5 * (self.ny as f64 - 1.0) * pitch_y_m;
        match self.wiring {
            ChannelWiring::PerElement => self.element_positions(center),
            // One channel per long-axis position j; centroid over the short axis is
            // the array centre in x.
            ChannelWiring::ColumnsAsChannels => (0..self.ny)
                .map(|j| {
                    [
                        center[0],
                        Length::from_unit::<Meter>((j as f64).mul_add(pitch_y_m, y0)),
                        center[2],
                    ]
                })
                .collect(),
            ChannelWiring::RowsAsChannels => (0..self.nx)
                .map(|i| {
                    [
                        Length::from_unit::<Meter>((i as f64).mul_add(pitch_x_m, x0)),
                        center[1],
                        center[2],
                    ]
                })
                .collect(),
        }
    }
}

/// Whether a centre-to-centre `pitch` is grating-lobe-free at `wavelength`
/// (the λ/2 spatial-Nyquist criterion).
#[must_use]
pub fn is_grating_lobe_free(pitch: Length<f64>, wavelength: Length<f64>) -> bool {
    pitch.in_unit::<Meter>() <= NYQUIST_PITCH_FRACTION * wavelength.in_unit::<Meter>()
}

/// Element count and realized pitch on one axis: the smallest `n ≥ 1` whose
/// uniform pitch `aperture / n` does not exceed `max_pitch`. An aperture `≤ 0`
/// collapses to a single element spanning `max_pitch`.
fn resolve_axis(aperture: Length<f64>, max_pitch: Length<f64>) -> (usize, Length<f64>) {
    let aperture_m = aperture.in_unit::<Meter>();
    let max_pitch_m = max_pitch.in_unit::<Meter>();
    if aperture_m <= 0.0 {
        return (1, max_pitch);
    }
    let n = (aperture_m / max_pitch_m).ceil().max(1.0) as usize;
    (n, Length::from_unit::<Meter>(aperture_m / n as f64))
}

/// Synthesize a complete [`ArrayDesign`] from an [`ApertureDesignSpec`].
///
/// # Errors
/// `KwaversError::Config` if the frequency or sound speed is non-positive, the
/// pitch fraction is outside `(0, 2]`, or the kerf fraction is outside `[0, 0.95]`.
pub fn design_array(spec: &ApertureDesignSpec) -> KwaversResult<ArrayDesign> {
    let invalid = |parameter: &str, value: f64, constraint: &str| {
        KwaversError::Config(ConfigError::InvalidValue {
            parameter: parameter.to_owned(),
            value: value.to_string(),
            constraint: constraint.to_owned(),
        })
    };
    // `!is_finite() || <= 0` rejects NaN/±inf as well as non-positive values.
    let frequency_hz = spec.frequency.in_unit::<Hertz>();
    let sound_speed_m_s = spec.sound_speed.in_unit::<MeterPerSecond>();
    let aperture_x_m = spec.aperture_x.in_unit::<Meter>();
    let aperture_y_m = spec.aperture_y.in_unit::<Meter>();
    let max_pitch_fraction = spec.max_pitch_fraction.into_base();
    let kerf_fraction = spec.kerf_fraction.into_base();
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(invalid("frequency", frequency_hz, "> 0"));
    }
    if !sound_speed_m_s.is_finite() || sound_speed_m_s <= 0.0 {
        return Err(invalid("sound_speed", sound_speed_m_s, "> 0"));
    }
    if !aperture_x_m.is_finite() || aperture_x_m < 0.0 {
        return Err(invalid("aperture_x", aperture_x_m, "finite and >= 0"));
    }
    if !aperture_y_m.is_finite() || aperture_y_m < 0.0 {
        return Err(invalid("aperture_y", aperture_y_m, "finite and >= 0"));
    }
    if !(max_pitch_fraction > 0.0 && max_pitch_fraction <= 2.0) {
        return Err(invalid(
            "max_pitch_fraction",
            max_pitch_fraction,
            "in (0, 2]",
        ));
    }
    if !(0.0..=0.95).contains(&kerf_fraction) {
        return Err(invalid("kerf_fraction", kerf_fraction, "in [0, 0.95]"));
    }

    let wavelength = spec.wavelength();
    let wavelength_m = wavelength.in_unit::<Meter>();
    let max_pitch = Length::from_unit::<Meter>(max_pitch_fraction * wavelength_m);
    let (nx, pitch_x) = resolve_axis(spec.aperture_x, max_pitch);
    let (ny, pitch_y) = resolve_axis(spec.aperture_y, max_pitch);
    let kerf_x = Length::from_unit::<Meter>(kerf_fraction * pitch_x.in_unit::<Meter>());
    let kerf_y = Length::from_unit::<Meter>(kerf_fraction * pitch_y.in_unit::<Meter>());

    let n_channels = match spec.wiring {
        ChannelWiring::PerElement => nx * ny,
        ChannelWiring::ColumnsAsChannels => ny,
        ChannelWiring::RowsAsChannels => nx,
    };
    // The steered axis (where electronic steering happens) must be λ/2 to be
    // grating-lobe-free over the full steering range.
    let steered_pitch = match spec.wiring {
        ChannelWiring::ColumnsAsChannels => pitch_y,
        ChannelWiring::RowsAsChannels => pitch_x,
        ChannelWiring::PerElement => {
            if pitch_x.in_unit::<Meter>() >= pitch_y.in_unit::<Meter>() {
                pitch_x
            } else {
                pitch_y
            }
        }
    };

    Ok(ArrayDesign {
        nx,
        ny,
        pitch_x,
        pitch_y,
        element_x: Length::from_unit::<Meter>(
            pitch_x.in_unit::<Meter>() - kerf_x.in_unit::<Meter>(),
        ),
        element_y: Length::from_unit::<Meter>(
            pitch_y.in_unit::<Meter>() - kerf_y.in_unit::<Meter>(),
        ),
        kerf_x,
        kerf_y,
        wavelength,
        n_channels,
        wiring: spec.wiring,
        grating_lobe_free: is_grating_lobe_free(steered_pitch, wavelength),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brain_spec() -> ApertureDesignSpec {
        ApertureDesignSpec {
            aperture_x: Length::from_unit::<Meter>(5.0e-3),
            aperture_y: Length::from_unit::<Meter>(20.0e-3),
            frequency: Frequency::from_unit::<Hertz>(4.0e6),
            sound_speed: Velocity::from_unit::<MeterPerSecond>(1560.0),
            max_pitch_fraction: Dimensionless::from_base(NYQUIST_PITCH_FRACTION),
            kerf_fraction: Dimensionless::from_base(DEFAULT_KERF_FRACTION),
            wiring: ChannelWiring::ColumnsAsChannels,
        }
    }

    #[test]
    fn pitch_is_subwavelength_and_count_spans_aperture() {
        let d = design_array(&brain_spec()).unwrap();
        let lambda: f64 = 1560.0 / 4.0e6; // 390 µm
        let max_pitch = 0.5 * lambda; // 195 µm
                                      // Pitch never exceeds the λ/2 budget and is grating-lobe-free.
        assert!(d.pitch_y.in_unit::<Meter>() <= max_pitch + 1e-12);
        assert!(d.pitch_x.in_unit::<Meter>() <= max_pitch + 1e-12);
        assert!(d.grating_lobe_free);
        // ny = ceil(20mm / 195µm) = 103; the realized aperture matches.
        assert_eq!(d.ny, (20.0e-3 / max_pitch).ceil() as usize);
        assert!((d.aperture_y().in_unit::<Meter>() - 20.0e-3).abs() < 1e-9);
        // Kerf is 10% of pitch; element = pitch − kerf.
        assert!((d.kerf_y.in_unit::<Meter>() - 0.1 * d.pitch_y.in_unit::<Meter>()).abs() < 1e-15);
        assert!(
            (d.element_y.in_unit::<Meter>() - 0.9 * d.pitch_y.in_unit::<Meter>()).abs() < 1e-15
        );
        assert!((d.fill_factor().into_base() - 0.81).abs() < 1e-9);
    }

    #[test]
    fn columns_wire_into_long_axis_channels() {
        let d = design_array(&brain_spec()).unwrap();
        assert_eq!(d.n_channels, d.ny, "columns-as-channels ⇒ ny channels");
        // Elements sharing a long-axis position j share a channel.
        assert_eq!(d.channel_of_index(0), 0);
        assert_eq!(d.channel_of_index(d.ny), 0, "next column, same channel 0");
        assert_eq!(d.channel_of_index(d.ny - 1), d.ny - 1);
    }

    #[test]
    fn coarser_pitch_fraction_halves_channel_count() {
        let mut s = brain_spec();
        s.max_pitch_fraction = Dimensionless::from_base(1.0); // λ instead of λ/2
        let d = design_array(&s).unwrap();
        let lambda: f64 = 1560.0 / 4.0e6;
        assert_eq!(d.ny, (20.0e-3 / lambda).ceil() as usize);
        // λ pitch is NOT grating-lobe-free over full steering.
        assert!(!d.grating_lobe_free);
    }

    #[test]
    fn zero_aperture_axis_collapses_to_single_element() {
        let mut s = brain_spec();
        s.aperture_x = Length::from_unit::<Meter>(0.0); // 1-D linear
        let d = design_array(&s).unwrap();
        assert_eq!(d.nx, 1);
    }

    #[test]
    fn element_positions_span_aperture_and_match_pitch() {
        let d = design_array(&brain_spec()).unwrap();
        let center = [
            Length::from_unit::<Meter>(0.0),
            Length::from_unit::<Meter>(0.0),
            Length::from_unit::<Meter>(0.030),
        ];
        let pos = d.element_positions(center);
        assert_eq!(pos.len(), d.n_elements());

        // Centred: mean position is the centre; extent equals (n-1)·pitch.
        let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &pos {
            assert!((p[2].in_unit::<Meter>() - center[2].in_unit::<Meter>()).abs() < 1e-15);
            xmin = xmin.min(p[0].in_unit::<Meter>());
            xmax = xmax.max(p[0].in_unit::<Meter>());
            ymin = ymin.min(p[1].in_unit::<Meter>());
            ymax = ymax.max(p[1].in_unit::<Meter>());
        }
        assert!((0.5 * (xmin + xmax) - center[0].in_unit::<Meter>()).abs() < 1e-12);
        assert!((0.5 * (ymin + ymax) - center[1].in_unit::<Meter>()).abs() < 1e-12);
        assert!((xmax - xmin - (d.nx as f64 - 1.0) * d.pitch_x.in_unit::<Meter>()).abs() < 1e-12);
        assert!((ymax - ymin - (d.ny as f64 - 1.0) * d.pitch_y.in_unit::<Meter>()).abs() < 1e-12);

        // Adjacent long-axis neighbours (idx and idx+1) are one pitch_y apart.
        let step = pos[1][1].in_unit::<Meter>() - pos[0][1].in_unit::<Meter>();
        assert!((step - d.pitch_y.in_unit::<Meter>()).abs() < 1e-12);
    }

    #[test]
    fn channel_positions_collapse_columns_to_a_linear_array() {
        let d = design_array(&brain_spec()).unwrap(); // ColumnsAsChannels
        let center = [
            Length::from_unit::<Meter>(0.0),
            Length::from_unit::<Meter>(0.0),
            Length::from_unit::<Meter>(0.030),
        ];
        let chan = d.channel_positions(center);
        assert_eq!(chan.len(), d.n_channels);
        assert_eq!(chan.len(), d.ny);

        // All channels sit at the short-axis centre (x = center) and step by pitch_y.
        for c in &chan {
            assert!((c[0].in_unit::<Meter>() - center[0].in_unit::<Meter>()).abs() < 1e-12);
        }
        let step = chan[1][1].in_unit::<Meter>() - chan[0][1].in_unit::<Meter>();
        assert!((step - d.pitch_y.in_unit::<Meter>()).abs() < 1e-12);

        // Each channel centroid equals the mean of its member elements' positions.
        let elems = d.element_positions(center);
        for (idx, e) in elems.iter().enumerate() {
            let c = d.channel_of_index(idx);
            assert!(
                (e[1].in_unit::<Meter>() - chan[c][1].in_unit::<Meter>()).abs() < 1e-12,
                "element y matches its channel"
            );
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut s = brain_spec();
        s.frequency = Frequency::from_unit::<Hertz>(0.0);
        assert!(design_array(&s).is_err());
        let mut s = brain_spec();
        s.kerf_fraction = Dimensionless::from_base(1.5);
        assert!(design_array(&s).is_err());
        let mut s = brain_spec();
        s.max_pitch_fraction = Dimensionless::from_base(0.0);
        assert!(design_array(&s).is_err());
    }
}
