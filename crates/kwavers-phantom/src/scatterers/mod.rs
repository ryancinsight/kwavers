//! Discrete point-scatterer clouds and synthetic-aperture RF synthesis.
//!
//! This is the kwavers analogue of the Field II core abstraction: a tissue model
//! as a cloud of discrete point scatterers, each with a position and a scattering
//! amplitude, from which per-element radio-frequency (RF) echo traces are
//! synthesized.
//!
//! # Echo model
//!
//! [`ScattererCloud::synthesize_rf`] uses the **monostatic synthetic-aperture**
//! point-element model: each transducer element independently transmits and
//! receives. For element at position `e` and scatterer `s`,
//!
//! ```text
//! RF_e(t) = Σ_s ( a_s / r² ) · pulse( t − 2r/c ),     r = ‖e − s‖
//! ```
//!
//! where `a_s` is the scatterer amplitude, `2r/c` the pulse-echo time of flight,
//! and `1/r²` the round-trip spherical spreading (`1/r` on transmit and `1/r` on
//! receive). This is the correct far-field point-element limit and the basis of
//! synthetic-aperture imaging.
//!
//! Optional power-law tissue attenuation scales each echo by the round-trip
//! amplitude factor `exp(−α(f₀)·2r)` (see [`RfSynthesisConfig`]); with it
//! disabled the model reduces to pure spreading.
//!
//! It does **not** model finite-aperture diffraction — the Tupholme–Stepanishen
//! spatial impulse response that Field II convolves in for extended elements.
//! [`ScattererCloud::synthesize_rf_with_aperture`] adds that refinement through
//! an injected kernel (ADR 113); the point-element model is exact for point
//! elements and far-field scatterers. [`ScattererCloud::synthesize_rf_with_array_transmit`]
//! is the full Field II model for a delayed, apodized array transmit received
//! on every element, amplitude and all from the spatial impulse responses.
//! [`ScattererCloud::synthesize_rf_with_transmit`] additionally models one
//! active plane-wave or virtual-source transmit event received by every element.
//! The event's travel-time and spreading law are represented once by
//! [`TransmitWavefront`], allowing the same event to drive RF synthesis and
//! downstream transmit-aware DAS.
//!
//! # References
//! - Jensen, J. A. (1991). "A model for the propagation and scattering of
//!   ultrasound in tissue." *J. Acoust. Soc. Am.* 89(1), 182–190 (Field II model).
//! - Jensen, J. A., & Svendsen, N. B. (1992). "Calculation of pressure fields from
//!   arbitrarily shaped, apodized, and excited ultrasound transducers."
//!   *IEEE Trans. UFFC* 39(2), 262–267 (spatial impulse response).

mod aperture;

pub use aperture::{ApertureElement, ApertureKernel};

use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::numerics::convolution::{convolve_into, SupportSamples};
use leto::Array2;

/// A single discrete point scatterer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointScatterer {
    /// Position \`m`.
    pub position: [f64; 3],
    /// Scattering amplitude (dimensionless reflectivity; sign carries phase).
    pub amplitude: f64,
}

/// A cloud of discrete point scatterers (the Field II tissue model).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScattererCloud {
    scatterers: Vec<PointScatterer>,
}

/// Configuration for [`ScattererCloud::synthesize_rf`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RfSynthesisConfig {
    /// Sound speed `c` \[m/s].
    pub sound_speed: f64,
    /// Sampling frequency `fs` \`Hz`.
    pub sampling_frequency: f64,
    /// Number of output time samples per element.
    pub num_samples: usize,
    /// Minimum element–scatterer distance \`m`; closer scatterers are skipped to
    /// avoid the `1/r²` singularity (a scatterer inside the element footprint is
    /// outside the far-field point model).
    pub min_distance: f64,
    /// Power-law tissue attenuation `α₀` in dB/(cm·MHz). `0` ⇒ lossless (the
    /// pure spreading model). When positive, each echo is scaled by the
    /// round-trip amplitude factor `exp(−α(f₀)·2r)` evaluated at
    /// `center_frequency_hz`.
    pub attenuation_db_cm_mhz: f64,
    /// Pulse centre frequency `f₀` \`Hz` for the power-law attenuation. Only used
    /// when `attenuation_db_cm_mhz > 0`.
    pub center_frequency_hz: f64,
}

/// Active transmit event used by point-scatterer RF synthesis.
///
/// A plane event has unit transmit spreading and advances from `origin_m` in a
/// normalized `direction`. A diverging event starts at `source_m` and carries
/// the point-source `1/r_tx` transmit spreading. Both events use the physical
/// total path `r_tx + r_rx` for time of flight and attenuation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmitWavefront {
    origin_m: [f64; 3],
    kind: TransmitWavefrontKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TransmitWavefrontKind {
    Plane { direction_unit: [f64; 3] },
    Diverging,
}

impl TransmitWavefront {
    /// Construct a plane wave whose phase front passes `origin_m` at `t = 0`.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when either coordinate is
    /// non-finite or `direction` has zero length.
    pub fn plane(origin_m: [f64; 3], direction: [f64; 3]) -> KwaversResult<Self> {
        validate_coordinates("plane-wave origin_m", origin_m)?;
        validate_coordinates("plane-wave direction", direction)?;
        let norm = direction[0].hypot(direction[1]).hypot(direction[2]);
        if !norm.is_finite() || norm == 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "plane-wave direction must have finite nonzero length, got {direction:?}"
            )));
        }
        Ok(Self {
            origin_m,
            kind: TransmitWavefrontKind::Plane {
                direction_unit: [
                    direction[0] / norm,
                    direction[1] / norm,
                    direction[2] / norm,
                ],
            },
        })
    }

    /// Construct a spherical wave diverging from `source_m` at `t = 0`.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when `source_m` is non-finite.
    pub fn diverging(source_m: [f64; 3]) -> KwaversResult<Self> {
        validate_coordinates("diverging-wave source_m", source_m)?;
        Ok(Self {
            origin_m: source_m,
            kind: TransmitWavefrontKind::Diverging,
        })
    }

    /// Arrival time at `point_m` in seconds.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for non-finite coordinates or
    /// sound speed, a point behind a plane-wave origin, or a point at a
    /// diverging source singularity.
    pub fn arrival_time_s(&self, point_m: [f64; 3], sound_speed: f64) -> KwaversResult<f64> {
        validate_sound_speed(sound_speed)?;
        let (transmit_distance, _) = self.path_to(point_m)?;
        Ok(transmit_distance / sound_speed)
    }

    fn path_to(&self, point_m: [f64; 3]) -> KwaversResult<(f64, f64)> {
        validate_coordinates("transmit point_m", point_m)?;
        let dx = point_m[0] - self.origin_m[0];
        let dy = point_m[1] - self.origin_m[1];
        let dz = point_m[2] - self.origin_m[2];
        match self.kind {
            TransmitWavefrontKind::Plane { direction_unit } => {
                let distance = dx.mul_add(
                    direction_unit[0],
                    dy.mul_add(direction_unit[1], dz * direction_unit[2]),
                );
                if !distance.is_finite() || distance < 0.0 {
                    return Err(KwaversError::InvalidInput(format!(
                        "plane-wave point {point_m:?} is behind origin {:?}",
                        self.origin_m
                    )));
                }
                Ok((distance, 1.0))
            }
            TransmitWavefrontKind::Diverging => {
                let distance = dx.hypot(dy).hypot(dz);
                if !distance.is_finite() || distance == 0.0 {
                    return Err(KwaversError::InvalidInput(format!(
                        "diverging-wave point {point_m:?} coincides with source {:?}",
                        self.origin_m
                    )));
                }
                Ok((distance, 1.0 / distance))
            }
        }
    }

    fn is_diverging(self) -> bool {
        matches!(self.kind, TransmitWavefrontKind::Diverging)
    }
}

/// One transmit event of an element array: a delay and an apodization
/// weight per element, in element order.
///
/// Delays are non-negative (the earliest-firing element defines `t = 0`) and
/// apply on the sample grid of the synthesis they enter.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayTransmit {
    delays_s: Vec<f64>,
    apodization: Vec<f64>,
}

impl ArrayTransmit {
    /// Construct from per-element delays \[s] and apodization weights.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when the lengths differ, a value
    /// is non-finite, or a delay is negative.
    pub fn new(delays_s: Vec<f64>, apodization: Vec<f64>) -> KwaversResult<Self> {
        if delays_s.len() != apodization.len() {
            return Err(KwaversError::InvalidInput(format!(
                "ArrayTransmit requires one apodization weight per delay, got {} delays and {} weights",
                delays_s.len(),
                apodization.len()
            )));
        }
        if let Some(delay) = delays_s.iter().find(|d| !d.is_finite() || **d < 0.0) {
            return Err(KwaversError::InvalidInput(format!(
                "ArrayTransmit delays must be finite and >= 0, got {delay}"
            )));
        }
        if let Some(weight) = apodization.iter().find(|w| !w.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "ArrayTransmit apodization must be finite, got {weight}"
            )));
        }
        Ok(Self {
            delays_s,
            apodization,
        })
    }

    /// Every element firing at once with unit weight: a plane wave along the
    /// array normal.
    #[must_use]
    pub fn plane(element_count: usize) -> Self {
        Self {
            delays_s: vec![0.0; element_count],
            apodization: vec![1.0; element_count],
        }
    }

    /// Unit-weight delays that bring every element's arrival at `focus_m`
    /// into coincidence: `τ_m = (l_max − l_m)/c` with `l_m` the element-centre
    /// distance, so the farthest element fires first at `t = 0`.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for a non-finite focus or a
    /// non-positive sound speed.
    pub fn focused(
        elements: &[ApertureElement],
        focus_m: [f64; 3],
        sound_speed: f64,
    ) -> KwaversResult<Self> {
        validate_coordinates("focus_m", focus_m)?;
        validate_sound_speed(sound_speed)?;
        let distances: Vec<f64> = elements
            .iter()
            .map(|e| distance(e.position, focus_m))
            .collect();
        let farthest = distances.iter().copied().fold(0.0_f64, f64::max);
        Ok(Self {
            delays_s: distances
                .iter()
                .map(|l| (farthest - l) / sound_speed)
                .collect(),
            apodization: vec![1.0; elements.len()],
        })
    }

    /// Number of elements the event addresses.
    #[must_use]
    pub fn len(&self) -> usize {
        self.delays_s.len()
    }

    /// `true` when the event addresses no element.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.delays_s.is_empty()
    }

    /// Per-element delays \[s].
    #[must_use]
    pub fn delays_s(&self) -> &[f64] {
        &self.delays_s
    }

    /// Per-element apodization weights.
    #[must_use]
    pub fn apodization(&self) -> &[f64] {
        &self.apodization
    }
}

impl ScattererCloud {
    /// Empty cloud.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from explicit positions and matching amplitudes.
    ///
    /// # Errors
    /// - `KwaversError::InvalidInput` if the lengths differ.
    pub fn from_points(positions: &[[f64; 3]], amplitudes: &[f64]) -> KwaversResult<Self> {
        if positions.len() != amplitudes.len() {
            return Err(KwaversError::InvalidInput(format!(
                "ScattererCloud::from_points: {} positions vs {} amplitudes",
                positions.len(),
                amplitudes.len()
            )));
        }
        Ok(Self {
            scatterers: positions
                .iter()
                .zip(amplitudes)
                .map(|(&position, &amplitude)| PointScatterer {
                    position,
                    amplitude,
                })
                .collect(),
        })
    }

    /// Append a scatterer.
    pub fn push(&mut self, position: [f64; 3], amplitude: f64) {
        self.scatterers.push(PointScatterer {
            position,
            amplitude,
        });
    }

    /// The scatterers.
    #[must_use]
    pub fn scatterers(&self) -> &[PointScatterer] {
        &self.scatterers
    }

    /// Number of scatterers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scatterers.len()
    }

    /// Whether the cloud is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scatterers.is_empty()
    }

    /// Synthesize per-element monostatic pulse-echo RF, shape `(n_elements, num_samples)`.
    ///
    /// See the [module docs](self) for the echo model. Each element transmits and
    /// receives the `pulse` (a sampled excitation at `config.sampling_frequency`).
    ///
    /// # Errors
    /// - `KwaversError::InvalidInput` if `sound_speed` or `sampling_frequency`
    ///   is non-finite/`≤ 0`, `num_samples == 0`, `pulse` is empty, or
    ///   `min_distance < 0`.
    pub fn synthesize_rf(
        &self,
        element_positions: &[[f64; 3]],
        pulse: &[f64],
        config: &RfSynthesisConfig,
    ) -> KwaversResult<Array2<f64>> {
        let attenuation_np_m = self.validate_synthesis(element_positions, pulse, config)?;
        self.synthesize_with_paths(
            element_positions,
            pulse,
            config,
            attenuation_np_m,
            |element, scatterer| {
                let range = distance(element, scatterer.position);
                if range < config.min_distance || range == 0.0 {
                    return Ok(None);
                }
                Ok(Some((
                    2.0 * range / config.sound_speed,
                    2.0 * range,
                    1.0 / (range * range),
                )))
            },
        )
    }

    /// Synthesize active-imaging RF for one plane-wave or virtual-source event.
    ///
    /// Every receive element observes `pulse` after the shared transmit arrival
    /// plus its own one-way receive travel time. Plane waves use unit transmit
    /// spreading; diverging waves use `1/r_tx`, so the complete point-scatterer
    /// amplitude is `a/r_rx` or `a/(r_tx·r_rx)`, respectively.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for invalid synthesis inputs or a
    /// scatterer outside the selected event's physically forward domain.
    pub fn synthesize_rf_with_transmit(
        &self,
        receive_positions: &[[f64; 3]],
        pulse: &[f64],
        config: &RfSynthesisConfig,
        transmit: TransmitWavefront,
    ) -> KwaversResult<Array2<f64>> {
        let attenuation_np_m = self.validate_synthesis(receive_positions, pulse, config)?;
        self.synthesize_with_paths(
            receive_positions,
            pulse,
            config,
            attenuation_np_m,
            |receiver, scatterer| {
                let receive_distance = distance(receiver, scatterer.position);
                if receive_distance < config.min_distance || receive_distance == 0.0 {
                    return Ok(None);
                }
                let (transmit_distance, transmit_spreading) =
                    transmit.path_to(scatterer.position)?;
                if transmit.is_diverging() && transmit_distance < config.min_distance {
                    return Ok(None);
                }
                let total_distance = transmit_distance + receive_distance;
                Ok(Some((
                    total_distance / config.sound_speed,
                    total_distance,
                    transmit_spreading / receive_distance,
                )))
            },
        )
    }

    fn validate_synthesis(
        &self,
        element_positions: &[[f64; 3]],
        pulse: &[f64],
        config: &RfSynthesisConfig,
    ) -> KwaversResult<f64> {
        validate_sound_speed(config.sound_speed)?;
        if !config.sampling_frequency.is_finite() || config.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires sampling_frequency > 0, got {}",
                config.sampling_frequency
            )));
        }
        if config.num_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "synthesize_rf requires num_samples > 0".to_owned(),
            ));
        }
        if pulse.is_empty() || !pulse.iter().all(|sample| sample.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "synthesize_rf requires finite non-empty pulse samples".to_owned(),
            ));
        }
        if !config.min_distance.is_finite() || config.min_distance < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires min_distance >= 0, got {}",
                config.min_distance
            )));
        }
        if !config.attenuation_db_cm_mhz.is_finite() || config.attenuation_db_cm_mhz < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires attenuation_db_cm_mhz >= 0, got {}",
                config.attenuation_db_cm_mhz
            )));
        }
        if config.attenuation_db_cm_mhz > 0.0
            && (!config.center_frequency_hz.is_finite() || config.center_frequency_hz <= 0.0)
        {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires center_frequency_hz > 0 when attenuation is enabled, got {}",
                config.center_frequency_hz
            )));
        }
        for &position in element_positions {
            validate_coordinates("element position", position)?;
        }
        for scatterer in &self.scatterers {
            validate_coordinates("scatterer position", scatterer.position)?;
            if !scatterer.amplitude.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "scatterer amplitude must be finite, got {}",
                    scatterer.amplitude
                )));
            }
        }
        Ok(if config.attenuation_db_cm_mhz > 0.0 {
            config.attenuation_db_cm_mhz * (config.center_frequency_hz / MHZ_TO_HZ) * 100.0
                / NP_TO_DB
        } else {
            0.0
        })
    }

    /// Synthesize RF with the finite-aperture diffraction refinement.
    ///
    /// Same echo model as [`Self::synthesize_rf`] — monostatic
    /// synthetic-aperture, `1/r²` round-trip spreading, optional power-law
    /// attenuation — with each echo convolved by the element's round-trip
    /// spatial impulse response for that scatterer's geometry.
    ///
    /// The kernel enters as a **unit-area** filter, so it contributes the
    /// aperture's temporal response and leaves the amplitude law alone. That is
    /// what makes this a refinement of the point-element model rather than a
    /// different one: as the aperture shrinks the kernel tends to a delta and
    /// the output converges on `synthesize_rf`. Convolving the raw kernel
    /// instead would converge on silence, because its area
    /// `(√(z²+a²) − z)²` tends to zero with the radius. See ADR 113.
    ///
    /// Per element, every scatterer's scaled kernel is placed at its round-trip
    /// delay into one trace, and the pulse is convolved into that trace once;
    /// the per-pair cost is the kernel the provider returns, which samples its
    /// own support (see [`ApertureKernel`]); a sub-sample aperture arrives as
    /// its impulse equivalent and normalizes to the point-element impulse.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for a non-finite or non-positive
    /// configuration, an empty pulse, a non-finite synthesized amplitude, or a
    /// kernel that is empty, of zero area, or carries a non-finite sample.
    pub fn synthesize_rf_with_aperture(
        &self,
        elements: &[ApertureElement],
        pulse: &[f64],
        config: &RfSynthesisConfig,
        kernel: &impl ApertureKernel,
    ) -> KwaversResult<Array2<f64>> {
        let positions: Vec<[f64; 3]> = elements.iter().map(|e| e.position).collect();
        let attenuation_np_m = self.validate_synthesis(&positions, pulse, config)?;
        let dt = 1.0 / config.sampling_frequency;
        let num_samples = config.num_samples;

        let mut rf = Array2::<f64>::zeros([elements.len(), num_samples]);
        // Reused across elements: the accumulated diffraction trace and the
        // pulse-convolved row, so the per-element cost carries no allocation.
        let mut trace = vec![0.0_f64; num_samples];
        let mut row = vec![0.0_f64; num_samples];
        for (element_index, element) in elements.iter().enumerate() {
            trace.fill(0.0);
            for &scatterer in &self.scatterers {
                let distance = distance(element.position, scatterer.position);
                if distance < config.min_distance {
                    continue;
                }
                let Some([x, y, z]) = element.field_point(scatterer.position) else {
                    continue;
                };
                let spreading = 1.0 / (distance * distance);
                let amplitude =
                    scatterer.amplitude * spreading * (-attenuation_np_m * 2.0 * distance).exp();
                if !amplitude.is_finite() {
                    return Err(KwaversError::InvalidInput(format!(
                        "synthesize_rf_with_aperture produced non-finite amplitude for scatterer {:?}",
                        scatterer.position
                    )));
                }
                let time_s = 2.0 * distance / config.sound_speed;
                let sample_delay = (time_s * config.sampling_frequency).round();
                if !sample_delay.is_finite() || sample_delay < 0.0 {
                    continue;
                }
                let sample_delay = sample_delay as usize;
                if sample_delay >= num_samples {
                    continue;
                }
                let samples = kernel.round_trip(x, y, z, dt).samples;
                if !samples.iter().all(|s| s.is_finite()) {
                    return Err(KwaversError::InvalidInput(format!(
                        "round-trip kernel has a non-finite sample for scatterer {:?} in element frame [{x:.3e}, {y:.3e}, {z:.3e}]",
                        scatterer.position
                    )));
                }
                let Some((support, area)) = aperture::support_and_area(&samples, dt) else {
                    return Err(KwaversError::InvalidInput(format!(
                        "round-trip kernel is empty or has no area for scatterer {:?} in element frame [{x:.3e}, {y:.3e}, {z:.3e}]; a provider returns the impulse equivalent for a sub-sample support",
                        scatterer.position
                    )));
                };
                let scale = amplitude / area;
                for (slot, &sample) in trace[sample_delay..].iter_mut().zip(&samples[support]) {
                    *slot += scale * sample;
                }
            }
            row.fill(0.0);
            convolve_into(&mut row, &trace, pulse, dt);
            for (index, &value) in row.iter().enumerate() {
                rf[[element_index, index]] = value;
            }
        }
        Ok(rf)
    }

    /// Synthesize RF for one transmit event of a delayed, apodized element
    /// array, received on every element, with finite-aperture diffraction on
    /// both legs.
    ///
    /// This is the Field II model in full (Jensen 1991; Jensen & Svendsen
    /// 1992): for scatterer `s` and receive element `e`,
    ///
    /// ```text
    /// RF_e(t)    = Σ_s a_s · ( pulse ⊛ h_tx,s ⊛ h_rx,e,s )(t)
    /// h_tx,s(t)  = Σ_m w_m · e^{−α·l_m,s} · h_m,s(t − τ_m)
    /// h_rx,e,s(t) = e^{−α·l_e,s} · h_e,s(t)
    /// ```
    ///
    /// where `h_m,s` is element `m`'s one-way spatial impulse response at the
    /// scatterer ([`ApertureKernel::response`]), `τ_m` and `w_m` its transmit
    /// delay and apodization ([`ArrayTransmit`]), and `pulse` the caller's
    /// pulse-echo wavelet. The amplitude comes from the responses themselves —
    /// each carries `1/l` in its area — so, unlike
    /// [`Self::synthesize_rf_with_aperture`], no separate spreading law applies
    /// and the kernels are not normalized: the apodization and the array's
    /// directivity are in the trace. Attenuation is the power-law amplitude
    /// factor per element path. Delays round to the sample grid, as the
    /// point-element times of flight do. Each echo lands at the sum of its
    /// kernels' own onsets; no centre-distance delay is applied. `min_distance`
    /// excludes an element–scatterer pair from both legs — here it bounds the
    /// model's validity near the face rather than a `1/r²` singularity — and a
    /// scatterer behind an element's face is likewise skipped for it.
    ///
    /// Per scatterer the element responses are evaluated once and serve both
    /// legs: the transmit response is their delayed, weighted superposition
    /// and each receive echo one cross-convolution with it — the separable
    /// transmit/receive structure of Rivera, Demené & Tanter (2026, Eq. 24)
    /// in the time domain. The pulse is convolved once per receiver.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for an invalid configuration, an
    /// empty pulse, a `transmit` whose length differs from `elements`, or a
    /// kernel that is empty, of zero area, or carries a non-finite sample.
    pub fn synthesize_rf_with_array_transmit(
        &self,
        elements: &[ApertureElement],
        transmit: &ArrayTransmit,
        pulse: &[f64],
        config: &RfSynthesisConfig,
        kernel: &impl ApertureKernel,
    ) -> KwaversResult<Array2<f64>> {
        let positions: Vec<[f64; 3]> = elements.iter().map(|e| e.position).collect();
        let attenuation_np_m = self.validate_synthesis(&positions, pulse, config)?;
        if transmit.len() != elements.len() {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf_with_array_transmit requires one delay and apodization per element, got {} for {} elements",
                transmit.len(),
                elements.len()
            )));
        }
        let dt = 1.0 / config.sampling_frequency;
        let num_samples = config.num_samples;
        let shifts: Vec<usize> = transmit
            .delays_s()
            .iter()
            .map(|delay| (delay * config.sampling_frequency).round() as usize)
            .collect();

        let mut traces = Array2::<f64>::zeros([elements.len(), num_samples]);
        let mut responses: Vec<Option<SupportSamples>> = Vec::with_capacity(elements.len());
        for &scatterer in &self.scatterers {
            responses.clear();
            for element in elements {
                let distance = distance(element.position, scatterer.position);
                let field_point = element.field_point(scatterer.position);
                let Some([x, y, z]) = field_point.filter(|_| distance >= config.min_distance)
                else {
                    responses.push(None);
                    continue;
                };
                let mut response = kernel.response(x, y, z, dt);
                let finite = response.samples.iter().all(|s| s.is_finite());
                if response.is_empty() || !finite || response.area(dt) <= 0.0 {
                    return Err(KwaversError::InvalidInput(format!(
                        "one-way kernel is empty, has no area, or has a non-finite sample for scatterer {:?} in element frame [{x:.3e}, {y:.3e}, {z:.3e}]; a provider returns the impulse equivalent for a sub-sample support",
                        scatterer.position
                    )));
                }
                response.scale((-attenuation_np_m * distance).exp());
                responses.push(Some(response));
            }

            let mut transmit_response = SupportSamples::zero();
            for ((response, &shift), &weight) in
                responses.iter().zip(&shifts).zip(transmit.apodization())
            {
                if let Some(response) = response {
                    transmit_response.superpose(response, shift, weight);
                }
            }
            if transmit_response.is_empty() {
                continue;
            }
            for (element_index, response) in responses.iter().enumerate() {
                let Some(response) = response else {
                    continue;
                };
                let echo = transmit_response.convolve(response, dt);
                let placed = echo
                    .samples
                    .iter()
                    .enumerate()
                    .map(|(offset, &sample)| (echo.first_sample + offset, sample))
                    .take_while(|&(index, _)| index < num_samples);
                for (index, sample) in placed {
                    traces[[element_index, index]] += scatterer.amplitude * sample;
                }
            }
        }

        let mut rf = Array2::<f64>::zeros([elements.len(), num_samples]);
        let mut trace = vec![0.0_f64; num_samples];
        let mut row = vec![0.0_f64; num_samples];
        for element_index in 0..elements.len() {
            for (index, slot) in trace.iter_mut().enumerate() {
                *slot = traces[[element_index, index]];
            }
            row.fill(0.0);
            convolve_into(&mut row, &trace, pulse, dt);
            for (index, &value) in row.iter().enumerate() {
                rf[[element_index, index]] = value;
            }
        }
        Ok(rf)
    }

    fn synthesize_with_paths(
        &self,
        element_positions: &[[f64; 3]],
        pulse: &[f64],
        config: &RfSynthesisConfig,
        attenuation_np_m: f64,
        mut path: impl FnMut([f64; 3], PointScatterer) -> KwaversResult<Option<(f64, f64, f64)>>,
    ) -> KwaversResult<Array2<f64>> {
        let mut rf = Array2::<f64>::zeros([element_positions.len(), config.num_samples]);
        for (element_index, &element) in element_positions.iter().enumerate() {
            for &scatterer in &self.scatterers {
                let Some((time_s, path_length_m, spreading)) = path(element, scatterer)? else {
                    continue;
                };
                let amplitude =
                    scatterer.amplitude * spreading * (-attenuation_np_m * path_length_m).exp();
                if !amplitude.is_finite() {
                    return Err(KwaversError::InvalidInput(format!(
                        "synthesize_rf produced non-finite amplitude for scatterer {:?}",
                        scatterer.position
                    )));
                }
                let sample_delay = (time_s * config.sampling_frequency).round();
                if !sample_delay.is_finite() || sample_delay < 0.0 {
                    continue;
                }
                let sample_delay = sample_delay as usize;
                for (offset, &pulse_sample) in pulse.iter().enumerate() {
                    let Some(sample_index) = sample_delay.checked_add(offset) else {
                        continue;
                    };
                    if sample_index < config.num_samples {
                        rf[[element_index, sample_index]] += amplitude * pulse_sample;
                    }
                }
            }
        }
        Ok(rf)
    }
}

fn validate_sound_speed(sound_speed: f64) -> KwaversResult<()> {
    if sound_speed.is_finite() && sound_speed > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "sound_speed must be finite and > 0, got {sound_speed}"
        )))
    }
}

fn validate_coordinates(name: &str, coordinates: [f64; 3]) -> KwaversResult<()> {
    if coordinates.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must contain only finite coordinates, got {coordinates:?}"
        )))
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2])
}

#[cfg(test)]
mod tests;
