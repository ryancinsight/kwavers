# kwavers-signal

Excitation signal generation and processing for
[kwavers](https://github.com/ryancinsight/kwavers): waveforms, pulses, frequency sweeps,
modulation, windowing, and filters.

Everything a kwavers source emits is driven by a `Signal` — a time-domain function of
amplitude, frequency, and phase. This crate owns that trait and the waveform library that
implements it, independent of any grid, medium, or transducer geometry.

## What it provides

| Module | Responsibility |
|---|---|
| `traits` | The `Signal` trait: amplitude, instantaneous frequency and phase, optional duration |
| `waveform` | Continuous waveforms — `SineWave`, `SquareWave`, `TriangleWave` |
| `pulse` | Gaussian, rectangular, tone-burst, and Ricker pulses |
| `frequency_sweep` | Linear, logarithmic, and hyperbolic chirps |
| `modulation` | AM, FM, PM, QAM, PWM |
| `window` | `SignalWindowType` and window evaluation |
| `filter` | `Filter`, `FrequencyFilter` |
| `amplitude` / `frequency` / `phase` | Composable constant and time-varying components |
| `analytic` | Analytic-signal (Hilbert) utilities |
| `special` | `NullSignal`, `TimeVaryingSignal` |

A waveform is composed from independent amplitude, frequency, and phase components, so a
time-varying envelope or a chirped carrier is a component swap rather than a new waveform
type.

## Example

```rust
use kwavers_signal::{Signal, SineWave};

// 1 MHz carrier, 1 MPa peak, zero phase.
let carrier = SineWave::new(1.0e6, 1.0e6, 0.0);

assert_eq!(carrier.amplitude(0.0), 0.0);
assert_eq!(carrier.frequency(0.0), 1.0e6);

// A quarter period into the cycle the sine is at its peak.
let quarter_period = 0.25 / 1.0e6;
assert!((carrier.amplitude(quarter_period) - 1.0e6).abs() < 1.0);
```

## Documentation

- API reference: <https://docs.rs/kwavers-signal>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
