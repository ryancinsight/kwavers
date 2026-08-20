# Example: Heterogeneous Power-Law Attenuation

**Crate**: `kwavers`

**Run**: `cargo run -p kwavers --release --example heterogeneous_power_law_attenuation`

**Source**: [`crates/kwavers/examples/heterogeneous_power_law_attenuation.rs`](../../../crates/kwavers/examples/heterogeneous_power_law_attenuation.rs)

## What This Example Demonstrates

Soft-tissue attenuation is commonly represented by

\[
\alpha_k(f) = \alpha_{0,k}\left(\frac{f}{f_\mathrm{ref}}\right)^{\gamma_k},
\]

where the coefficient \(\alpha_{0,k}\) and exponent \(\gamma_k\) belong to
material region \(k\). A single exponent cannot represent a path that crosses
tissues with different exponents. The example therefore exercises
`ViscoacousticMemorySolver::from_power_law_fields` with both fields varying by
voxel.

The parameter envelope follows the heterogeneous attenuation study and
[Fullwave 2.5](https://github.com/pinton-lab/fullwave25):

| Quantity | Values |
|---|---|
| \(\alpha_0\) | 0.25, 0.50, 0.75 dB·cm⁻¹·MHz⁻ᵞ |
| \(\gamma\) | 0.4, 0.7, 1.0, 1.3, 1.6 |
| Measurement frequencies | 0.6–4.6 MHz at eight fixed points |
| Relaxation memory fields | 3 per voxel |
| Heterogeneous path | alternating fat and muscle layers between two sensors |

The reference paper validates spatially heterogeneous power laws over a wider
clinical range and uses multiple relaxation mechanisms to retain local
time-domain updates ([Sode and Pinton, 2026](https://arxiv.org/abs/2606.11103)).
This Kwavers example reproduces the published coefficient/exponent envelope;
it does not compare against cached Fullwave output.

## Independent Measurement Oracle

The example measures the propagated field instead of reading the relaxation
fit. For each frequency, it normalizes the lossy near/far sensor ratio by an
otherwise identical lossless run:

\[
\widehat{\alpha}(f) =
-\frac{1}{d}\ln\left[
\frac{P_\mathrm{far}(f)/P_\mathrm{near}(f)}
     {P^0_\mathrm{far}(f)/P^0_\mathrm{near}(f)}
\right].
\]

The double ratio removes the source spectrum, acquisition gate, and lossless
solver response. Only absorption accumulated over the sensor separation
\(d\) remains.

The heterogeneous oracle is the exact plane-wave path integral:

\[
\alpha_\mathrm{path}(f) =
\frac{1}{d}\sum_k \alpha_{0,k}
\left(\frac{f}{f_\mathrm{ref}}\right)^{\gamma_k} L_k.
\]

Because the layer exponents differ, this sum cannot be replaced by one uniform
power law without changing its frequency dependence.

## Measured Result

The checked release run produces 120 homogeneous sweep rows and 8 layered-path
rows:

| Contract | Worst relative error |
|---|---:|
| Full 0.6–4.6 MHz envelope | 0.034137 |
| Band interior, 1.2–3.8 MHz | 0.004532 |
| Fat/muscle path integral | 0.009939 |

The edge error concentrates where the broadband source has the least spectral
energy. The source uses a rectangular direct-arrival gate deliberately: a
taper weights the dispersively broadened far pulse differently from the near
pulse and biases the recovered attenuation.

## Generated Artifacts

The run writes inspectable, gitignored output under
`target/fullwave_attenuation/`:

- `attenuation_sweep.csv`: prescribed and measured homogeneous values;
- `layered_medium.csv`: exact path-weighted and measured layered values;
- `attenuation_sweep.png`: log-log curves and measured markers with physical
  units.

The output directory keeps experimental results out of source control while
the example remains the single source that regenerates them.

## Book Chapter

[← Media and Tissue Models](../media_and_tissue_models.md)
