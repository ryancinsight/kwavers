<a id="kw-img-048"></a>

## KW-IMG-048 — Binned-distribution cloud sampling for the triplet cloud path (formation → imaging coupling) [minor] — done (Nov 21, 2026)

priority: feature; needs: none; scope: `kwavers-simulation` imaging (`imaging/triplet_pulse/binned_cloud.rs`); milestone: M-imaging-capability

- **Outcome:** a `TripletPulseCloud` can be populated by sampling bubble positions/radii from a binned size distribution in the formation model's 80-bin grid format (`b3_clouds.json`: bin-centre radii + normalised counts), so formation-model clouds feed the existing triplet cloud / Keller–Miksis path end to end in Rust.
- **Driver:** Build 14 B3 closed the formation → imaging coupling in Python and found the coupled cloud's contrast indistinguishable from a monodisperse cloud at its own median radius (|Δ| ≤ 0.5 dB at every PNP; N-shape invariant over N = 50 / 250 / 1000). The Rust cloud path previously accepted only prescribed monodisperse/added bubbles.
- **Acceptance:** size-stratified largest-remainder sampling matching B3's scheme (bin fractions preserved to within one bubble at any N, counts deterministic with no RNG), radii assigned to disk positions in seeded shuffled order (no spatial size segregation in the source model), deterministic under a fixed seed, distribution moments (mean, count median) preserved.

### Resolution (Nov 21, 2026)

- **Implementation.** `crates/kwavers-simulation/src/imaging/triplet_pulse/binned_cloud.rs`: `BinnedSizeDistribution` (with `from_um` for the `b3_clouds.json` representation, `fractions`, `mean_radius_m`, `median_radius_m`), `largest_remainder_counts`, `sample_binned_cloud` (local SplitMix64 stream; no new crate dependency), and `TripletPulseCloud::add_binned_cloud`.
- **Tests.** Largest-remainder totals and per-bin shares at N = 50 / 250 / 1000 on the B3 Kuji C = 0.2 distribution (89.4 % @ 2.772 µm, 9.9 % @ 2.380 µm, 0.7 % @ 2.043 µm); determinism under a fixed seed and seed-independence of the radius multiset; sampled mean within tolerance of the distribution mean; sampled positions inside the cloud disk; cloud constructor populates the triplet cloud; invalid distributions rejected.
