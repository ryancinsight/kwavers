<a id="kw-img-054"></a>

## KW-IMG-054 - Transmit-time models (TxModel) for the corrected metrics' DAS [minor] - todo

priority: research; needs: none; scope: kwavers-analysis triplet-pulse metrics DAS (KW-IMG-047/049); milestone: M-imaging-capability

- **Problem (Build 16, `B16_DASFIX.md`):** the corrected Kuji/Iwasaki metrics beamform with a transmit distance. For plane-wave transmit d_tx = pixel depth is correct, but focused and diverging transmits need their own legs, derived from the transmit delay laws:
  - foc: d_tx = d_max + s*|P - F|, d_max = max_e |elem_e - F|, s = +1 deeper than focus F, -1 shallower.
  - div: d_tx = |P - V| - D, V = virtual source, D = its depth behind the array plane.
  Omitting the leg misplaces windows by ~2000 samples (foc) / ~1400 (div) at dt 20 ns and silently zeroes or misreads the ROI. (The Python metric DAS omitted the array-to-reference leg entirely; the replica harness carried the same bug in both halves - fixed there, plane smoke bit-identical.)
- **Python reference:** `b16/dasfix.py` (tx_kind / tx_distance), verified against RF probe-row envelope peaks to <=0.39 us and a DAS A-line sweep; plane control bit-exact vs the B15 checkpoint.
- **Ask:** add a TxModel enum (Plane / Focused{focus, d_max} / Diverging{vsource, D}) to the DAS used by `triplet_pulse_metrics.rs` (KW-IMG-047/049), defaulting to Plane (parity preserved), with unit tests reproducing the B16_DASFIX arrival times (foc 43.26 us at 65 mm; div 43.10 us) from synthetic delay laws.
