# ADR-001 — 150 V 16-channel US neuromodulation driver, toaster-fit, size+availability optimized

Status: **Proposed** (Foundation phase). Supersedes all prior holohv designs (deleted).

## Source of truth
Javid, Biswas, Ilham, Kiani, *"High-Voltage Phased Array Electronics for Ultrasound
Neuromodulation,"* IEEE MWSCAS 2024 (`docs/10560355.pdf`), Fig. 2.

## Functional spec (fixed by the article)
- **16 channels**, modular / extendable to more by tiling.
- **Up to 150 V** bipolar pulses; **5 ns** delay resolution; 1–2 MHz f_s.
- Half-bridge class-D switching amplifier per channel into a piezo (BVD: 50 pF ∥ (54 Ω + 12 pF +
  0.49 mH)), 56 Ω series R per element, optional ferrite bead.
- **3 supplies:** 3.3 V (isolator logic side), 5 V (gate drive / isolator output side),
  adjustable **VDDH ≤ 150 V** (HV rail, external).
- Galvanic isolation between LV control (FPGA) and HV driver.
- FPGA (XC7A200T) is **off-board** (PC → FPGA → HV driver), per Fig. 1.

## Optimization objective (this project)
Minimize **total volume** of the HV-driver electronics so the stack fits a toaster-sized
compartment, subject to:
1. **Integration first** — merge the article's discrete chain (8× TC8220 power transistors + 8×
   MD1822 gate drivers + D1–D4 clamps per 16 ch) into integrated pulser ICs wherever a single
   part performs the merged function. This is the dominant area lever.
2. **DigiKey availability is a hard filter** — every selected MPN must be orderable on DigiKey
   (in stock or sane lead time). Verified per part before it enters the BOM.
3. Manufacturing best practices (DFM) and **150 V creepage** (HV→LV ≥ 0.5 mm surface) enforced by
   the layout engine, not post-hoc.
4. Thermal: power devices are the hot spot — reserve bottom-side heatsink area + copper pour.

### Compartment budget (ASSUMPTION — confirm)
"Toaster-sized" taken as a stacked volume of **~250 × 130 × 100 mm** (W×D×H): a 250 × 130 mm
per-board footprint (the prior toaster footprint) × ~100 mm stack height. The design is modular,
so stack height trades against board count. **To be confirmed / refined by the user.**

## Engine
`kicad-routing` (Rust) performs the layout optimization:
- **Placement** (to build): physics-guided SA fitting components into the board rectangle under
  creepage + thermal + courtyard-overlap energy → minimizes area.
- **Routing** (built): negotiated-congestion PathFinder with the physics cost field.
- **IO** (to build): `.kicad_pcb` emit so the optimized board opens in KiCad and DRCs.

## Decisions
- **D1 — RESOLVED: HV7355K6-G** (Microchip), 8-ch 0–150 V unipolar 1.5 A pulser, 56-VQFN 8×8 mm.
  Merges power MOSFETs + gate drivers + level translators + floating gate regulator per 8 ch →
  **2 chips = 16 ch** (vs the article's 16 ICs + diodes). 2.5–3.3 V CMOS control, direct from
  isolator. DigiKey verified 2026-06-20: **244 in stock, Part Status Active**, $16.30/1
  ($12.36/100); 42-wk replenishment lead is irrelevant at prototype volume. Datasheet 150 V /
  "up to 150 V swings" confirms the article's spec. Availability filter PASS.

## Open decisions (tracked in backlog)
- D2: Isolation — quad digital isolator (ISO7740) count vs serialized control to cut isolation
  channels.
- D3: Power tree — buck choice for 3.3/5 V; negative gate-supply need depends on D1.
- D4: Control interface to off-board FPGA (parallel timing vs SPI config + parallel fire).
