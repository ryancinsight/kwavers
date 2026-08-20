# kwavers-phantom

Clinical tissue-phantom builders for [kwavers](https://github.com/ryancinsight/kwavers):
blood-oxygenation, layered tissue, tumor, and vascular-network presets.

Validating a photoacoustic or optical reconstruction needs a ground truth. This crate
constructs that ground truth — realistic tissue phantoms with known optical and acoustic
properties — so an algorithm can be measured against a known answer rather than against
another algorithm.

## What it provides

| Item | Responsibility |
|---|---|
| `PhantomBuilder` | Entry point; dispatches to the per-phantom builders below |
| `BloodOxygenationPhantomBuilder` | Oxygenation-contrast phantoms for sO₂ estimation |
| `LayeredTissuePhantomBuilder` | Epidermis / dermis / fat / muscle stacks |
| `TumorDetectionPhantomBuilder` | Background tissue with an embedded lesion |
| `VascularPhantomBuilder` | Arteries, veins, and tumor vasculature |
| `ClinicalPhantoms` | Named presets over the builders (skin, breast tumor, vascular network) |
| `SheppLogan` | The Shepp–Logan reference phantom and its variants |
| `ScattererCloud` / `PointScatterer` | Discrete scatterer fields for RF speckle synthesis |
| `PhantomTissueType`, `LayerSpec`, `TumorSpec`, `VesselSpec` | Phantom composition types |

Wavelength-dependent optical coefficients come from `hyperion`, the stack's reference
spectra crate; phantoms compose those coefficients into a spatial `OpticalPropertyMap`
rather than re-tabulating them.

## Example

```rust
use kwavers_grid::GridDimensions;
use kwavers_phantom::ClinicalPhantoms;

let dims = GridDimensions {
    nx: 32, ny: 32, nz: 32,
    dx: 1e-4, dy: 1e-4, dz: 1e-4,
};

// Four-layer skin model (1 mm epidermis, 2 mm dermis, 7 mm fat, muscle below) at 800 nm.
let skin = ClinicalPhantoms::skin_tissue(dims);

// Every voxel carries validated absorption and reduced-scattering coefficients.
let epidermis = skin.get_properties(16, 16, 2).unwrap();
let muscle = skin.get_properties(16, 16, 30).unwrap();
assert!(epidermis.absorption_coefficient() > 0.0);
assert!(muscle.absorption_coefficient() > 0.0);
```

## Documentation

- API reference: <https://docs.rs/kwavers-phantom>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
