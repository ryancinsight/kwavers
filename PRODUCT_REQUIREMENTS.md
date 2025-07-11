# Product Requirements Document: kwavers - Ultrasound Simulation Toolbox

## 1. Introduction and Vision

**kwavers** is envisioned as a modern, high-performance, open-source computational toolbox for simulating ultrasound wave propagation and its interactions with complex biological media. It aims to provide researchers, engineers, and medical professionals with a powerful and flexible platform for modeling various ultrasound-based diagnostic and therapeutic applications.

The core vision is to offer capabilities comparable to or exceeding existing toolboxes like k-Wave, but with a focus on modern software engineering practices, performance leveraging contemporary hardware (CPUs, potentially GPUs in the future), and an idiomatic, extensible API primarily in Rust, while also considering future interoperability with other languages like Python.

## 2. Goals

*   **Accuracy:** Provide physically accurate simulations of wave phenomena.
*   **Performance:** Achieve high computational speed suitable for large-scale 3D simulations.
*   **Modularity & Extensibility:** Design a modular architecture that allows easy addition of new physical models, material properties, source types, and algorithms.
*   **Usability:** Offer a clear and well-documented API for setting up, running, and analyzing simulations.
*   **Feature Richness:** Support a comprehensive set of physical phenomena relevant to medical ultrasound, including:
    *   Nonlinear wave propagation.
    *   Linear elastic wave propagation (including shear waves).
    *   Heterogeneous and attenuating media (based on tissue properties).
    *   Thermal effects (heating due to absorption).
    *   Cavitation dynamics and sonoluminescence.
    *   Complex transducer modeling.
*   **Open Source:** Foster a collaborative community for development and validation.

## 3. Target Audience

*   **Academic Researchers:** In medical physics, biomedical engineering, acoustics, and related fields.
*   **Medical Device Engineers:** Developing and optimizing ultrasound equipment and therapies.
*   **Clinical Scientists:** Investigating novel ultrasound applications.
*   **Students:** Learning about ultrasound physics and computational modeling.

## 4. Key Feature Areas

### 4.1. Wave Solvers
*   **Acoustic Wave Propagation:**
    *   Linear acoustics.
    *   Nonlinear acoustics (e.g., Westervelt equation, KZK equation, first-order k-space pseudospectral).
    *   Support for k-space and time-domain methods.
*   **Elastic Wave Propagation (New - Initial Implementation Complete):**
    *   Linear isotropic elastic wave model (velocity-stress formulation).
    *   Support for compressional (P) and shear (S) waves.
    *   *Future:* Anisotropic media, nonlinear elasticity.
*   **Viscoelastic Models:**
    *   Models incorporating frequency-dependent absorption and dispersion based on viscoelastic material properties.

### 4.2. Medium Definition
*   **Homogeneous Media:** Uniform material properties.
*   **Heterogeneous Media:**
    *   Spatially varying properties defined by maps or functions.
    *   Pre-defined tissue properties library (acoustic, thermal, elastic).
    *   Ability to define custom materials.
*   **Attenuation Models:**
    *   Power-law frequency-dependent absorption.
    *   Viscous absorption.
    *   Relaxation-based absorption (future).

### 4.3. Acoustic Sources
*   **Transducer Geometries:**
    *   Piston sources (circular, rectangular).
    *   Linear arrays.
    *   Matrix arrays.
    *   Curved arrays (future).
    *   Intravascular/catheter-based sources (future).
*   **Source Characteristics:**
    *   Focusing (geometric, phased array steering).
    *   Apodization.
    *   Arbitrary time-varying excitation signals (sine, pulse, chirp, custom).
*   **Source Types:**
    *   Pressure sources.
    *   Velocity sources.
    *   Force/Stress sources (for elastic models).

### 4.4. Sensors & Recording
*   **Sensor Types:** Pressure, particle velocity components, stress tensor components, temperature, light intensity.
*   **Sensor Geometries:** Point sensors, lines, planes, full domain.
*   **Data Recording:**
    *   Time series data at sensor locations.
    *   Spatial field snapshots at specified time intervals.
    *   Frequency domain data (future).
*   **Output Formats:** CSV, potentially HDF5 or other standard scientific formats.

### 4.5. Boundary Conditions
*   **Perfectly Matched Layers (PMLs):**
    *   For acoustic waves.
    *   For light diffusion.
    *   For elastic waves (P & S waves - future, current is placeholder).
*   Pressure release / rigid boundaries (future).
*   Symmetry conditions (future).

### 4.6. Multi-Physics Modeling
*   **Thermal Modeling:**
    *   Bioheat equation (Pennes').
    *   Heat diffusion, perfusion, metabolic heat generation.
    *   Acoustic heat deposition.
*   **Cavitation Modeling:**
    *   Bubble dynamics (e.g., Rayleigh-Plesset, Gilmore, Keller-Miksis).
    *   Bubble cloud effects.
    *   Acoustic emissions from cavitation.
*   **Sonoluminescence & Sonochemistry:**
    *   Light emission modeling from collapsing bubbles.
    *   Basic chemical reaction kinetics influenced by cavitation/temperature (future).
*   **Acoustic Streaming:** Modeling fluid flow induced by acoustic waves.

### 4.7. Performance & Usability
*   **Parallelization:** Leverage multi-core CPUs (e.g., via Rayon).
*   **GPU Acceleration:** Future consideration.
*   **API:**
    *   Primary Rust API: Ergonomic, well-documented, type-safe.
    *   Configuration via files (e.g., TOML).
    *   Python bindings (future, for broader accessibility).
*   **Visualization:** Basic plotting utilities, interoperability with common plotting libraries (e.g., Python).
*   **Validation:** Rigorous testing against analytical solutions, benchmarks, and other established toolboxes.

## 5. Current State (as of this PRD version)

*   Solid foundation for acoustic wave simulation (nonlinear, k-space).
*   Initial implementation of linear isotropic elastic wave propagation.
*   Heterogeneous medium support with a basic tissue library.
*   PMLs for acoustic waves. Placeholder PML application for elastic waves.
*   Basic transducer types (linear array).
*   Core solver infrastructure with support for multiple physics modules.
*   Models for cavitation, thermal effects, light diffusion, acoustic streaming, and basic chemical effects are present.
*   Performance optimizations using `ndarray` and `rayon`.

## 6. Future Considerations / Potential Enhancements (Post current cycle)

*   **Advanced Elastic Models:** Anisotropy, nonlinear elasticity, full elastic PMLs.
*   **GPU Acceleration:** Explore GPU porting for significant speedups.
*   **Python API:** Greatly expand user base and ease of scripting.
*   **GUI:** For simplified simulation setup and visualization (long-term).
*   **Comprehensive Material Library:** Expand tissue and material properties, including frequency-dependent data.
*   **Advanced Transducer Modeling:** More complex geometries, beamforming algorithms.
*   **Inverse Problems & Optimization:** E.g., for transducer design or material characterization (long-term).
*   **Fluid-Structure Interaction:** For modeling waves in vessels, etc. (advanced).

## 7. Non-Goals (for initial phases)

*   Full electromagnetic wave simulation (focus is on acoustics/ultrasound).
*   General-purpose CFD solver (though acoustic streaming is included).
*   Real-time simulation for interactive applications (performance goal is for offline, detailed simulations).

This PRD provides a high-level overview and will be a living document, updated as the project evolves.
