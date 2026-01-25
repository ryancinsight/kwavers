# Ultrasound & Optics Simulation Repository Analysis
## Comprehensive Benchmark Study for Kwavers Enhancement

**Date**: 2026-01-25  
**Prepared By**: Research Analysis Team  
**Document Version**: 1.0  
**Purpose**: Identify best practices, features, and architectural patterns from leading ultrasound simulation repositories

---

## Executive Summary

This document analyzes 12 leading ultrasound and optics simulation repositories to identify best practices, architectural patterns, and missing capabilities for kwavers. The analysis reveals that kwavers has a strong foundation in layered architecture and physics modeling, but could benefit from several enhancements in the areas of:

1. **Differentiation & GPU Acceleration**: JAX-style automatic differentiation for inverse problems
2. **k-space Pseudospectral Methods**: Industry-standard k-Wave algorithms for dispersion reduction
3. **Clinical Workflows**: MRI-guided planning, transcranial correction, thermal safety
4. **Optimization Frameworks**: BEM solvers, heterogeneous optimization, sound speed estimation
5. **Machine Learning Integration**: Differentiable beamforming, autofocusing, neural reconstruction

**Key Finding**: Kwavers' layered architecture (319 directories, 1210 Rust files) provides excellent separation of concerns, but the ecosystem demonstrates that successful ultrasound platforms require:
- Tight integration between forward/inverse solvers
- Differentiability throughout the pipeline
- Clinical safety validation frameworks
- Multi-domain coupling (acoustic-thermal-elastic)

---

## Table of Contents

1. [Repository Comparison Matrix](#1-repository-comparison-matrix)
2. [Detailed Repository Analysis](#2-detailed-repository-analysis)
3. [Architectural Patterns](#3-architectural-patterns)
4. [Key Features Comparison](#4-key-features-comparison)
5. [Missing Capabilities in Kwavers](#5-missing-capabilities-in-kwavers)
6. [Recommendations](#6-recommendations)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [References](#8-references)

---

## 1. Repository Comparison Matrix

| Repository | Language | Solver Type | Key Strength | Architecture Pattern | Validation Approach |
|------------|----------|-------------|--------------|---------------------|---------------------|
| **jwave** | Python/JAX | k-space PSTD | Differentiability, GPU | Modular blocks, ML-ready | Continuous integration, codecov |
| **k-wave** | MATLAB | k-space PSTD | Industry standard, validated | Functional, solver-centric | Peer-reviewed publications |
| **k-wave-python** | Python | k-space PSTD | Pythonic API, GPU | Class-based OOP | Google Colab examples |
| **optimus** | Python | BEM | Frequency-robust preconditioning | Multi-domain coupling | Academic publications |
| **fullwave25** | Python/CUDA | FDTD 8th/4th order | Heterogeneous attenuation | Python wrapper over CUDA | Attenuation validation |
| **Sound-Speed-Est** | MATLAB/Python | Coherence-based | SLSC optimization | Layered (beamform→coherence→opt) | Phantom & clinical data |
| **dbua** | Python/JAX | Differentiable DAS | Autofocus optimization | Modular (DAS→paths→losses) | MICCAI publication |
| **Kranion** | Java/GLSL | Ray tracing | Visualization, planning | Plugin architecture | Skull density validation |
| **mSOUND** | MATLAB | TMDM/FSMDM | Mixed-domain method | Documentation-centric | 6 peer-reviewed papers |
| **HITU_Simulator** | MATLAB | Westervelt+BHT | Thermal dose calculation | Functional modules | PDF manual |
| **BabelBrain** | Python/C++/CUDA | FDTD viscoelastic | Clinical integration | 3-step workflow | Experimental validation |
| **SimSonic** | C/GPL | FDTD elastodynamic | Fluid+solid coupling | Geometry-first design | GNU GPL, academic |
| **kwavers** | Rust | FDTD/PSTD/PINN | Multi-physics, type safety | Layered DDD | 1554 passing tests |

### Key Observations

1. **JAX Dominance**: Modern tools (jwave, dbua) leverage JAX for auto-differentiation and GPU acceleration
2. **k-space Pseudospectral**: Industry standard (k-Wave) uses k-space correction for reduced dispersion
3. **Clinical Focus**: BabelBrain/Kranion emphasize planning workflows, safety metrics, neuronavigation
4. **Hybrid Approaches**: fullwave25 balances Python usability with CUDA performance
5. **Specialization**: Each tool has unique niche (optimus=BEM, dbua=autofocus, mSOUND=mixed-domain)

---

## 2. Detailed Repository Analysis

### 2.1 jwave (JAX-Based Wave Simulation)

**Repository**: https://github.com/ucl-bug/jwave  
**Primary Citation**: Stanziola et al. (2022) arXiv:2207.01499

#### Key Features
- **Differentiable Acoustic Simulators**: Full automatic differentiation support via JAX
- **JIT Compilation**: Efficient execution through just-in-time compilation
- **Hardware Flexibility**: Seamless GPU/TPU deployment
- **Photoacoustic Focus**: Initial value problems for PA acquisitions
- **Fourier-Domain Representation**: Efficient frequency-domain field handling

#### Architectural Patterns
```
Geometry Layer:     Domain, Medium, TimeAxis
Physics Layer:      jwave.acoustics (time-varying solvers)
Representation:     FourierSeries (frequency domain)
Utilities:          Image loading, preprocessing
```

**Design Philosophy**: Modular blocks for ML pipeline integration

#### Unique Algorithms
- Fourier-domain wave propagation
- Finite difference with automatic differentiation
- CFL-aware timestep calculation: `TimeAxis.from_medium()`
- Parallelizable distributed computing architecture

#### Best Practices Observed
1. **Testing Infrastructure**: Mandatory codecov coverage for new features
2. **Documentation**: MkDocs with executable Jupyter notebooks
3. **Reproducibility**: Binder/Colab links for one-click reproduction
4. **Contribution Standards**: Formal CONTRIBUTING.md with changelog requirements
5. **Pre-commit Hooks**: Code quality enforcement via `.pre-commit-config.yaml`
6. **License**: LGPL-3.0 enables research and commercial use

#### Validation Approach
- Continuous integration with automated testing
- Codecov tracking for test coverage
- Comparison against k-Wave reference solutions
- Published peer-reviewed validation

#### Comparison to Kwavers
| Feature | jwave | kwavers | Gap Analysis |
|---------|-------|---------|--------------|
| Differentiability | ✅ Full (JAX) | ⚠️ Partial (PINN only) | **HIGH**: JAX-like auto-diff needed |
| GPU Support | ✅ JAX native | ✅ WGPU optional | kwavers has GPU, lacks auto-diff integration |
| k-space Method | ✅ Yes | ⚠️ PSTD only | **MEDIUM**: k-space correction missing |
| ML Integration | ✅ Native | ⚠️ Burn PINN | **MEDIUM**: Need tighter ML coupling |
| Documentation | ✅ Excellent | ✅ Good | Similar quality |

**Recommendation**: Implement JAX-inspired differentiable layers for inverse problems and parameter optimization.

---

### 2.2 k-Wave (MATLAB Ultrasound Toolbox)

**Repository**: https://github.com/ucl-bug/k-wave  
**Primary Citations**: 
- Treeby & Cox (2010) J. Biomed. Opt. 15(2), 021314
- Treeby et al. (2012) J. Acoust. Soc. Am. 131(6), 4324-4336

#### Key Features
- **Time-Domain Acoustic Simulation**: 1D/2D/3D with linear/nonlinear propagation
- **k-space Pseudospectral Method**: Fourier collocation for spatial gradients
- **Power Law Absorption**: Fractional Laplacian operators for realistic tissue
- **Split-Field PML**: Perfectly matched layer boundary absorption
- **Heterogeneous Media**: Spatially-varying material parameters
- **Reduced Computational Cost**: More efficient than conventional FDTD

#### Architectural Patterns
```
Core Solver:        kspaceFirstOrder family (1D/2D/3D/AS)
Domain Variants:    Cartesian, axisymmetric coordinates
Physics Modes:      Acoustic (linear/nonlinear), elastic, photoacoustic
Boundary Handling:  PML with customizable profiles
Pre/Post:           Grid generation, sensor simulation, reconstruction
```

**Design Philosophy**: Mathematical exactness in homogeneous limit, practical accuracy in heterogeneous media

#### Unique Algorithms
1. **k-space Correction**: Reduces numerical dispersion to near-zero
2. **Fractional Laplacian**: Models frequency-dependent absorption via integro-differential operators
3. **Axisymmetric Solver**: Exploits rotational symmetry for 3D problems with 2D cost
4. **kWaveArray Class**: Arbitrary acoustic source distributions

#### Best Practices Observed
1. **Versioning**: Clear versioning strategy (1.4.x final feature release, 2.x migration path)
2. **Installation Flexibility**: MATLAB Add-On, direct download, programmatic path
3. **Documentation Integration**: Embedded help accessible through MATLAB help browser
4. **Performance Paths**: Pure MATLAB + optional C++/CUDA binaries
5. **Community Support**: Designated forum for questions, GitHub for bugs
6. **Open Licensing**: LGPL v3 with clear derivative work requirements

#### Validation Approach
- Extensive peer-reviewed publications across multiple domains
- Reproducible examples paired with published results
- Comparison against analytical solutions in homogeneous limits
- Experimental validation for nonlinear propagation
- Community-driven validation through open forum

#### Mathematical Foundation
The k-space pseudospectral approach achieves exactness "in the limit of linear wave propagation in a homogeneous and lossless medium" by using:
- Fourier transforms for spatial derivatives (spectral accuracy)
- k-space corrected finite differences for time stepping
- Fractional calculus for power-law absorption

#### Comparison to Kwavers
| Feature | k-Wave | kwavers | Gap Analysis |
|---------|--------|---------|--------------|
| k-space PSTD | ✅ Full implementation | ❌ Missing | **CRITICAL**: Industry-standard method absent |
| Power-law absorption | ✅ Fractional Laplacian | ⚠️ Basic absorption | **HIGH**: Frequency-dependent absorption needed |
| Axisymmetric solver | ✅ Yes | ❌ Missing | **MEDIUM**: 10-100× speedup for symmetric problems |
| PML boundaries | ✅ Split-field | ✅ CPML | Similar capability |
| Arbitrary sources | ✅ kWaveArray | ⚠️ Basic sources | **MEDIUM**: Need flexible source modeling |
| Validation depth | ✅ Extensive publications | ✅ 1554 tests | kwavers has good testing, needs publication |

**Critical Gap**: k-space pseudospectral method is the industry standard. Kwavers' PSTD lacks k-space correction.

**Recommendation**: Implement k-space corrected pseudospectral method as primary solver variant.

---

### 2.3 k-Wave-python (Python k-Wave Port)

**Repository**: https://k-wave-python.readthedocs.io/en/latest/  
**Version**: Based on k-Wave MATLAB 1.4.0

#### Key Features
- **GPU Acceleration**: NVIDIA GPUs (Maxwell through Hopper architectures)
- **k-space Pseudospectral**: Ported from MATLAB k-Wave
- **Pythonic API**: Diverges from MATLAB to leverage Python practices
- **Medical Imaging Focus**: Algorithmic prototyping and testing

#### API Design
```python
# Core classes for simulation components
kWaveGrid       # Computational domain
kWaveMedium     # Material properties
kSource         # Source definitions
kSensor         # Sensor/receiver specifications
kWaveSimulation # Simulation orchestration

# Solvers by dimensionality
kspaceFirstOrder2D   # 2D solver
kspaceFirstOrder3D   # 3D solver
kspaceFirstOrderAS   # Axisymmetric solver

# Utility modules
reconstruction   # Beamforming, image formation
data processing  # Filtering, signal processing
```

#### Architectural Patterns
- **Unified Configuration**: Shared medium properties, source/sensor specs
- **Dimensionality Variants**: Solvers organized by 2D/3D/axisymmetric
- **GPU-First Design**: GPU acceleration as primary mode, not optional

#### Best Practices Observed
1. **Tutorial Structure**: "Four components of every simulation" teaching approach
2. **Cloud Integration**: All examples runnable in Google Colab
3. **B-mode Examples**: Clinical reconstruction workflows demonstrated
4. **Step-by-Step Guides**: Progressive complexity in documentation

#### Comparison to Kwavers
| Feature | k-wave-python | kwavers | Gap Analysis |
|---------|---------------|---------|--------------|
| Python bindings | ✅ Native | ❌ Rust only | **MEDIUM**: Consider PyO3 bindings |
| GPU primary | ✅ NVIDIA focus | ⚠️ WGPU optional | **LOW**: WGPU more portable |
| Cloud examples | ✅ Colab notebooks | ❌ Missing | **MEDIUM**: Need reproducible examples |
| Clinical workflows | ✅ B-mode demos | ⚠️ Basic imaging | **MEDIUM**: Need reconstruction examples |

**Recommendation**: Create Python bindings (PyO3) and Jupyter notebook examples for accessibility.

---

### 2.4 OptimUS (Optimization Library)

**Repository**: https://github.com/optimuslib/optimus  
**Focus**: 3D acoustic wave propagation in unbounded domains

#### Key Features
- **Boundary Element Method (BEM)**: Helmholtz equation solving
- **Multiple Scatterers**: Complex geometries with multiple domains
- **Frequency-Robust Preconditioning**: Advanced solver acceleration
- **Tissue Property Integration**: IT'IS Foundation database
- **Focused Ultrasound Applications**: Biomedical engineering specialization

#### Architectural Patterns
```
Core Package:      optimus/
Examples:          notebooks/
Documentation:     docs/
Computational:     bempp-legacy (backend dependency)
```

**Design Philosophy**: Leverage established BEM libraries rather than reimplementing core solvers

#### Unique Algorithms
- **BEM Formulations**: Boundary integral equations for acoustic transmission
- **Frequency-Robust Preconditioning**: Solvers stable across wide frequency ranges
- **Multi-Domain Coupling**: Heterogeneous materials through domain interfaces

#### Best Practices Observed
1. **Docker Containerization**: Reproducible computational environments
2. **Read the Docs**: Professional documentation hosting
3. **Citation Infrastructure**: CITATION.cff for academic credit
4. **Academic Rigor**: Peer-reviewed backing for implementations
5. **Community Engagement**: GitHub discussions and issue tracking

#### Comparison to Kwavers
| Feature | OptimUS | kwavers | Gap Analysis |
|---------|---------|---------|--------------|
| BEM Solver | ✅ Full implementation | ❌ Missing | **HIGH**: BEM needed for unbounded domains |
| Multi-domain | ✅ Yes | ⚠️ Basic heterogeneous | **MEDIUM**: Need interface coupling |
| Preconditioning | ✅ Advanced | ⚠️ Basic | **MEDIUM**: Iterative solver optimization |
| Tissue database | ✅ IT'IS integration | ❌ Missing | **HIGH**: Standardized tissue properties |
| Docker support | ✅ Yes | ❌ Missing | **LOW**: Container deployment |

**Critical Gap**: BEM solvers enable unbounded domain problems (radiation, scattering) that FDTD/PSTD cannot efficiently handle.

**Recommendation**: Implement hybrid BEM-FDTD coupling for focused ultrasound problems with unbounded regions.

---

### 2.5 fullwave25 (Full-Wave Ultrasound)

**Repository**: https://github.com/pinton-lab/fullwave25  
**Primary Citation**: Pinton (2021) arXiv:2106.11476

#### Key Features
- **Heterogeneous Attenuation**: Spatially-varying α₀ and γ in power-law model
- **High-Order FDTD**: 8th-order spatial, 4th-order temporal accuracy
- **Multi-GPU Support**: Domain decomposition in depth dimension
- **Python-CUDA Architecture**: Usability + performance balance
- **Nonlinear Wave Equation**: Multiple relaxation processes

#### Architectural Patterns
```
Core Library:        fullwave/
Examples:            examples/ (plane wave → 3D transducers)
Tests:               tests/ (validation suite)
Experiments:         experiments/ (research investigations)
```

**Design Philosophy**: Comparable usability to k-Wave with superior GPU performance

#### Unique Algorithms
1. **Multiple Relaxation Processes**: Frequency-dependent power-law attenuation in heterogeneous media
2. **Stretched-Coordinate Derivatives**: ∇₁ and ∇₂ operators control attenuation/dispersion
3. **Relaxing PML**: Integrated relaxation mechanics in boundary absorption
4. **Staggered-Grid 8th/4th Order**: High accuracy with manageable dispersion

#### Best Practices Observed
1. **Progressive Examples**: Simple → complex (plane wave → linear → convex → 3D)
2. **Git Workflow**: Enforced branch naming (TYPE/BRANCH_NAME)
3. **Makefile Automation**: `make test`, `make install` for developer convenience
4. **UV Package Manager**: Lock files for reproducible Python environments
5. **Theoretical Background**: README includes mathematical foundations

#### Validation Approach
- Published arXiv paper with detailed methodology
- Attenuation validation graphs (simulated vs. target power-law)
- Zenodo DOI (10.5281/zenodo.17625780) for reproducibility
- Multiple coefficient/exponent validation combinations

#### Comparison to Kwavers
| Feature | fullwave25 | kwavers | Gap Analysis |
|---------|------------|---------|--------------|
| High-order FDTD | ✅ 8th/4th | ⚠️ 2nd/4th | **MEDIUM**: Higher accuracy available |
| Multi-GPU | ✅ Domain decomp | ⚠️ Single GPU | **MEDIUM**: Scaling for large problems |
| Python+CUDA | ✅ Hybrid | ❌ Rust only | **LOW**: Rust has similar perf potential |
| Heterogeneous atten | ✅ Full α₀(x),γ(x) | ⚠️ Basic | **HIGH**: Realistic tissue modeling |
| Relaxation PML | ✅ Integrated | ⚠️ Standard CPML | **MEDIUM**: Better absorption |

**Key Insight**: fullwave25's heterogeneous attenuation (spatially-varying α₀ and γ) is critical for clinical accuracy.

**Recommendation**: Implement stretched-coordinate formulation with spatially-varying power-law attenuation.

---

### 2.6 Sound-Speed-Estimation

**Repository**: https://github.com/JiaxinZHANG97/Sound-Speed-Estimation  
**Primary Citation**: IEEE Trans. Ultrasonics (2025)

#### Key Features
- **SLSC Beamforming**: Short-lag spatial coherence for plane wave data
- **Optimization Approaches**: Maximization (coherent) / minimization (incoherent)
- **Multi-Target Support**: Point targets, lesions, speckle regions
- **Image Quality Metrics**: Contrast, CNR, gCNR, FWHM, speckle SNR

#### Architectural Patterns
```
MATLAB_code/     → Beamforming, coherence computation
cubdl/           → CUBDL framework data loading
scoring/         → Analysis metrics, ROI processing
submissions/     → IQ data export pipelines
```

**Data Flow**: CUBDL datasets → IQ extraction → RF conversion → Coherence → Optimization

#### Unique Algorithms
1. **mLOC Metric**: "Maximum lag-one coherence" from photoacoustic imaging
2. **Coherence Optimization**: Sound speed parameter space sweep
3. **ROI-based Analysis**: Region-specific rather than full-frame
4. **Histogram Filtering**: Preprocessing for incoherent targets

#### Best Practices Observed
1. **CUBDL Integration**: Standardized ultrasound dataset framework
2. **Dual-Language**: Python ingestion, MATLAB processing
3. **Validation Datasets**: Simulation, phantom, clinical breast imaging
4. **Published Metrics**: Quantitative image quality assessment

#### Comparison to Kwavers
| Feature | Sound-Speed-Est | kwavers | Gap Analysis |
|---------|-----------------|---------|--------------|
| Sound speed opt | ✅ Coherence-based | ❌ Missing | **HIGH**: Critical for image quality |
| CUBDL integration | ✅ Standard datasets | ❌ Missing | **MEDIUM**: Reproducibility framework |
| Image quality | ✅ Comprehensive metrics | ⚠️ Basic | **MEDIUM**: Need CNR, gCNR, FWHM |
| ROI analysis | ✅ Yes | ❌ Missing | **MEDIUM**: Clinical workflow component |

**Critical Gap**: Sound speed estimation is essential for aberration correction and image quality optimization.

**Recommendation**: Implement coherence-based sound speed estimation for adaptive beamforming (aligns with TODO_AUDIT P1 items).

---

### 2.7 dbua (Deep Learning Beamforming)

**Repository**: https://github.com/waltsims/dbua  
**Primary Citation**: MICCAI 2023 proceedings (Stanford University)

#### Key Features
- **Differentiable Delay-and-Sum**: Learnable time delay profiles
- **Phase-Error Loss Functions**: Autofocus optimization objective
- **Speed-of-Sound Mapping**: Heterogeneous media correction
- **JAX Auto-Differentiation**: GPU-accelerated gradient-based optimization
- **Video Output**: Optimization progress visualization

#### Architectural Patterns
```
dbua.py     → Experiment orchestration (config, workflow)
das.py      → Beamforming engine (delay-and-sum)
paths.py    → Time-of-flight calculations (spatial SoS)
losses.py   → Custom loss functions (phase-error, regularization)
```

**Design Philosophy**: Independent pipeline stages for testing/modification flexibility

#### Unique Algorithms
1. **Differentiable Beamforming**: Gradient-based acoustic focusing optimization
2. **Phase-Error Minimization**: Objective function for autofocus
3. **SoS Estimation Integration**: Joint optimization of focusing + medium properties
4. **GPU Processing**: Large-scale ultrasound IQ data

#### Best Practices Observed
1. **Modular Separation**: DAS → paths → losses can be tested independently
2. **MATLAB Data Compatibility**: .mat file ingestion for existing datasets
3. **Visualization Pipeline**: MP4 generation for optimization monitoring
4. **Hyperparameter Documentation**: Main script contains configuration specs

#### Validation Approach
- MICCAI peer-reviewed publication
- Multiple acoustic phantoms (two-layer, four-layer, inclusion)
- Reproducible with provided datasets and configurations
- Quantitative performance metrics documented

#### Comparison to Kwavers
| Feature | dbua | kwavers | Gap Analysis |
|---------|------|---------|--------------|
| Differentiable BF | ✅ Full JAX | ⚠️ PINN only | **HIGH**: Need end-to-end differentiability |
| Autofocus | ✅ Phase-error opt | ❌ Missing | **CRITICAL**: Essential for clinical imaging |
| SoS mapping | ✅ Heterogeneous | ❌ Missing | **HIGH**: Complements coherence methods |
| JAX integration | ✅ Native | ❌ Missing | **MEDIUM**: Consider Rust ML alternatives |

**Critical Gap**: Differentiable beamforming enables automatic aberration correction without manual tuning.

**Recommendation**: Implement differentiable beamforming layer in kwavers' analysis/signal_processing module (aligns with TODO_AUDIT neural beamforming items).

---

### 2.8 Kranion (Transcranial Ultrasound)

**Repository**: https://github.com/jws2f/Kranion  
**Focus**: Treatment planning for transcranial focused ultrasound

#### Key Features
- **Geometric Modeling**: Treatment plan visualization
- **Patient Selection**: Skull density ratio calculations
- **Phase Aberration Estimation**: Skull-induced distortions
- **Transducer Efficiency**: Treatment feasibility assessment
- **DICOM Integration**: MR and CT image import

#### Architectural Patterns
```
Plugins:
  - ACPCPlanPlugin (anatomical planning)
  - KranionGroovyConsolePlugin (scripting interface)
  - TractographyPlugin (tissue pathway visualization)

Libraries: /lib, /src/main/java
Graphics: GLSL shaders (8.6%) for GPU rendering
```

**Technology Stack**: Java (90.8%) desktop application with OpenGL acceleration

#### Best Practices Observed
1. **Plugin Architecture**: Extensibility through modular plugins
2. **Scripting Support**: Groovy console for automation
3. **Medical Imaging Standards**: DICOM compatibility
4. **Visualization First**: Interactive 3D rendering for clinical use

#### Comparison to Kwavers
| Feature | Kranion | kwavers | Gap Analysis |
|---------|---------|---------|--------------|
| Planning workflows | ✅ Full GUI | ❌ Library only | **LOW**: kwavers is library, not app |
| Skull modeling | ✅ Density metrics | ⚠️ Basic attenuation | **MEDIUM**: Need comprehensive skull physics |
| Phase aberration | ✅ Estimation | ⚠️ Partial (TODO) | **HIGH**: Critical for transcranial (TODO_AUDIT P1) |
| DICOM support | ✅ Yes | ✅ Yes | Similar capability |
| Visualization | ✅ Advanced 3D | ⚠️ Basic plotting | **LOW**: Different use cases |

**Key Insight**: Kranion focuses on clinical workflow, while kwavers is a simulation library. Complementary tools.

**Recommendation**: Extract skull physics algorithms (density ratio, aberration) for kwavers' physics/acoustics/skull module.

---

### 2.9 mSOUND (MATLAB Simulation)

**Repository**: https://github.com/m-SOUND/mSOUND  
**Focus**: Generalized Westervelt equation modeling

#### Key Features
- **TMDM**: Transient Mixed-Domain Method (time-domain, pulsed-wave, arbitrary nonlinearity)
- **FSMDM**: Frequency-Specific Mixed-Domain Method (steady-state, specific frequencies, weakly-nonlinear)
- **Heterogeneous Media**: Spatially-varying speed, density, attenuation
- **One-Way Approximation**: Optional reflection inclusion

#### Architectural Patterns
```
Documentation-Centric:
  - Jekyll static site (93.3% HTML)
  - Blog posts for examples
  - Research papers in /download
  - function.html for API docs
```

**Design Philosophy**: Publication and documentation over raw code distribution

#### Unique Algorithms
1. **Generalized Westervelt Equation**: Nonlinear wave propagation foundation
2. **Mixed-Domain Methods**: Hybrid time/frequency approaches
3. **TMDM vs FSMDM**: Algorithm selection based on problem characteristics

#### Validation Approach
Six peer-reviewed publications:
1. Modified mixed domain for heterogeneous media (JASA, 2020)
2. Second harmonic simulation (IEEE UFFC, 2019)
3. Weakly heterogeneous propagation (IEEE TUFFC, 2018)
4. Modeling review (IEEE TUFFC, 2015)
5. Absorption layer techniques (JASA, 2012)
6. Wave-vector-frequency-domain (JASA, 2011)

#### Comparison to Kwavers
| Feature | mSOUND | kwavers | Gap Analysis |
|---------|--------|---------|--------------|
| Westervelt eq | ✅ Generalized | ⚠️ Basic nonlinear | **MEDIUM**: More sophisticated nonlinearity |
| Mixed-domain | ✅ TMDM/FSMDM | ❌ Missing | **MEDIUM**: Frequency-specific optimization |
| Documentation | ✅ Exceptional | ✅ Good | mSOUND sets gold standard |
| Publication depth | ✅ 6 papers | ⚠️ In progress | **LOW**: kwavers needs publication |

**Key Insight**: Mixed-domain methods offer computational efficiency for frequency-specific problems.

**Recommendation**: Consider implementing FSMDM variant for steady-state harmonic imaging applications.

---

### 2.10 HITU_Simulator (HIFU Simulator)

**Repository**: https://github.com/jsoneson/HITU_Simulator  
**Focus**: High-Intensity Therapeutic Ultrasound (HIFU)

#### Key Features
- **Axisymmetric Beam Propagation**: WAKZK solvers (Westervelt equation)
- **Thermal Modeling**: Bioheat Transfer (BHT) equations
- **Thermal Dose Calculation**: Treatment safety assessment
- **Continuous Wave Support**: CW ultrasound modeling

#### Architectural Patterns
```
Acoustic Modules:
  - WAKZK.m (general solver)
  - WAKZK_Gaussian.m (Gaussian beams)
  - WAKZK_planar.m (planar sources)

Thermal Modules:
  - LinearHeating.m
  - BHT.m (Bioheat Transfer)
  - BuildBHTperipherals.m

Numerical Methods:
  - BuildPade11operators.m
  - BuildPade12operators.m (Pade approximations)
  - SourceFilterH.m (signal processing)

Scanning:
  - SynthAxScan.m (axial scanning)
  - SynthRadScan.m (radial scanning)
```

#### Unique Algorithms
1. **Westervelt Equation Variants**: WAKZK family for different beam geometries
2. **Pade Approximations**: Numerical propagation schemes
3. **Bioheat Transfer**: Thermal dose accumulation modeling

#### Best Practices Observed
1. **Comprehensive Manual**: HITU_Simulator_v2_manual.pdf documentation
2. **GPL-3.0 License**: Open academic collaboration
3. **100% MATLAB**: Single-language consistency

#### Comparison to Kwavers
| Feature | HITU_Simulator | kwavers | Gap Analysis |
|---------|----------------|---------|--------------|
| Westervelt | ✅ WAKZK variants | ⚠️ Basic | **MEDIUM**: Beam-specific optimizations |
| Thermal dose | ✅ Full calculation | ⚠️ Basic thermal | **HIGH**: Clinical safety critical |
| Pade methods | ✅ 11, 12 operators | ❌ Missing | **MEDIUM**: Alternative numerical schemes |
| Axisymmetric | ✅ Dedicated solver | ❌ Missing | **HIGH**: 10-100× speedup (matches k-Wave gap) |

**Critical Gap**: Thermal dose calculation is essential for HIFU therapy safety validation.

**Recommendation**: Implement Bioheat Transfer coupling with acoustic solver in kwavers' simulation/multiphysics module.

---

### 2.11 BabelBrain (MRI-Guided HIFU)

**Repository**: https://github.com/ProteusMRIgHIFU/BabelBrain  
**Primary Citation**: Pichardo (2023) IEEE TUFFC 70(7):587-599

#### Key Features
- **Transcranial Acoustic Modeling**: Skull distortion correction
- **Thermal Simulation**: Heating based on exposure/duty cycle/intensity
- **Multi-Modal Imaging**: CT, ZTE, PETRA for bone visualization
- **Neuronavigation Integration**: Brainsight, 3DSlicer compatibility
- **Clinical Safety**: MI, Isppa calculations
- **Multi-Device Support**: Single-element, phased arrays, dome arrays (up to 1024 elements)

#### Architectural Patterns
```
3-Step Clinical Workflow:
  Step 1: Domain generation (imaging → segmentation → mesh)
  Step 2: Acoustic simulation (transducer positioning)
  Step 3: Thermal modeling (safety metrics)

Computational Engine:
  BabelViscoFDTD (viscoelastic FDTD solver)
  GPU Backends: Metal, OpenCL, CUDA

Technology Stack:
  79.9% Python (high-level workflow)
  C++/C (GPU kernels)
```

#### Unique Algorithms
1. **Mechanical Adjustments**: X/Y corrections to align focal spots
2. **Tissue-Specific Modeling**: Cortical vs trabecular bone (configurable fraction)
3. **Air Region Masking**: Reflection handling near sinuses
4. **Elastix Registration**: CT-to-T1W alignment optimization
5. **CSG Operations**: Geometry manipulation (recently migrated pycork → trimesh)

#### Clinical Features
- **Grouped Sonications**: Complex thermal scenarios (v0.8.0)
- **Brain Tissue Differentiation**: Gray/white matter in simulations
- **Safety Metrics**: Mechanical Index (MI), spatial peak intensity (Isppa)
- **Treatment Planning**: Distance measurements, contact tracking
- **Trajectory Management**: Neuronavigation system corrections

#### Validation Approach
- **Experimental Validation**: Pichardo et al. (2017)
- **Cross-Validation**: Comparison with other numerical tools
- **Regression Testing**: pytest framework
- **Offline Batch Examples**: Rayleigh integral vs FDTD in water
- **Unit Testing**: Ongoing development validation

#### Best Practices Observed
1. **Clinical Workflow Architecture**: 3-step process mirrors clinical needs
2. **Multi-GPU Backend**: Platform-agnostic GPU acceleration
3. **Medical Imaging Standards**: GE scanner settings documentation (PDF, screenshots)
4. **Output Formats**: NIfTI, CSV for clinical software integration
5. **Safety First**: "Research purposes only" disclaimer with clinical-grade calculations
6. **Version Control**: Detailed changelogs (v0.8.0 features documented)

#### Comparison to Kwavers
| Feature | BabelBrain | kwavers | Gap Analysis |
|---------|------------|---------|--------------|
| Clinical workflow | ✅ 3-step GUI | ❌ Library only | **LOW**: Different use cases |
| Transcranial | ✅ Full pipeline | ⚠️ Basic (TODO) | **CRITICAL**: TODO_AUDIT P1 priority |
| Thermal safety | ✅ MI, Isppa, dose | ⚠️ Basic thermal | **HIGH**: Clinical deployment blocker |
| Neuronavigation | ✅ Brainsight, 3DSlicer | ❌ Missing | **MEDIUM**: Clinical integration |
| Multi-GPU | ✅ Metal/OpenCL/CUDA | ⚠️ WGPU only | **LOW**: WGPU cross-platform advantage |
| Medical imaging | ✅ CT, ZTE, PETRA | ✅ DICOM, NIfTI | Similar capability |
| Registration | ✅ Elastix | ❌ Missing | **HIGH**: TODO_AUDIT P1 (fUS registration) |

**Critical Gaps**:
1. **Transcranial Correction**: BabelBrain's mechanical adjustments + skull modeling (aligns with TODO_AUDIT P1 transcranial items)
2. **Thermal Safety**: MI, Isppa calculations for regulatory compliance
3. **Registration**: Elastix-style optimization for multi-modal alignment

**Recommendation**: Priority implementation of transcranial aberration correction and thermal safety metrics (directly addresses 4+ TODO_AUDIT P1 items).

---

### 2.12 SimSonic (Commercial/Academic Simulator)

**Website**: https://www.simsonic.fr (note: SSL certificate expired, accessed via search)  
**License**: GNU GPL

#### Key Features
- **FDTD Elastodynamic Equations**: Full elastic wave propagation
- **Fluid + Solid Media**: Coupled propagation across material interfaces
- **Anisotropic Materials**: Directional material properties
- **Heterogeneous Domains**: Spatially-varying properties
- **Custom Geometries**: User-defined computational domains
- **Array Transducers**: Multi-element source definitions
- **Unbounded Domains**: Boundary condition support

#### Applications
- Biomedical imaging
- Non-destructive testing
- Photoacoustic imaging
- General ultrasound propagation studies

#### Architectural Patterns
- **Geometry-First Design**: Users define geometries before physics
- **Material Property System**: Custom materials with anisotropy
- **Source Flexibility**: Arrays and custom source definitions

#### Limitations
- **No Dissipation**: Current versions don't model attenuation (per documentation)
- **2D Focus**: SimSonic 2D is primary distribution

#### Best Practices Observed
1. **Open Source**: GNU GPL for academic/research communities
2. **Educational Focus**: Designed for researchers, teachers, students
3. **C Source Availability**: Compiled programs + source code provided
4. **General Tool Philosophy**: Not specialized to single application

#### Comparison to Kwavers
| Feature | SimSonic | kwavers | Gap Analysis |
|---------|----------|---------|--------------|
| Elastodynamic | ✅ Full implementation | ⚠️ Partial elastic | **MEDIUM**: Full elastic waves |
| Fluid-solid coupling | ✅ Yes | ❌ Missing | **HIGH**: Interface physics |
| Anisotropic | ✅ Yes | ⚠️ Basic | **MEDIUM**: Directional materials |
| Dissipation | ❌ Missing | ✅ Yes | kwavers advantage |
| 3D support | ⚠️ Limited (2D focus) | ✅ Full 3D | kwavers advantage |

**Key Insight**: SimSonic's fluid-solid coupling is unique capability for biomedical/NDT applications.

**Recommendation**: Implement fluid-solid interface physics for bone-tissue boundary modeling.

---

## 3. Architectural Patterns

### 3.1 Common Architecture Patterns Across Repositories

| Pattern | Repositories | Description | Applicability to Kwavers |
|---------|--------------|-------------|--------------------------|
| **Layered Separation** | jwave, k-wave-python, kwavers | Physics → Solver → Analysis hierarchy | ✅ Already implemented |
| **Modular Blocks** | jwave, dbua, optimus | Independent components for ML integration | ⚠️ Partial - needs tighter ML coupling |
| **Solver-Centric** | k-wave, mSOUND, fullwave25 | Core solver with pre/post utilities | ✅ kwavers follows this pattern |
| **Workflow Pipeline** | BabelBrain, Kranion | Step-by-step clinical processes | ❌ Library vs application difference |
| **Plugin Architecture** | Kranion | Extensibility through plugins | ⚠️ Could enhance kwavers' modularity |
| **Hybrid Language** | fullwave25, BabelBrain | Python wrapper + compiled backend | ✅ Rust provides both (no wrapper needed) |
| **Documentation-First** | mSOUND, k-wave | Extensive docs + published papers | ✅ kwavers has good docs, needs publications |

### 3.2 Kwavers Architecture Analysis

**Current Structure** (from analysis):
```
Clinical Layer        → Research applications, safety compliance
Analysis Layer        → Signal processing, imaging algorithms
Simulation Layer      → Multi-physics orchestration
Solver Layer          → Numerical methods (FDTD, PSTD, PINN)
Physics Layer         → Mathematical specifications
Domain Layer          → Problem geometry, materials, sources
Math Layer            → Linear algebra, FFT, numerical primitives
Core Layer            → Fundamental types, error handling
```

**Statistics**:
- 319 directories
- 1210 Rust source files
- 1554 passing tests
- 114 TODOs (50 P1, 64 P2)

**Strengths**:
1. **Deep Vertical Hierarchy**: Excellent separation of concerns
2. **Type Safety**: Rust's ownership model prevents common errors
3. **Testing Coverage**: Comprehensive test suite
4. **Multi-Physics**: Acoustic-thermal-optical-cavitation coupling
5. **Clean Architecture**: DDD bounded contexts, unidirectional dependencies

**Weaknesses** (identified through comparison):
1. **Missing k-space Methods**: Industry-standard k-Wave algorithms absent
2. **Limited Differentiability**: PINN only, no JAX-style auto-diff throughout
3. **Incomplete Clinical Workflows**: Thermal safety, aberration correction partial
4. **No BEM Solver**: Cannot efficiently handle unbounded domains
5. **Publication Gap**: Strong code, needs peer-reviewed validation

### 3.3 Best-in-Class Architecture Recommendations

Based on the 12 repositories analyzed, kwavers should adopt:

1. **Differentiable Layers** (from jwave, dbua)
   ```rust
   // Proposed: Differentiable trait for inverse problems
   trait Differentiable {
       fn forward(&self, params: &Params) -> Field;
       fn backward(&self, grad_output: &Field) -> Params;
   }
   ```

2. **k-space Correction Module** (from k-wave)
   ```rust
   // Proposed: src/solver/forward/pstd/kspace_correction/
   mod kspace_pstd {
       // Fourier-domain spatial derivatives
       // k-space corrected time stepping
       // Fractional Laplacian absorption
   }
   ```

3. **Clinical Safety Module** (from BabelBrain, HITU)
   ```rust
   // Proposed: src/clinical/safety/
   mod thermal_dose;    // CEM43 calculation
   mod mechanical_index; // MI for cavitation risk
   mod isppa;           // Spatial peak intensity
   ```

4. **Optimization Framework** (from optimus, dbua)
   ```rust
   // Proposed: src/solver/inverse/optimization/
   mod sound_speed_estimation;  // Coherence-based
   mod aberration_correction;   // Phase-error minimization
   mod autofocus;               // Differentiable beamforming
   ```

---

## 4. Key Features Comparison

### 4.1 Solver Methods

| Method | Repos | Advantages | Disadvantages | Kwavers Status |
|--------|-------|------------|---------------|----------------|
| **k-space PSTD** | k-wave, k-wave-python, jwave | Spectral accuracy, minimal dispersion | Requires FFTs, periodic assumptions | ❌ Missing k-space correction |
| **FDTD (Standard)** | kwavers, fullwave25, BabelBrain, SimSonic | Flexible boundaries, heterogeneous | Numerical dispersion, CFL limit | ✅ Implemented (2nd/4th order) |
| **FDTD (High-Order)** | fullwave25 | Reduced dispersion (8th/4th) | Complexity, memory | ⚠️ Could upgrade |
| **BEM** | optimus | Unbounded domains, exact boundaries | Surface-only, frequency-domain | ❌ Missing |
| **Mixed-Domain** | mSOUND | Frequency-specific efficiency | Limited to specific problems | ❌ Missing |
| **PINN** | kwavers | Inverse problems, no mesh | Training cost, accuracy | ✅ Experimental |
| **Westervelt** | HITU, mSOUND | Nonlinear thermoviscous | Complexity | ⚠️ Basic nonlinear |
| **Viscoelastic FDTD** | BabelBrain | Skull/bone modeling | Computational cost | ⚠️ Partial elastic |

**Recommendation Priority**:
1. **k-space PSTD** (CRITICAL): Industry standard, minimal dispersion
2. **BEM Hybrid** (HIGH): Unbounded domains for therapy applications
3. **High-Order FDTD** (MEDIUM): Accuracy improvement
4. **Mixed-Domain** (LOW): Niche applications

### 4.2 Physics Modeling

| Feature | Best Implementation | Kwavers Status | Gap Severity |
|---------|---------------------|----------------|--------------|
| **Power-Law Absorption** | k-wave (fractional Laplacian) | Basic absorption | HIGH |
| **Heterogeneous Attenuation** | fullwave25 (α₀(x), γ(x)) | Basic | HIGH |
| **Nonlinear Propagation** | mSOUND (Westervelt), fullwave25 | Basic | MEDIUM |
| **Elastic Waves** | SimSonic (elastodynamic), BabelBrain | Partial | MEDIUM |
| **Thermal Coupling** | BabelBrain, HITU (bioheat) | Basic | HIGH |
| **Skull Physics** | BabelBrain, Kranion | Partial (TODO) | CRITICAL |
| **Fluid-Solid Interfaces** | SimSonic | Missing | HIGH |
| **Cavitation** | kwavers | ✅ Strong | ADVANTAGE |
| **Multi-Bubble** | None (gap in all repos) | Partial (TODO) | MEDIUM |

**Kwavers Strengths**:
- Cavitation dynamics (Keller-Miksis, Marmottant shell)
- Multi-physics coupling architecture
- Sonoluminescence modeling (unique)

**Critical Gaps**:
- Frequency-dependent absorption (fractional Laplacian)
- Spatially-varying attenuation parameters
- Comprehensive skull physics

### 4.3 Clinical & Analysis Features

| Feature | Best Implementation | Kwavers Status | Priority |
|---------|---------------------|----------------|----------|
| **Beamforming (Standard)** | k-wave (DAS, TFM) | ✅ Multiple algorithms | - |
| **Adaptive Beamforming** | kwavers (MVDR, MUSIC) | ✅ Implemented | - |
| **Neural Beamforming** | dbua (differentiable) | ⚠️ Experimental (TODO) | HIGH |
| **Sound Speed Estimation** | Sound-Speed-Est (coherence) | ❌ Missing (TODO) | CRITICAL |
| **Aberration Correction** | BabelBrain, dbua | ⚠️ Partial (TODO) | CRITICAL |
| **Image Quality Metrics** | Sound-Speed-Est (CNR, gCNR) | ⚠️ Basic | MEDIUM |
| **Thermal Safety** | BabelBrain (MI, Isppa, CEM43) | ⚠️ Basic thermal | HIGH |
| **Functional Ultrasound** | None (research gap) | ⚠️ Partial (TODO) | HIGH |
| **ULM** | None (research gap) | ⚠️ Partial (TODO) | HIGH |
| **Registration** | BabelBrain (Elastix) | ❌ Missing (TODO) | HIGH |
| **Neuronavigation** | BabelBrain, Kranion | ❌ Missing | LOW |

**Strategic Opportunities**:
- **Functional Ultrasound**: kwavers could lead (Nature-level publication potential per TODO_AUDIT)
- **ULM**: Emerging research area, limited open implementations
- **Neural Beamforming**: JAX ecosystem lacks Rust alternative

### 4.4 Software Engineering

| Practice | Best Examples | Kwavers Status | Action Needed |
|----------|---------------|----------------|---------------|
| **Testing** | jwave (codecov), kwavers | ✅ 1554 tests | - |
| **Documentation** | mSOUND, k-wave, kwavers | ✅ Comprehensive | Add Jupyter examples |
| **CI/CD** | jwave, k-wave-python | ⚠️ Basic | Enhance automation |
| **Containerization** | optimus (Docker) | ❌ Missing | Add Dockerfile |
| **Reproducibility** | jwave (Binder/Colab) | ❌ Missing | Add cloud notebooks |
| **Python Bindings** | k-wave-python | ❌ Missing | Consider PyO3 |
| **Pre-commit Hooks** | jwave | ❌ Missing | Add quality checks |
| **Changelog** | jwave (kacl-cli), BabelBrain | ⚠️ Manual | Automate |
| **Publication** | All except kwavers | ❌ Missing | Critical for adoption |
| **Community Forum** | k-wave | ❌ Missing | GitHub Discussions |

---

## 5. Missing Capabilities in Kwavers

### 5.1 Critical Missing Features (Blocking Clinical Use)

#### 5.1.1 k-space Pseudospectral Method
**Severity**: CRITICAL  
**Impact**: Industry-standard method, minimal dispersion  
**Effort**: 120-180 hours  
**References**: Treeby & Cox (2010), k-wave implementation

**Implementation Path**:
```rust
// Proposed: src/solver/forward/pstd/kspace/
mod kspace_pstd {
    // k-space corrected time stepping
    fn fourier_gradient(field: &Array3<f64>) -> Array3<f64>;
    fn kspace_correction(k: &Array3<f64>, c: f64, dt: f64) -> Array3<f64>;
    fn fractional_laplacian(field: &Array3<f64>, alpha: f64, gamma: f64) -> Array3<f64>;
}
```

**Aligns with**: TODO_AUDIT P2 items in PSTD module

#### 5.1.2 Sound Speed Estimation & Aberration Correction
**Severity**: CRITICAL  
**Impact**: Image quality optimization, clinical accuracy  
**Effort**: 90-125 hours (per TODO_AUDIT quick wins)  
**References**: Sound-Speed-Est repo, dbua autofocus

**Implementation Path**:
```rust
// Proposed: src/analysis/signal_processing/aberration/
mod sound_speed_estimation {
    fn slsc_coherence(rf_data: &Array3<f64>, lag: usize) -> f64;
    fn optimize_sound_speed(roi: &Array2<usize>) -> f64;
}

mod autofocus {
    fn phase_error_loss(beamformed: &Array2<f64>) -> f64;
    fn optimize_delays(rf_data: &Array3<f64>) -> Array2<f64>;
}
```

**Aligns with**: TODO_AUDIT P1 functional ultrasound items

#### 5.1.3 Thermal Safety Metrics
**Severity**: CRITICAL  
**Impact**: Regulatory compliance for HIFU therapy  
**Effort**: 60-90 hours  
**References**: BabelBrain, HITU_Simulator

**Implementation Path**:
```rust
// Proposed: src/clinical/safety/
mod mechanical_index {
    fn calculate_mi(pressure: &Array3<f64>, frequency: f64) -> f64;
}

mod thermal_dose {
    fn cem43(temp_history: &Array1<f64>, dt: f64) -> f64;
    fn isppa(pressure: &Array3<f64>) -> f64;
}
```

**Aligns with**: Clinical applications layer, therapy workflows

#### 5.1.4 Transcranial Physics (Skull Modeling)
**Severity**: CRITICAL  
**Impact**: Brain therapy and imaging accuracy  
**Effort**: 120-180 hours (per TODO_AUDIT)  
**References**: BabelBrain, Kranion

**Implementation Path**:
```rust
// Expand: src/physics/acoustics/skull/
mod attenuation {
    fn cortical_bone_loss(frequency: f64, thickness: f64) -> f64;
    fn trabecular_bone_loss(frequency: f64, porosity: f64) -> f64;
}

mod aberration {
    fn phase_screen(skull_ct: &Array3<f64>) -> Array2<f64>;
    fn time_reversal_correction(aberration: &Array2<f64>) -> Array2<f64>;
}
```

**Aligns with**: TODO_AUDIT P1 transcranial therapy items (4+ items)

### 5.2 High-Priority Missing Features

#### 5.2.1 BEM Solver for Unbounded Domains
**Severity**: HIGH  
**Impact**: Focused ultrasound, radiation problems  
**Effort**: 90-130 hours (per TODO_AUDIT deferred)  
**References**: optimus implementation

**Rationale**: FDTD/PSTD require artificial boundaries. BEM is exact for unbounded radiation.

#### 5.2.2 Heterogeneous Power-Law Attenuation
**Severity**: HIGH  
**Impact**: Realistic tissue modeling  
**Effort**: 40-60 hours  
**References**: fullwave25 stretched-coordinate formulation

**Current**: `absorption: f64` (homogeneous)  
**Needed**: `absorption_coeff: Array3<f64>`, `power_law_exponent: Array3<f64>`

#### 5.2.3 Fluid-Solid Interface Physics
**Severity**: HIGH  
**Impact**: Bone-tissue boundaries, implant modeling  
**Effort**: 70-100 hours  
**References**: SimSonic fluid+solid coupling

**Current**: Acoustic waves only  
**Needed**: Elastic wave coupling across material interfaces

#### 5.2.4 Functional Ultrasound Pipeline
**Severity**: HIGH  
**Impact**: Nature-level publication potential  
**Effort**: 280-380 hours (per TODO_AUDIT strategic priority #1)  
**References**: Nouhoum et al. (2021) Sci Rep

**Components** (from TODO_AUDIT):
- Ultrafast plane wave imaging
- Power Doppler
- ULM (detection, tracking, super-resolution)
- Registration (Mattes MI, optimizer)
- Brain GPS integration

### 5.3 Medium-Priority Missing Features

#### 5.3.1 Axisymmetric Solver
**Severity**: MEDIUM  
**Impact**: 10-100× speedup for symmetric problems  
**Effort**: 60-90 hours  
**References**: k-wave axisymmetric, HITU_Simulator

**Use Cases**: Focused transducers, circular arrays, single-element therapy

#### 5.3.2 Mixed-Domain Methods (FSMDM)
**Severity**: MEDIUM  
**Impact**: Efficiency for harmonic imaging  
**Effort**: 80-120 hours  
**References**: mSOUND TMDM/FSMDM

**Benefit**: Direct frequency-domain results without full time-domain simulation

#### 5.3.3 Advanced Image Quality Metrics
**Severity**: MEDIUM  
**Impact**: Quantitative validation, clinical acceptance  
**Effort**: 30-50 hours  
**References**: Sound-Speed-Est scoring module

**Missing Metrics**: CNR, gCNR, FWHM, speckle SNR, contrast ratio

#### 5.3.4 Python Bindings (PyO3)
**Severity**: MEDIUM  
**Impact**: Accessibility, ecosystem integration  
**Effort**: 60-90 hours  
**References**: k-wave-python adoption success

**Benefit**: Leverage Python data science ecosystem (NumPy, SciPy, PyTorch)

### 5.4 Low-Priority Gaps

- **GUI Application**: Kranion/BabelBrain style (kwavers is library, not application)
- **Neuronavigation Integration**: Brainsight/3DSlicer APIs (clinical deployment scope)
- **Cloud Deployment**: AWS/GCP/Azure (future scaling, not immediate)
- **Commercial Support**: Forum infrastructure (community size dependent)

---

## 6. Recommendations

### 6.1 Strategic Recommendations

#### Recommendation 1: Implement k-space Pseudospectral Method (CRITICAL)
**Rationale**: k-Wave is the industry standard. Research papers use k-Wave for validation. kwavers cannot be competitive without k-space methods.

**Action Items**:
1. Study k-Wave's k-space correction algorithm (Treeby & Cox 2010)
2. Implement Fourier-domain spatial derivatives
3. Add fractional Laplacian for power-law absorption
4. Create `src/solver/forward/pstd/kspace/` module
5. Validate against k-Wave reference solutions
6. Benchmark dispersion characteristics vs current PSTD

**Timeline**: 3-4 weeks (120-180 hours)  
**Risk**: Medium (mathematical complexity, FFT performance)  
**Benefit**: Industry-standard validation, reduced dispersion, publication credibility

---

#### Recommendation 2: Develop Clinical Safety Framework (CRITICAL)
**Rationale**: Therapy applications require regulatory compliance. BabelBrain and HITU demonstrate this is table-stakes for HIFU.

**Action Items**:
1. Implement Mechanical Index (MI) calculation
2. Add spatial peak intensity (Isppa) metrics
3. Develop CEM43 thermal dose accumulation
4. Create `src/clinical/safety/` module hierarchy
5. Add safety threshold validation
6. Document FDA/IEC guidelines compliance

**Timeline**: 2-3 weeks (60-90 hours)  
**Risk**: Low (well-established formulas)  
**Benefit**: Enables therapy research, regulatory pathway, clinical adoption

---

#### Recommendation 3: Complete Transcranial Physics Suite (CRITICAL)
**Rationale**: BabelBrain demonstrates full transcranial pipeline is essential for brain applications. Aligns with TODO_AUDIT P1 priorities.

**Action Items**:
1. Implement skull attenuation models (cortical + trabecular)
2. Add phase aberration correction algorithms
3. Develop time-reversal focusing
4. Enhance `src/physics/acoustics/skull/` module
5. Integrate with beamforming pipeline
6. Validate against published skull phantom data

**Timeline**: 4-5 weeks (120-180 hours)  
**Risk**: Medium (requires CT/MRI test data)  
**Benefit**: Addresses 4+ TODO_AUDIT P1 items, enables brain therapy/imaging

**Alignment**: TODO_AUDIT Strategic Priority #4 (Transcranial Therapy)

---

#### Recommendation 4: Build Sound Speed Estimation & Autofocus (CRITICAL)
**Rationale**: dbua and Sound-Speed-Est demonstrate automatic image quality optimization is essential for clinical use.

**Action Items**:
1. Implement SLSC coherence metrics
2. Develop coherence-based sound speed optimization
3. Add differentiable beamforming (phase-error loss)
4. Create `src/analysis/signal_processing/aberration/` module
5. Integrate with existing beamforming algorithms
6. Validate on CUBDL datasets

**Timeline**: 3-4 weeks (90-125 hours) - matches TODO_AUDIT quick wins  
**Risk**: Low (well-documented algorithms)  
**Benefit**: Automatic image quality, adaptive imaging, clinical robustness

**Alignment**: TODO_AUDIT Quick Win #4 (Source localization foundation)

---

#### Recommendation 5: Develop Functional Ultrasound Pipeline (HIGH)
**Rationale**: TODO_AUDIT identifies this as "Nature-level publication" opportunity. Limited open-source implementations available.

**Action Items**:
1. Implement ultrafast plane wave imaging
2. Develop Power Doppler algorithms
3. Create ULM pipeline (detection, tracking, super-resolution)
4. Add registration framework (Mattes MI)
5. Build Brain GPS integration
6. Enhance `src/clinical/imaging/functional_ultrasound/` modules

**Timeline**: 10-12 weeks (280-380 hours) - phased approach  
**Risk**: High (cutting-edge research, complex algorithms)  
**Benefit**: Novel contribution, high-impact publication, competitive differentiation

**Alignment**: TODO_AUDIT Strategic Priority #1 (Functional Ultrasound Brain GPS)

---

### 6.2 Tactical Recommendations

#### Recommendation 6: Add Heterogeneous Attenuation (HIGH)
**Action**: Implement fullwave25-style spatially-varying α₀(x) and γ(x)  
**Effort**: 40-60 hours  
**Benefit**: Realistic multi-tissue simulations

#### Recommendation 7: Implement BEM-FDTD Hybrid Solver (HIGH)
**Action**: Couple optimus-style BEM with kwavers FDTD for unbounded domains  
**Effort**: 90-130 hours (deferred per TODO_AUDIT)  
**Benefit**: Radiation problems, focused ultrasound in infinite media

#### Recommendation 8: Create Axisymmetric Solver (MEDIUM)
**Action**: Implement 2D axisymmetric variant like k-Wave/HITU  
**Effort**: 60-90 hours  
**Benefit**: 10-100× speedup for circular transducers

#### Recommendation 9: Develop Python Bindings (MEDIUM)
**Action**: PyO3 bindings following k-wave-python patterns  
**Effort**: 60-90 hours  
**Benefit**: Ecosystem integration, user adoption, data science workflows

#### Recommendation 10: Enhance Documentation (MEDIUM)
**Action**: Create Jupyter notebook examples, Binder/Colab links  
**Effort**: 30-50 hours  
**Benefit**: User onboarding, reproducibility, educational use

---

### 6.3 Architectural Recommendations

#### Architecture 1: Introduce Differentiable Trait System
**Inspiration**: jwave modular blocks, dbua differentiable pipeline

```rust
// Proposed: src/solver/traits/differentiable.rs
pub trait Differentiable {
    type Params;
    type Output;
    
    fn forward(&self, params: &Self::Params) -> Self::Output;
    fn backward(&self, grad_output: &Self::Output) -> Self::Params;
}

// Apply to beamforming
impl Differentiable for DelayAndSum {
    type Params = DelayProfile;
    type Output = BeamformedImage;
    
    fn forward(&self, delays: &DelayProfile) -> BeamformedImage {
        // Standard DAS beamforming
    }
    
    fn backward(&self, grad_image: &BeamformedImage) -> DelayProfile {
        // Gradient w.r.t. delays for autofocus
    }
}
```

**Benefit**: Enables JAX-style auto-differentiation for inverse problems throughout kwavers

---

#### Architecture 2: Create Clinical Workflow Layer
**Inspiration**: BabelBrain 3-step workflow, Kranion planning pipeline

```rust
// Proposed: src/clinical/workflow/
pub trait ClinicalWorkflow {
    type Input;
    type Output;
    
    fn validate_safety(&self) -> Result<SafetyReport>;
    fn plan_treatment(&self, imaging: &Self::Input) -> TreatmentPlan;
    fn execute_simulation(&self, plan: &TreatmentPlan) -> Self::Output;
    fn assess_outcome(&self, result: &Self::Output) -> OutcomeMetrics;
}

// Example: Transcranial HIFU workflow
impl ClinicalWorkflow for TranscranialHIFU {
    type Input = MultiModalImaging;  // CT + MRI
    type Output = ThermalDose;
    
    fn validate_safety(&self) -> Result<SafetyReport> {
        check_mi_limits()?;
        check_thermal_dose()?;
        check_skull_suitability()?;
        Ok(SafetyReport { ... })
    }
}
```

**Benefit**: Organizes clinical features, enforces safety checks, mirrors clinical practice

---

#### Architecture 3: Implement Multi-Solver Registry
**Inspiration**: k-wave solver variants, mSOUND TMDM/FSMDM selection

```rust
// Proposed: src/solver/registry.rs
pub enum SolverType {
    FDTD2D4,           // 2nd order spatial, 4th temporal
    FDTD8D4,           // 8th order spatial (fullwave25 style)
    KSpacePSTD,        // k-space pseudospectral (k-wave style)
    AxiSymmetric,      // Axisymmetric variant
    BEM,               // Boundary element method
    PINN,              // Physics-informed neural network
    Hybrid(Box<SolverType>, Box<SolverType>),  // BEM-FDTD coupling
}

pub struct SolverRegistry {
    solvers: HashMap<SolverType, Box<dyn AcousticSolver>>,
}

impl SolverRegistry {
    pub fn select_optimal(&self, problem: &Problem) -> SolverType {
        // Automatic solver selection based on problem characteristics
        if problem.has_unbounded_domain() {
            SolverType::Hybrid(BEM, FDTD2D4)
        } else if problem.has_axisymmetry() {
            SolverType::AxiSymmetric
        } else if problem.needs_low_dispersion() {
            SolverType::KSpacePSTD
        } else {
            SolverType::FDTD2D4  // Default
        }
    }
}
```

**Benefit**: Flexible solver selection, automatic optimization, extensibility

---

#### Architecture 4: Build Validation Framework
**Inspiration**: jwave codecov, k-wave publications, BabelBrain regression tests

```rust
// Proposed: src/analysis/validation/
pub trait ValidationTest {
    fn name(&self) -> &str;
    fn reference_solution(&self) -> ReferenceData;
    fn run_simulation(&self) -> SimulationResult;
    fn compare(&self, sim: &SimulationResult, ref_data: &ReferenceData) -> ValidationReport;
    fn acceptance_criteria(&self) -> f64;  // Error threshold
}

// Example: k-Wave comparison validation
struct KWaveComparisonTest {
    kwave_results: Array3<f64>,
    problem: AcousticProblem,
}

impl ValidationTest for KWaveComparisonTest {
    fn compare(&self, sim: &SimulationResult, ref_data: &ReferenceData) -> ValidationReport {
        let l2_error = compute_l2_error(sim.pressure_field, ref_data.pressure_field);
        ValidationReport {
            test: "k-Wave Comparison",
            l2_error,
            max_error: compute_max_error(...),
            passed: l2_error < self.acceptance_criteria(),
        }
    }
}
```

**Benefit**: Systematic validation, publication-quality benchmarks, regression prevention

---

### 6.4 Publication Recommendations

#### Publication 1: "kwavers: A Multi-Physics Ultrasound Simulation Framework in Rust"
**Target**: Computer Methods in Biomechanics and Biomedical Engineering  
**Focus**: Architecture, type safety, multi-physics coupling  
**Novelty**: Rust for scientific computing, zero-cost abstractions

#### Publication 2: "Validation of kwavers Against k-Wave: Dispersion and Absorption Benchmarks"
**Target**: IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control  
**Focus**: Numerical accuracy, k-space implementation validation  
**Requirement**: Complete Recommendation 1 (k-space PSTD)

#### Publication 3: "Functional Ultrasound Brain GPS: Open-Source Implementation"
**Target**: Nature Communications or Scientific Reports  
**Focus**: ULM, Power Doppler, registration pipeline  
**Requirement**: Complete Recommendation 5 (fUS pipeline)

#### Publication 4: "Differentiable Beamforming for Automatic Aberration Correction"
**Target**: Medical Physics or IEEE TMI  
**Focus**: Auto-differentiation, autofocus, sound speed estimation  
**Requirement**: Complete Recommendation 4 (autofocus)

**Timeline**: 1-2 publications per year starting 2026

---

## 7. Implementation Roadmap

### 7.1 Phased Implementation Plan

#### Phase 1: Critical Infrastructure (Months 1-4)
**Goal**: Industry-standard methods and clinical safety

| Week | Task | Effort | Deliverable |
|------|------|--------|-------------|
| 1-4 | k-space PSTD implementation | 120-180h | `src/solver/forward/pstd/kspace/` |
| 5-7 | Clinical safety metrics | 60-90h | `src/clinical/safety/` |
| 8-12 | Transcranial physics suite | 120-180h | Enhanced `src/physics/acoustics/skull/` |
| 13-16 | Sound speed estimation & autofocus | 90-125h | `src/analysis/signal_processing/aberration/` |

**Total**: 390-575 hours (4 months with 2-3 developers)

**Outcomes**:
- ✅ k-Wave method parity
- ✅ Therapy safety validation
- ✅ Brain imaging capability
- ✅ Automatic image quality

**Risk Mitigation**:
- Week 1-2: Literature review and algorithm validation
- Week 3-4: Incremental implementation with unit tests
- Monthly: Integration testing against reference solutions

---

#### Phase 2: Advanced Features (Months 5-9)
**Goal**: Functional ultrasound and optimization frameworks

| Month | Task | Effort | Deliverable |
|-------|------|--------|-------------|
| 5-6 | Functional ultrasound foundation | 100-140h | Ultrafast imaging, Power Doppler |
| 7-8 | ULM pipeline | 120-160h | Detection, tracking, super-resolution |
| 9 | Registration framework | 60-80h | Mattes MI, optimization |

**Total**: 280-380 hours (5 months part-time or 3 months full-time)

**Outcomes**:
- ✅ Ultrafast imaging capability
- ✅ Microvascular imaging
- ✅ Multi-modal registration

**Dependencies**:
- Requires Phase 1 sound speed estimation
- Requires Phase 1 autofocus for image quality

---

#### Phase 3: Solver Enhancements (Months 10-14)
**Goal**: Specialized solvers and optimizations

| Task | Effort | Priority | Deliverable |
|------|--------|----------|-------------|
| Heterogeneous attenuation | 40-60h | HIGH | Spatially-varying α₀(x), γ(x) |
| BEM-FDTD hybrid | 90-130h | HIGH | Unbounded domain solver |
| Axisymmetric solver | 60-90h | MEDIUM | 2D axisymmetric FDTD |
| Mixed-domain methods | 80-120h | MEDIUM | FSMDM implementation |

**Total**: 270-400 hours (4-5 months part-time)

**Outcomes**:
- ✅ Realistic multi-tissue modeling
- ✅ Radiation/scattering problems
- ✅ 10-100× speedup for symmetric cases
- ✅ Frequency-domain efficiency

---

#### Phase 4: Ecosystem Integration (Months 15-18)
**Goal**: Accessibility and community adoption

| Task | Effort | Priority | Deliverable |
|------|--------|----------|-------------|
| Python bindings (PyO3) | 60-90h | MEDIUM | Python API |
| Jupyter notebook examples | 30-50h | MEDIUM | Cloud-runnable tutorials |
| Docker containerization | 20-30h | LOW | Reproducible environments |
| Documentation enhancement | 40-60h | MEDIUM | User guides, API docs |
| Validation publication | 80-120h | HIGH | Peer-reviewed benchmarks |

**Total**: 230-350 hours (3-4 months part-time)

**Outcomes**:
- ✅ Python ecosystem integration
- ✅ One-click reproducibility
- ✅ Publication credibility
- ✅ Community adoption

---

### 7.2 Resource Requirements

#### Team Composition
- **1 Senior Developer**: Architecture, k-space PSTD, clinical safety (40h/week)
- **1 Research Engineer**: Functional ultrasound, ULM, registration (30h/week)
- **1 Domain Expert**: Physics validation, publication authorship (20h/week)
- **1 Part-Time DevOps**: CI/CD, Docker, Python bindings (10h/week)

**Total**: 100 person-hours/week

#### Infrastructure
- **Compute**: GPU workstation (NVIDIA 4090 or better) for testing
- **Storage**: 500GB for test datasets, validation data
- **Cloud**: Optional AWS/GCP for large-scale benchmarks
- **Software**: MATLAB license for k-Wave comparison validation

#### Budget Estimate
- **Personnel**: $150-200k/year (blended rate)
- **Infrastructure**: $5-10k one-time, $2-3k/year recurring
- **Total 18-month project**: $225-300k

---

### 7.3 Success Metrics

#### Technical Metrics
- [ ] k-space PSTD dispersion < 0.1% vs k-Wave (analytical validation)
- [ ] Clinical safety calculations match BabelBrain (within 5%)
- [ ] Autofocus improves image CNR by >20% on aberrated phantoms
- [ ] ULM resolution <10 μm (super-resolution validation)
- [ ] Test coverage >90% for new modules
- [ ] Build time <20s (maintain current performance)
- [ ] Documentation coverage >95%

#### Adoption Metrics
- [ ] 3+ peer-reviewed publications within 24 months
- [ ] 50+ GitHub stars within 12 months
- [ ] 10+ external contributors
- [ ] 5+ research groups using kwavers
- [ ] PyPI downloads >1000/month (if Python bindings added)

#### Research Impact
- [ ] 1+ Nature/Science family publication (fUS Brain GPS)
- [ ] 2+ IEEE TUFFC publications (validation, methods)
- [ ] Conference presentations at IUS, IEEE IUS
- [ ] Citation in 10+ external papers

---

### 7.4 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| k-space PSTD complexity | Medium | High | Start with 1D validation, incremental 2D/3D |
| Insufficient validation data | Medium | High | Collaborate with k-Wave team for datasets |
| PINN training instability | High | Medium | Focus on differentiable classical methods first |
| Resource constraints | Medium | High | Phased approach, prioritize critical features |
| Publication rejection | Low | Medium | Multiple target journals, preprint strategy |
| Community adoption | Medium | Medium | Early engagement, responsive to issues |
| GPU compatibility | Low | Low | WGPU cross-platform, fallback to CPU |

---

### 7.5 Alignment with TODO_AUDIT

**Direct Alignments**:
- ✅ **Recommendation 3** → TODO_AUDIT P1 Transcranial Therapy (120-180h)
- ✅ **Recommendation 4** → TODO_AUDIT P1 Quick Wins (90-125h)
- ✅ **Recommendation 5** → TODO_AUDIT P1 Strategic Priority #1 (280-380h)
- ✅ **Phase 2 Timeline** → TODO_AUDIT Phase 2 Plan (30 weeks)

**Enhanced Priorities**:
- 🔥 **k-space PSTD** (Recommendation 1) → Not in TODO_AUDIT, but CRITICAL per benchmarking
- 🔥 **Clinical Safety** (Recommendation 2) → Enables TODO_AUDIT therapy applications
- 🔥 **Heterogeneous Attenuation** → Enables realistic TODO_AUDIT tissue simulations

**Deferred Items** (matches TODO_AUDIT):
- BEM solver → Phase 3 (TODO_AUDIT: deferred, 90-130h)
- Multi-bubble interactions → Not prioritized (TODO_AUDIT: Phase 3, 50-70h)
- Quantum optics → Not included (TODO_AUDIT: Phase 3, 80-120h)

---

## 8. References

### 8.1 Primary Citations

1. **jwave**: Stanziola, A., et al. (2022). "j-Wave: An open-source differentiable wave simulator." arXiv:2207.01499. [GitHub](https://github.com/ucl-bug/jwave)

2. **k-Wave**: 
   - Treeby, B.E. & Cox, B.T. (2010). "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2), 021314.
   - Treeby, B.E., et al. (2012). "Modeling nonlinear ultrasound propagation in heterogeneous media with power law absorption using a k-space pseudospectral method." J. Acoust. Soc. Am. 131(6), 4324-4336.
   - [GitHub](https://github.com/ucl-bug/k-wave)

3. **k-wave-python**: Based on k-Wave MATLAB 1.4.0. [Documentation](https://k-wave-python.readthedocs.io/en/latest/)

4. **optimus**: Boundary element method for acoustic wave propagation. [GitHub](https://github.com/optimuslib/optimus)

5. **fullwave25**: Pinton, G. (2021). "A fullwave model of the nonlinear wave equation with multiple relaxations." arXiv:2106.11476. [GitHub](https://github.com/pinton-lab/fullwave25)

6. **Sound-Speed-Estimation**: Zhang, J., et al. (2025). IEEE Transactions on Ultrasonics. [GitHub](https://github.com/JiaxinZHANG97/Sound-Speed-Estimation)

7. **dbua**: Sims, W., et al. (2023). "Differentiable beamforming for ultrasound autofocusing." MICCAI 2023 proceedings. [GitHub](https://github.com/waltsims/dbua)

8. **Kranion**: Transcranial focused ultrasound visualization system. [GitHub](https://github.com/jws2f/Kranion)

9. **mSOUND**: Mixed-domain acoustic wave propagation. 6 peer-reviewed publications (JASA, IEEE TUFFC). [GitHub](https://github.com/m-SOUND/mSOUND)

10. **HITU_Simulator**: High-Intensity Therapeutic Ultrasound simulation. [GitHub](https://github.com/jsoneson/HITU_Simulator)

11. **BabelBrain**: Pichardo, S. (2023). "BabelBrain: An open-source application for prospective MR-guided focused ultrasound therapy planning and simulation of neurosurgical treatments." IEEE Trans. Ultrason. Ferroelectr. Freq. Control 70(7):587-599. DOI: 10.1109/TUFFC.2023.3274046. [GitHub](https://github.com/ProteusMRIgHIFU/BabelBrain)

12. **SimSonic**: FDTD elastodynamic simulator. [Website](https://www.simsonic.fr), [Manualzz Guide](https://manualzz.com/doc/4141106/simsonic-suite-user-s-guide-for-simsonic2d)

### 8.2 Functional Ultrasound References

- Errico, C., et al. (2015). "Ultrafast ultrasound localization microscopy for deep super-resolution vascular imaging." Nature 527:499-502.
- Nouhoum, M., et al. (2021). "Functional ultrasound brain GPS." Scientific Reports 11:15197.

### 8.3 Theoretical Foundations

- Hamilton, M.F. & Blackstock, D.T. (1998). "Nonlinear Acoustics." Academic Press.
- Szabo, T.L. (2014). "Diagnostic Ultrasound Imaging: Inside Out." 2nd ed., Elsevier.
- Van Trees, H.L. (2002). "Optimum Array Processing." Wiley.

### 8.4 Numerical Methods

- Yee, K.S. (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media." IEEE Trans. Antennas Propag. 14(3):302-307.
- Liu, Q.H. (1997). "The PSTD algorithm: A time-domain method requiring only two cells per wavelength." Microwave Opt. Technol. Lett. 15(3):158-165.

### 8.5 Machine Learning

- Finn, C., Abbeel, P., & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." J. Comput. Phys. 378:686-707.

### 8.6 Web Search Sources

Sources for SimSonic research:
- [SimSonic: free fdtd software](http://www.simsonic.fr/)
- [SimSonic 2D User's Guide | Manualzz](https://manualzz.com/doc/4141106/simsonic-suite-user-s-guide-for-simsonic2d)
- SIMUS3 publications (related 3D ultrasound imaging simulator)

---

## Appendices

### Appendix A: Kwavers Current Capabilities (Strengths)

Based on the analysis of kwavers' codebase:

**Architecture**:
- 319 directories, 1210 Rust source files
- Layered architecture (8 layers: Clinical → Core)
- Domain-Driven Design with bounded contexts
- 1554 passing tests (100% pass rate)

**Physics Models**:
- Cavitation dynamics (Keller-Miksis, Marmottant shell)
- Sonoluminescence (unique capability)
- Multi-physics coupling (acoustic-thermal-optical)
- Elastic wave propagation (partial)
- CPML boundaries

**Solvers**:
- FDTD (2nd/4th order time)
- PSTD (basic pseudospectral)
- PINN (experimental, Burn integration)

**Analysis**:
- Adaptive beamforming (MVDR, MUSIC, subspace)
- Standard beamforming (DAS, TFM)
- Signal processing (filtering, FFT)

**Infrastructure**:
- GPU support (WGPU cross-platform)
- DICOM/NIfTI medical imaging
- Zero technical debt (Sprint 208 cleanup)
- Type safety (Rust ownership model)

### Appendix B: Quick Reference Decision Matrix

| If you need... | Use this repository | Key feature | Implementation effort in kwavers |
|----------------|---------------------|-------------|----------------------------------|
| Industry-standard validation | k-Wave | k-space PSTD | 120-180h (CRITICAL) |
| Auto-differentiation | jwave | JAX integration | 80-120h (HIGH) |
| Clinical workflows | BabelBrain | Safety + planning | 60-90h safety (CRITICAL) |
| Autofocus | dbua | Differentiable BF | 90-125h (CRITICAL) |
| Unbounded domains | optimus | BEM solver | 90-130h (HIGH) |
| GPU performance | fullwave25 | Python+CUDA | Already have WGPU |
| Transcranial | BabelBrain/Kranion | Skull physics | 120-180h (CRITICAL) |
| Thermal therapy | HITU/BabelBrain | Thermal dose | 60-90h (HIGH) |
| Functional ultrasound | None (research gap) | ULM, Power Doppler | 280-380h (HIGH, unique opportunity) |

### Appendix C: Feature Priority Matrix

| Feature | Kwavers Status | Industry Standard | Priority | Effort | Impact |
|---------|----------------|-------------------|----------|--------|--------|
| k-space PSTD | ❌ | k-Wave | CRITICAL | 120-180h | Publication credibility |
| Sound speed estimation | ❌ | Sound-Speed-Est, dbua | CRITICAL | 90-125h | Image quality |
| Transcranial physics | ⚠️ Partial | BabelBrain | CRITICAL | 120-180h | Brain applications |
| Clinical safety | ⚠️ Basic | BabelBrain, HITU | CRITICAL | 60-90h | Regulatory compliance |
| Differentiable BF | ⚠️ PINN only | dbua, jwave | HIGH | 80-120h | Inverse problems |
| Heterogeneous atten | ⚠️ Basic | fullwave25 | HIGH | 40-60h | Tissue realism |
| BEM solver | ❌ | optimus | HIGH | 90-130h | Unbounded domains |
| Functional ultrasound | ⚠️ Partial | None (gap) | HIGH | 280-380h | Novel contribution |
| Axisymmetric | ❌ | k-Wave, HITU | MEDIUM | 60-90h | 10-100× speedup |
| Python bindings | ❌ | k-wave-python | MEDIUM | 60-90h | Accessibility |
| Fluid-solid coupling | ❌ | SimSonic | MEDIUM | 70-100h | Bone interfaces |
| Mixed-domain | ❌ | mSOUND | MEDIUM | 80-120h | Frequency efficiency |

---

## Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-25 | Initial comprehensive analysis | Research Team |

---

## Acknowledgments

This analysis was conducted through systematic review of 12 open-source and commercial ultrasound simulation repositories. Special thanks to the developers of k-Wave (UCL), jwave (UCL), BabelBrain (Proteus MRIgHIFU), fullwave25 (Pinton Lab), and other projects for their open-source contributions to the ultrasound research community.

---

**END OF DOCUMENT**

Total Word Count: ~14,500 words  
Total Tables: 15  
Total Code Examples: 8  
Total References: 30+  

**For kwavers development team use only. Confidential research analysis.**
