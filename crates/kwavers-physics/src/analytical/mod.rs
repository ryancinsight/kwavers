//! Cross-domain analytical physics kernels.
//!
//! Each submodule covers an analytical model family used by validation,
//! documentation, and PyO3 figure-generation bindings. All computation is in
//! Rust; Python examples call these via PyO3 bindings and are responsible only
//! for plotting.
//!
//! # Submodule index
//!
//! | Module | Domain | Chapters |
//! |---|---|---|
//! | [`wave`] | Wave physics, harmonics, dispersion | ch01, ch02, ch03, ch08 |
//! | [`transducer`] | Array beamforming, directivity | ch04, ch11 |
//! | [`cavitation`] | Bubble dynamics, RP / KM ODEs | ch07, ch09 |
//! | [`tissue`] | Tissue acoustical properties | ch12 |
//! | [`safety`] | Dosimetry, MI, TI, CEM43 | ch15 |
//! | [`skull`] | Transcranial, CT-to-medium | ch16, ch25 |
//! | [`photoacoustics`] | PA signal, spectroscopy | ch13 |
//! | [`elastography`] | Shear waves, MRE, acousto-elastic | ch10 |
//! | [`murnaghan`] | Third-order (Murnaghan) elastic constitutive law | ch10, ch11 |
//! | [`imaging`] | PSF, Doppler, compounding, CEUS | ch05, ch24 |
//! | [`thermal`] | Bioheat, HIFU, Beer-Lambert Q(z) | ch06 |
//! | [`inverse`] | Tikhonov, Born inversion | ch17 |
//! | [`sonogenetics`] | Hill activation, ARF | ch18 |
//! | [`rtm`] | Reverse-time migration | ch25 |
//! | [`bbb`] | BBB permeability, CEUS, closure kinetics | ch24 |

pub mod acousto_optics;
pub mod bbb;
pub mod cavitation;
pub mod elastography;
pub mod imaging;
pub mod inverse;
pub mod murnaghan;
pub mod photoacoustics;
pub mod pulse_echo;
pub mod rtm;
pub mod safety;
pub mod skull;
pub mod sonogenetics;
pub mod thermal;
pub mod tissue;
pub mod transducer;
pub mod wave;
