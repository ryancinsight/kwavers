//! Burn → Coeus compatibility shim for the PINN module.
//!
//! Provides burn-compatible types and traits implemented over coeus Atlas crates.
//! All types are specialized to `f32` and `MoiraiBackend` (= burn's `NdArray<f32>`
//! with autodiff). Files that previously `use crate::burn::...` should migrate to
//! `use crate::inverse::pinn::ml::burn_compat::*;` (or specific imports).
//!
//! ## Migration status
//!
//! This shim makes all PINN source files compile against coeus. Per-file
//! adaptation to native coeus API is tracked separately; once complete, imports
//! from this module are removed in favour of direct coeus imports.
//!
//! ## Architecture
//!
//! | Burn concept | Coeus equivalent (this module) |
//! |---|---|
//! | `Backend` / `AutodiffBackend` | `BackendOps<f32>` + `Default` |
//! | `Tensor<B, N>` | `Tensor<B, N>` newtype wrapping `Var<f32, B>` |
//! | `Module` derive | Manual `Module<f32, B>` impl via trait |
//! | `Linear<B>` | `coeus_nn::Linear<f32, B>` |
//! | `LinearConfig::new(in, out).init(device)` | `Linear::new(in, out, true)` |
//! | `backward()` → `Gradients` + `map(&mut mapper)` | `backward()` → grads in Var leaves; `Optimizer::step()` |

use std::fmt;

use coeus_autograd::{scalar_mul, Var, VarScalarExt};
use coeus_core::{MoiraiBackend, Scalar};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor as CoeusTensor;

// ── Backend type aliases ──────────────────────────────────────────────────────

/// The default coeus CPU backend used for PINN execution.
///
/// Equivalent to burn's `NdArray<f32>` backend.
pub type CpuBackend = MoiraiBackend;

/// Type alias for `MoiraiBackend` as a drop-in for `NdArray<E>`.
///
/// Usage: `type B = NdArray<f32>;` → `type B = NdArray;`
pub type NdArray = MoiraiBackend;

/// Wrapper that signals autodiff is active over a backend.
///
/// In coeus, autodiff is provided by `Var<T, B>` directly; this type alias
/// passes through to `B` since there is no separate autodiff wrapper.
///
/// Usage: `type B = Autodiff<NdArray<f32>>;` → `type B = Autodiff<NdArray>;`
pub type Autodiff<B> = B;

// ── Backend traits ────────────────────────────────────────────────────────────

/// The device type — always `DefaultDevice` for this CPU-only shim.
///
/// Exposed as `B::Device` via the `Backend` trait.
pub type Device = DefaultDevice;

/// `Devices` is the list-of-devices type returned by `module.collect_devices()`.
///
/// In this CPU-only shim it is always `Vec<DefaultDevice>`.
pub type Devices = Vec<DefaultDevice>;

/// Replacement for `crate::burn::tensor::backend::Backend`.
///
/// In coeus the concept maps to `BackendOps<f32> + Default`.
pub trait Backend: BackendOps<f32> + Default + Clone + fmt::Debug {
    /// The device type.  Always `DefaultDevice` for this shim.
    type Device: Clone + fmt::Debug + Default + Send + Sync + PartialEq;
}
impl<B: BackendOps<f32> + Default + Clone + fmt::Debug> Backend for B {
    type Device = DefaultDevice;
}

/// Replacement for `crate::burn::tensor::backend::AutodiffBackend`.
///
/// In coeus autodiff is handled by `Var<T, B>` rather than by the backend;
/// all backends that satisfy `Backend` also satisfy this.
pub trait AutodiffBackend: Backend {
    /// The inner (non-autodiff) backend type.  In coeus this is the same type.
    type InnerBackend: Backend;
    /// Gradients — a unit struct because in coeus gradients are stored inside
    /// the `Var` leaf nodes and do not need to be threaded through calls.
    type Gradients: GradientsMap;
}

impl<B: Backend> AutodiffBackend for B {
    type InnerBackend = B;
    type Gradients = CoeusTensorGradients<B>;
}

// ── Gradients placeholder ─────────────────────────────────────────────────────

/// Marker trait for the gradients associated type on `AutodiffBackend`.
pub trait GradientsMap {}

/// Coeus gradient container: gradients live in `Var` leaves, not a separate map.
///
/// This is a zero-sized struct; burn code that pattern-matches on gradients is
/// adapted to read `var.grad()` directly instead.
pub struct CoeusTensorGradients<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> GradientsMap for CoeusTensorGradients<B> {}

impl<B: Backend> fmt::Debug for CoeusTensorGradients<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoeusTensorGradients")
    }
}

// ── Device ───────────────────────────────────────────────────────────────────

/// Replacement for `B::Device` — a unit struct for the CPU default device.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DefaultDevice;

// ── Tensor newtype ────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::tensor::Tensor<B, N>`.
///
/// Wraps `Var<f32, B>` — a differentiable variable.
/// The const `N` is kept for type-compatibility with existing call sites that
/// write `Tensor<B, 2>` etc.; it is not enforced at runtime (coeus uses
/// dynamic rank like numpy).
#[derive(Clone)]
pub struct Tensor<B: Backend, const N: usize = 1> {
    pub(crate) inner: Var<f32, B>,
}

impl<B: Backend, const N: usize> fmt::Debug for Tensor<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor<{}D, shape={:?}>",
            N,
            self.inner.tensor.shape()
        )
    }
}

impl<B: Backend, const N: usize> Tensor<B, N> {
    /// Wrap a coeus `Var` as a burn-compatible `Tensor`.
    #[inline]
    pub fn from_var(var: Var<f32, B>) -> Self {
        Self { inner: var }
    }

    /// Return the underlying `Var`.
    #[inline]
    pub fn into_var(self) -> Var<f32, B> {
        self.inner
    }

    /// Create from a flat float slice and a device (device is ignored; CPU-only).
    pub fn from_floats(data: &[f32], _device: &DefaultDevice) -> Self {
        let shape: Vec<usize> = vec![data.len()];
        let t = CoeusTensor::from_slice_on(shape, data, &B::default());
        Self::from_var(Var::new(t, false))
    }

    /// Reshape to a new dynamic shape while preserving the const rank parameter.
    ///
    /// # Panics
    /// Panics if the new element count differs from the current one.
    pub fn reshape<const M: usize>(self, shape: [usize; M]) -> Tensor<B, M> {
        let v = coeus_autograd::reshape(&self.inner, shape.to_vec());
        Tensor::<B, M>::from_var(v)
    }

    /// Create a zero tensor with the given shape.
    pub fn zeros(shape: impl IntoShape, _device: &DefaultDevice) -> Self {
        let s = shape.into_shape();
        let t = CoeusTensor::zeros_on(s, &B::default());
        Self::from_var(Var::new(t, false))
    }

    /// Concatenate tensors along `dim`.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let vars: Vec<Var<f32, B>> = tensors.into_iter().map(|t| t.inner).collect();
        let refs: Vec<&Var<f32, B>> = vars.iter().collect();
        Self::from_var(coeus_autograd::cat(&refs, dim))
    }

    /// Mark this tensor as requiring a gradient (for autodiff).
    pub fn require_grad(self) -> Self {
        let t = self.inner.tensor.clone();
        Self::from_var(Var::new(t, true))
    }

    /// Run reverse-mode autodiff from this scalar or summed tensor.
    ///
    /// Returns a gradients token. In coeus, gradients accumulate in the `Var`
    /// leaves rather than in a separate `Gradients` map; this call just triggers
    /// the backward pass and returns a unit-like token.
    pub fn backward(self) -> CoeusTensorGradients<B> {
        self.inner.backward();
        CoeusTensorGradients {
            _marker: std::marker::PhantomData,
        }
    }

    /// Read the gradient of this tensor.
    ///
    /// Returns `Some(Tensor)` when `require_grad` was set and `backward` has
    /// been called; otherwise `None`.
    pub fn grad(&self, _grads: &CoeusTensorGradients<B>) -> Option<Tensor<B, N>> {
        self.inner.grad().map(|g| {
            let v = Var::new(g, false);
            Self::from_var(v)
        })
    }

    /// Get the inner non-autodiff tensor.
    pub fn inner(&self) -> Tensor<<B as AutodiffBackend>::InnerBackend, N> {
        let t = self.inner.tensor.clone();
        let v = Var::new(t, false);
        Tensor::<<B as AutodiffBackend>::InnerBackend, N>::from_var(v)
    }

    /// Wrap a non-autodiff tensor back into an autodiff `Tensor`.
    pub fn from_inner(inner: Tensor<<B as AutodiffBackend>::InnerBackend, N>) -> Self {
        Self::from_var(inner.inner)
    }

    /// Element-wise tanh activation.
    pub fn tanh(self) -> Self {
        Self::from_var(coeus_autograd::tanh(&self.inner))
    }

    /// Element-wise ReLU activation.
    pub fn relu(self) -> Self {
        Self::from_var(coeus_autograd::relu(&self.inner))
    }

    /// Element-wise sigmoid activation.
    pub fn sigmoid(self) -> Self {
        Self::from_var(coeus_autograd::sigmoid(&self.inner))
    }

    /// Element-wise power: `self ^ exponent`.
    pub fn powf_scalar(self, exponent: f64) -> Self
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        Self::from_var(coeus_autograd::pow(&self.inner, exponent))
    }

    /// Mean over all elements.
    pub fn mean(self) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_var(coeus_autograd::mean(&self.inner))
    }

    /// Sum over all elements.
    pub fn sum(self) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_var(coeus_autograd::sum(&self.inner))
    }

    /// Element-wise multiplication by a scalar.
    pub fn mul_scalar(self, scalar: f64) -> Self {
        Self::from_var(scalar_mul(&self.inner, scalar as f32))
    }

    /// Get the device (returns `DefaultDevice` since coeus always uses CPU default).
    pub fn device(&self) -> DefaultDevice {
        DefaultDevice
    }

    /// Get the shape of this tensor.
    pub fn shape(&self) -> TensorShape<N> {
        let s = self.inner.tensor.shape().to_vec();
        TensorShape { dims: s }
    }

    /// Check whether this tensor is tracking gradients.
    pub fn is_require_grad(&self) -> bool {
        self.inner.grad.is_some()
    }

    /// Re-enable grad tracking on this tensor.
    pub fn set_require_grad(self, require_grad: bool) -> Self {
        if require_grad {
            self.require_grad()
        } else {
            let t = self.inner.tensor.clone();
            Self::from_var(Var::new(t, false))
        }
    }

    /// Convert to `TensorData` for extracting values.
    pub fn into_data(self) -> TensorData
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        TensorData {
            data: self.inner.tensor.as_slice().to_vec(),
            shape: self.inner.tensor.shape().to_vec(),
        }
    }

    /// Negate element-wise.
    pub fn neg(self) -> Self {
        Self::from_var(coeus_autograd::neg(&self.inner))
    }

    /// Element-wise subtraction.
    pub fn sub(self, rhs: Self) -> Self {
        Self::from_var(coeus_autograd::sub(&self.inner, &rhs.inner))
    }

    /// Element-wise addition.
    pub fn add(self, rhs: Self) -> Self {
        Self::from_var(coeus_autograd::add(&self.inner, &rhs.inner))
    }

    /// Element-wise multiplication.
    pub fn mul(self, rhs: Self) -> Self {
        Self::from_var(coeus_autograd::mul(&self.inner, &rhs.inner))
    }

    /// Element-wise division.
    pub fn div(self, rhs: Self) -> Self {
        Self::from_var(coeus_autograd::div(&self.inner, &rhs.inner))
    }
}

impl<B: Backend, const N: usize> std::ops::Add for Tensor<B, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_var(coeus_autograd::add(&self.inner, &rhs.inner))
    }
}

impl<B: Backend, const N: usize> std::ops::Sub for Tensor<B, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_var(coeus_autograd::sub(&self.inner, &rhs.inner))
    }
}

impl<B: Backend, const N: usize> std::ops::Mul for Tensor<B, N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_var(coeus_autograd::mul(&self.inner, &rhs.inner))
    }
}

impl<B: Backend, const N: usize> std::ops::Neg for Tensor<B, N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::from_var(coeus_autograd::neg(&self.inner))
    }
}

// ── TensorShape ───────────────────────────────────────────────────────────────

/// Replacement for burn's `Shape<N>`.
#[derive(Debug, Clone)]
pub struct TensorShape<const N: usize> {
    /// Dimension sizes.
    pub dims: Vec<usize>,
}

// ── TensorData ────────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::tensor::TensorData`.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Flat data buffer in row-major order.
    pub data: Vec<f32>,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
}

impl TensorData {
    /// Access the data as a typed slice (only `f32` is supported).
    pub fn as_slice<T: 'static>(&self) -> Option<&[T]> {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: T == f32 verified above
            let s: &[f32] = &self.data;
            let ptr = s.as_ptr().cast::<T>();
            Some(unsafe { std::slice::from_raw_parts(ptr, s.len()) })
        } else {
            None
        }
    }
}

// ── IntoShape helper ─────────────────────────────────────────────────────────

/// Helper trait to convert shape arguments to a `Vec<usize>`.
pub trait IntoShape {
    /// Convert to owned shape vector.
    fn into_shape(self) -> Vec<usize>;
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl IntoShape for TensorShape<1> {
    fn into_shape(self) -> Vec<usize> {
        self.dims
    }
}
impl IntoShape for TensorShape<2> {
    fn into_shape(self) -> Vec<usize> {
        self.dims
    }
}
impl IntoShape for TensorShape<3> {
    fn into_shape(self) -> Vec<usize> {
        self.dims
    }
}

// ── ElementConversion ────────────────────────────────────────────────────────

/// Replacement for `crate::burn::prelude::ElementConversion` / `ToElement`.
///
/// The one method needed is `elem::<f32>()` to extract a scalar from a tensor.
pub trait ElementConversion: Sized {
    /// Convert to `f32`.
    fn elem<T: Copy + 'static>(self) -> T;
}

impl ElementConversion for f32 {
    fn elem<T: Copy + 'static>(self) -> T {
        // SAFETY: only f32→f32 is used in PINN code
        let v = self;
        let ptr = &v as *const f32 as *const T;
        unsafe { *ptr }
    }
}

// ── Module trait ──────────────────────────────────────────────────────────────

/// Unit record type — the serialization record for all stub modules.
#[derive(Clone, Debug, Default)]
pub struct ModuleRecord;

/// Replacement for `crate::burn::module::Module<B>` in burn_compat.
///
/// All methods have default implementations so existing network structs can
/// satisfy the trait with just `impl<B: Backend> Module<B> for MyNet<B> {}`.
/// The `parameters()` method defaults to an empty vec; concrete impls should
/// override it to enable optimizer-based training.
///
/// Note: this trait is unused when burn is the active backend (burn provides its
/// own Module derive); it's only active during the coeus-migration transition.
pub trait Module<B: Backend>: Sized + fmt::Debug + Clone {
    /// Return all learnable parameters (default: none).
    fn parameters(&self) -> Vec<Var<f32, B>> { vec![] }

    /// Visit all float tensors.
    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {}

    /// Map all float parameters through a `ModuleMapper`.
    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self { self }

    /// Serialize to a ModuleRecord (stub).
    fn into_record(self) -> ModuleRecord { ModuleRecord }

    /// Deserialize from a ModuleRecord (stub: returns self).
    fn load_record(self, _record: ModuleRecord) -> Self { self }

    /// Move to device (no-op for CPU).
    fn to_device(self, _device: &<B as Backend>::Device) -> Self { self }

    /// Fork module for independent gradient paths.
    fn fork(self, _device: &<B as Backend>::Device) -> Self { self }

    /// Collect devices.
    fn collect_devices(&self) -> Devices { vec![DefaultDevice] }
}

/// Replacement for `crate::burn::module::AutodiffModule<B>`.
///
/// `valid()` defaults to a clone — concrete impls may override.
pub trait AutodiffModule<B: AutodiffBackend>: Module<B> {
    /// Detach to the inner (non-autodiff) representation (defaults to clone).
    fn valid(self) -> Self where Self: Clone { self }

    /// Move to device.
    fn to_device(self, _device: &<B as Backend>::Device) -> Self { self }

    /// Return device list.
    fn devices(&self) -> Devices { vec![DefaultDevice] }
}

// ── ModuleMapper ──────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::module::ModuleMapper<B>`.
pub trait ModuleMapper<B: AutodiffBackend> {
    /// Map a float parameter tensor.
    fn map_float<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>>;

    /// Pass integer parameters through unchanged.
    fn map_int<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        tensor
    }

    /// Pass boolean parameters through unchanged.
    fn map_bool<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        tensor
    }
}

// ── ModuleVisitor ─────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::module::ModuleVisitor<B>`.
pub trait ModuleVisitor<B: AutodiffBackend> {
    /// Visit a float parameter tensor.
    fn visit_float<const D: usize>(&mut self, id: &str, tensor: &Tensor<B, D>);
}

// ── Param ─────────────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::module::Param<T>`.
///
/// A thin wrapper around a value that carries a parameter ID.
#[derive(Debug, Clone)]
pub struct Param<T> {
    /// Parameter identifier (for serialisation / mapping).
    pub id: String,
    /// The wrapped value.
    pub value: T,
}

impl<T: Clone> Param<T> {
    /// Create from an existing tensor (generates a UUID-like id).
    pub fn from_tensor(value: T) -> Self {
        // Use a simple counter for unique IDs
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("param_{id}"),
            value,
        }
    }
    /// Create with a specific id.
    pub fn new(id: impl Into<String>, value: T) -> Self {
        Self {
            id: id.into(),
            value,
        }
    }
}

impl<B: Backend, const N: usize> std::ops::Deref for Param<Tensor<B, N>> {
    type Target = Tensor<B, N>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<B: AutodiffBackend, const N: usize> Param<Tensor<B, N>> {
    /// Extract the gradient of this parameter tensor.
    pub fn grad(&self, _grads: &CoeusTensorGradients<B>) -> Option<Tensor<B, N>> {
        self.value.inner.grad().map(|g| {
            Tensor::from_var(Var::new(g, false))
        })
    }
}

// ── Ignored wrapper ───────────────────────────────────────────────────────────

/// Replacement for `crate::burn::module::Ignored<T>`.
///
/// Marks a field as non-parameter (not visited by mappers/visitors).
#[derive(Debug, Clone)]
pub struct Ignored<T>(pub T);

impl<T> std::ops::Deref for Ignored<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Ignored<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ── Linear layer ─────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::nn::Linear<B>`.
///
/// Wraps `coeus_nn::Linear<f32, B>`.
#[derive(Clone)]
pub struct Linear<B: Backend> {
    pub(crate) inner: coeus_nn::Linear<f32, B>,
}

impl<B: Backend> fmt::Debug for Linear<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear")
    }
}

impl<B: Backend + coeus_ops::BackendOps<f32>> Linear<B> {
    /// Forward pass: `out = input · Wᵀ + b`.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        use coeus_nn::module::Module as CoeusModule;
        let out = self.inner.forward(&input.inner);
        Tensor::from_var(out)
    }

    /// List devices (always `[DefaultDevice]`).
    pub fn devices(&self) -> Vec<DefaultDevice> {
        vec![DefaultDevice]
    }
}

/// Replacement for `crate::burn::nn::LinearConfig`.
#[derive(Debug, Clone)]
pub struct LinearConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
}

impl LinearConfig {
    /// Create a new linear layer configuration.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
        }
    }

    /// Set whether to include a bias term (default: true).
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Build the layer (device parameter is ignored; always CPU).
    pub fn init<B: Backend + coeus_ops::BackendOps<f32>>(&self, _device: &<B as Backend>::Device) -> Linear<B> {
        let inner = coeus_nn::Linear::<f32, B>::new(self.in_features, self.out_features, self.bias);
        Linear { inner }
    }
}

// ── Burn module impl for Linear ───────────────────────────────────────────────

impl<B: Backend + coeus_ops::BackendOps<f32>> Module<B> for Linear<B> {
    fn parameters(&self) -> Vec<Var<f32, B>> {
        use coeus_nn::module::Module as CoeusModule;
        self.inner.parameters()
    }
}

impl<B: AutodiffBackend + coeus_ops::BackendOps<f32>> AutodiffModule<B> for Linear<B> {}

// ── Optimizers ────────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::optim::Adam<B, Lr, Wd>`.
///
/// The `Lr` and `Wd` type parameters are burn-specific; they're kept as
/// phantom parameters here to allow existing type annotations to compile.
pub struct Adam<B: Backend, Lr = (), Wd = ()>
where
    B: coeus_ops::BackendOps<f32>,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    inner: coeus_optim::Adam<f32, B>,
    _lr: std::marker::PhantomData<Lr>,
    _wd: std::marker::PhantomData<Wd>,
}

/// Replacement for `crate::burn::optim::AdamConfig`.
#[derive(Debug, Clone)]
pub struct AdamConfig {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: Option<f32>,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
        }
    }
}

impl AdamConfig {
    /// Create a new Adam config with specified learning rate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate.
    pub fn with_beta_1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }
    /// Set beta2.
    pub fn with_beta_2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }
    /// Set epsilon.
    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }
    /// Set weight decay.
    pub fn with_weight_decay(mut self, wd: WeightDecayConfig) -> Self {
        self.weight_decay = Some(wd.penalty);
        self
    }
}

/// Replacement for `crate::burn::optim::decay::WeightDecayConfig`.
#[derive(Debug, Clone)]
pub struct WeightDecayConfig {
    /// L2 weight decay penalty.
    pub penalty: f32,
}

impl WeightDecayConfig {
    /// Create a new weight decay config.
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

/// Replacement for `crate::burn::optim::GradientsParams`.
///
/// In coeus, gradients are in `Var` leaves; this is a no-op placeholder.
#[derive(Debug, Default, Clone)]
pub struct GradientsParams;

impl GradientsParams {
    /// Create from a gradients map and a module.
    pub fn from_grads<B: Backend, M: Module<B>>(
        _grads: CoeusTensorGradients<B>,
        _module: &M,
    ) -> Self {
        Self
    }
}

/// Replacement for `crate::burn::optim::Optimizer`.
pub trait Optimizer<M: Module<B>, B: AutodiffBackend> {
    /// Type after one optimisation step.
    type Record;
    /// Perform one parameter update step.
    fn step(
        &mut self,
        lr: f64,
        module: M,
        grads: GradientsParams,
    ) -> M;
    /// Save optimizer state.
    fn to_record(&self) -> Self::Record;
    /// Load optimizer state.
    fn load_record(self, record: Self::Record) -> Self;
}

/// Replacement for `crate::burn::optim::adaptor::OptimizerAdaptor`.
///
/// A placeholder that wraps a coeus `SGD`/`Adam` optimizer.
pub struct OptimizerAdaptor<O, M, B: AutodiffBackend> {
    _optim: O,
    _module: std::marker::PhantomData<M>,
    _backend: std::marker::PhantomData<B>,
}

// ── LR Scheduler stubs ────────────────────────────────────────────────────────

/// Replacement for burn's `LrScheduler` trait.
pub trait LrScheduler: Send + Sync {
    /// Compute the next learning rate.
    fn step(&mut self) -> f64;
    /// Current learning rate.
    fn current(&self) -> f64;
}

/// Constant LR scheduler.
#[derive(Debug, Clone)]
pub struct ConstantLr {
    lr: f64,
}

impl ConstantLr {
    /// Create a constant learning rate scheduler.
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LrScheduler for ConstantLr {
    fn step(&mut self) -> f64 {
        self.lr
    }
    fn current(&self) -> f64 {
        self.lr
    }
}

/// Cosine annealing LR scheduler parameters.
#[derive(Debug, Clone)]
pub struct CosineAnnealingLrSchedulerConfig {
    /// Total number of decay steps.
    pub num_iters: usize,
    /// Minimum learning rate.
    pub min_lr: f64,
}

impl CosineAnnealingLrSchedulerConfig {
    /// Init the scheduler (returns `Self` since it IS the config here).
    pub fn init(self) -> CosineAnnealingLrScheduler {
        CosineAnnealingLrScheduler {
            config: self,
            step: 0,
            initial_lr: 1e-3,
        }
    }
}

/// Cosine annealing LR scheduler.
pub struct CosineAnnealingLrScheduler {
    config: CosineAnnealingLrSchedulerConfig,
    step: usize,
    initial_lr: f64,
}

impl LrScheduler for CosineAnnealingLrScheduler {
    fn step(&mut self) -> f64 {
        let t = (self.step as f64 / self.config.num_iters as f64)
            .min(1.0)
            * std::f64::consts::PI;
        self.step += 1;
        self.config.min_lr
            + 0.5 * (self.initial_lr - self.config.min_lr) * (1.0 + t.cos())
    }
    fn current(&self) -> f64 {
        let t = (self.step as f64 / self.config.num_iters as f64)
            .min(1.0)
            * std::f64::consts::PI;
        self.config.min_lr
            + 0.5 * (self.initial_lr - self.config.min_lr) * (1.0 + t.cos())
    }
}

/// Momentum config stub.
#[derive(Debug, Clone, Default)]
pub struct MomentumConfig {
    /// Momentum coefficient (default 0.9).
    pub momentum: f64,
    /// Dampening factor.
    pub dampening: f64,
    /// Whether to use Nesterov momentum.
    pub nesterov: bool,
}

// ── Config derive macro stub ──────────────────────────────────────────────────

/// Marker trait (replacement for `#[derive(crate::burn::config::Config)]`).
///
/// Config in burn adds `::default()` and `::load()` / `::save()` helpers.
/// The migration keeps `Default` and `Debug`; serialisation is unused in PINN code.
pub trait Config: Default + fmt::Debug + Clone {}

// Auto-implement for all `Default + Debug + Clone` types so existing structs
// that add `impl Config for MyConfig {}` are happy.
impl<T: Default + fmt::Debug + Clone> Config for T {}

// ── Activation shims ──────────────────────────────────────────────────────────

/// Re-export activation functions compatible with burn's `tensor::activation` module.
pub mod activation {
    use super::{Backend, Tensor};

    /// Tanh activation.
    pub fn tanh<B: Backend, const N: usize>(t: Tensor<B, N>) -> Tensor<B, N> {
        t.tanh()
    }

    /// ReLU activation.
    pub fn relu<B: Backend, const N: usize>(t: Tensor<B, N>) -> Tensor<B, N> {
        t.relu()
    }

    /// Sigmoid activation.
    pub fn sigmoid<B: Backend, const N: usize>(t: Tensor<B, N>) -> Tensor<B, N> {
        t.sigmoid()
    }
}

// ── Tensor marker types ───────────────────────────────────────────────────────

/// Replacement for `crate::burn::tensor::Bool` (integer / bool tensor marker).
///
/// In coeus these are just `u8` tensors; this module exposes a type alias so
/// existing code compiles without change.
pub type Bool = u8;

/// Replacement for `crate::burn::tensor::Int` (integer tensor marker).
pub type Int = i32;

// ── Devices ───────────────────────────────────────────────────────────────────

/// Replacement for `crate::burn::module::Devices` — always returns `[DefaultDevice]`.
pub fn default_devices() -> Vec<DefaultDevice> {
    vec![DefaultDevice]
}

// ── Additional NN layers ──────────────────────────────────────────────────────

/// Replacement for `crate::burn::nn::Gelu` activation layer.
#[derive(Debug, Clone, Default)]
pub struct Gelu;

impl Gelu {
    /// Create a new GELU activation layer.
    pub fn new() -> Self {
        Self
    }

    /// Apply GELU activation.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        Tensor::from_var(coeus_autograd::gelu(&input.inner))
    }
}

impl<B: Backend> Module<B> for Gelu {
    fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![]
    }
}

/// Configuration for `LayerNorm`.
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Number of features.
    pub d_model: usize,
    /// Epsilon for numerical stability.
    pub epsilon: f32,
}

impl LayerNormConfig {
    /// Create a new layer norm config.
    pub fn new(d_model: usize) -> Self {
        Self { d_model, epsilon: 1e-5 }
    }

    /// Init the layer norm.
    pub fn init<B: Backend + coeus_ops::BackendOps<f32>>(&self, _device: &DefaultDevice) -> LayerNorm<B> {
        LayerNorm {
            d_model: self.d_model,
            epsilon: self.epsilon,
            _backend: std::marker::PhantomData,
        }
    }
}

/// Replacement for `crate::burn::nn::LayerNorm`.
pub struct LayerNorm<B: Backend> {
    d_model: usize,
    epsilon: f32,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> fmt::Debug for LayerNorm<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LayerNorm(d_model={})", self.d_model)
    }
}

impl<B: Backend> Clone for LayerNorm<B> {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            epsilon: self.epsilon,
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + coeus_ops::BackendOps<f32>> Module<B> for LayerNorm<B> {
    fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![]
    }
}

impl<B: Backend + coeus_ops::BackendOps<f32>> LayerNorm<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Apply layer normalization.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = coeus_nn::normalization::layer_norm(
            &input.inner,
            self.d_model,
            None,
            None,
            self.epsilon as f64,
        );
        Tensor::from_var(out)
    }
}

// ── Additional optimizers ─────────────────────────────────────────────────────

/// Replacement for `crate::burn::optim::AdamW` (Adam with weight decay).
///
/// Alias: same as `Adam` for migration purposes.
pub type AdamW<B> = Adam<B>;

/// Config for AdamW.
pub type AdamWConfig = AdamConfig;

/// Replacement for `crate::burn::optim::Sgd`.
///
/// Stub: wraps learning rate only (no momentum tracking during migration).
#[derive(Debug, Clone)]
pub struct Sgd<B: Backend, Wd = ()> {
    /// Learning rate.
    pub lr: f64,
    _backend: std::marker::PhantomData<B>,
    _wd: std::marker::PhantomData<Wd>,
}

impl<B: Backend, Wd> Sgd<B, Wd> {
    /// Create an SGD optimizer stub.
    pub fn new(lr: f64) -> Self {
        Self { lr, _backend: std::marker::PhantomData, _wd: std::marker::PhantomData }
    }
}

/// Config for SGD.
#[derive(Debug, Clone)]
pub struct SgdConfig {
    /// Learning rate.
    pub lr: f64,
    /// Momentum config.
    pub momentum: Option<MomentumConfig>,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: None,
        }
    }
}

impl SgdConfig {
    /// Create new SGD config.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set momentum config.
    pub fn with_momentum(mut self, m: MomentumConfig) -> Self {
        self.momentum = Some(m);
        self
    }
}

// ── Record module stub ────────────────────────────────────────────────────────

/// Replacement for `crate::burn::record` — serialisation stubs.
///
/// PINN training code uses these for model checkpointing. The stubs allow
/// compilation; actual checkpoint support can be wired later.
pub mod record {
    /// Stub for `BinFileRecorder`.
    #[derive(Debug, Clone, Default)]
    pub struct BinFileRecorder<P = FullPrecisionSettings>(std::marker::PhantomData<P>);

    impl<P> BinFileRecorder<P> {
        /// Create a new recorder.
        pub fn new() -> Self {
            Self(std::marker::PhantomData)
        }
    }

    /// Full-precision serialisation settings (stub).
    #[derive(Debug, Clone, Default)]
    pub struct FullPrecisionSettings;

    /// Half-precision serialisation settings (stub).
    #[derive(Debug, Clone, Default)]
    pub struct HalfPrecisionSettings;
}

// ── Module manual-impl helper macro ───────────────────────────────────────────

/// Macro to generate a minimal `Module<B>` impl for network structs.
///
/// Usage (inside the module where the struct is defined):
/// ```ignore
/// impl_module! {
///     MyNetwork<B: Backend + BackendOps<f32>> {
///         layer_a, layer_b
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_module {
    ( $struct:ident < $B:ident : $bound:path > { $( $field:ident ),* $(,)? } ) => {
        impl<$B: $bound> $crate::burn::module::Module<$B> for $struct<$B> {
            fn parameters(&self) -> Vec<coeus_autograd::Var<f32, $B>> {
                let mut params: Vec<coeus_autograd::Var<f32, $B>> = vec![];
                $(
                    params.extend(
                        $crate::burn::module::Module::<$B>::parameters(&self.$field)
                    );
                )*
                params
            }
        }
    };
    // Variant for Vec<Layer<B>> fields
    ( $struct:ident < $B:ident : $bound:path > {
        $( $field:ident ),*  ;  $( $vec_field:ident ),* $(,)?
    } ) => {
        impl<$B: $bound> $crate::burn::module::Module<$B> for $struct<$B> {
            fn parameters(&self) -> Vec<coeus_autograd::Var<f32, $B>> {
                let mut params: Vec<coeus_autograd::Var<f32, $B>> = vec![];
                $(
                    params.extend(
                        $crate::burn::module::Module::<$B>::parameters(&self.$field)
                    );
                )*
                $(
                    for layer in &self.$vec_field {
                        params.extend(
                            $crate::burn::module::Module::<$B>::parameters(layer)
                        );
                    }
                )*
                params
            }
        }
    };
}
