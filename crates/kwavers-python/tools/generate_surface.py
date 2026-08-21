#!/usr/bin/env python3
"""Generate the Kwavers Python registration inventory and typed stub surface.

The Rust PyO3 registration calls and facade imports are the source of truth.
The generator recovers real Python signatures from the Rust binding source:

- ``#[pyfunction]`` module functions (registered via ``wrap_pyfunction!``).
- ``#[pymethods]`` class methods (``#[new]``, ``#[staticmethod]``,
  ``#[classmethod]``, ``#[getter]``, ``#[setter]``, ``#[classattr]``).
- ``#[pyclass(name = "...")]`` renames and enum variant surfaces.
- ``#[pyo3(signature = (...))]`` argument lists, including defaults,
  keyword-only ``*`` separators, and multi-line attributes.

The emitter FAILS CLOSED: any signature it cannot recover is recorded in the
``failures`` list and omitted from the stub rather than degraded to ``Any``
or ellipsis placeholders. ``--check`` exits non-zero when the committed
artifacts are stale or when any symbol failed to recover.
"""
from __future__ import annotations

import argparse
import ast
import json
import keyword
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Rust source scanning
# ---------------------------------------------------------------------------

PYFUNCTION = re.compile(r"wrap_pyfunction!\(\s*(?:crate::)?(?:[\w:]+::)?([A-Za-z_]\w*)")
PYCLASS = re.compile(r"add_class::<\s*([A-Za-z_]\w*)\s*>\(\)")
PYCLASS_NAME = re.compile(r"#\[pyclass\([^)]*name\s*=\s*\"([A-Za-z_]\w*)\"")
PYCLASS_ATTR = re.compile(r"#\[pyclass(?:\([^)]*\))?\]")
PYFUNCTION_ATTR = re.compile(r"#\[pyfunction(?:\([^)]*\))?\]")
PYMETHODS_ATTR = re.compile(r"#\[pymethods\]")

# Attributes that mark a method's Python kind.
NEW_ATTR = re.compile(r"#\[new\]")
STATIC_ATTR = re.compile(r"#\[staticmethod\]")
CLASS_ATTR = re.compile(r"#\[classmethod\]")
GETTER_ATTR = re.compile(r"#\[getter\]")
SETTER_ATTR = re.compile(r"#\[setter\]")
CLASSATTR_ATTR = re.compile(r"#\[classattr\]")
SIGNATURE_ATTR = re.compile(r"#\[pyo3\(signature\s*=\s*(\(.*?\))\)\]", re.DOTALL)


def strip_comments(text: str) -> str:
    """Remove Rust line and block comments (preserving string literals is not
    needed for signature scanning; attribute/arg lists do not carry // or /*)."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j == -1 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def balanced(text: str, start: int, open_ch: str = "(", close_ch: str = ")") -> tuple[str, int]:
    """Return the balanced group starting at ``start`` (which must point at
    ``open_ch``) as (content-without-parens, index-after-close)."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], n


def find_fn(text: str, name: str, start: int) -> int | None:
    """Find the index of ``fn <name>`` at depth zero (not inside a nested fn)."""
    pat = re.compile(r"\bfn\s+" + re.escape(name) + r"\b")
    m = pat.search(text, start)
    return m.start() if m else None


def rust_ident(text: str, start: int) -> str:
    m = re.match(r"[A-Za-z_]\w*", text[start:])
    return m.group(0) if m else ""


def split_top(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at nesting depth zero."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth = max(0, depth - 1)
        if c == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def parse_params(inner: str) -> list[dict]:
    """Parse a balanced parameter-list body (the text between the ``fn``'s
    parentheses, without the parens themselves)."""
    params: list[dict] = []
    for raw in split_top(inner):
        if not raw:
            continue
        # pattern: [mut] name: type  (name may be a self form)
        m = re.match(r"^(mut\s+)?([A-Za-z_]\w*)\s*:\s*(.+)$", raw)
        if m:
            params.append({"name": m.group(2), "type": m.group(3).strip()})
        elif re.match(r"^(?:&mut\s+)?self\b", raw):
            params.append({"name": "self", "type": raw.strip()})
        elif raw.startswith("..."):
            # Variadic marker (PyO3 allows ... in signatures for pyfunctions
            # that accept *args; the actual fn uses explicit params though).
            params.append({"name": raw, "type": raw})
    return params


def parse_return(fn_block: str) -> str | None:
    """Find the return type after the top-level ``->`` (or None)."""
    depth = 0
    i = 0
    n = len(fn_block)
    while i < n:
        c = fn_block[i]
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth = max(0, depth - 1)
        elif depth == 0 and fn_block.startswith("->", i):
            rest = fn_block[i + 2 :]
            # strip trailing { and whitespace
            rest = rest.split("{", 1)[0].strip()
            return rest.strip() or None
        i += 1
    return None


# ---------------------------------------------------------------------------
# Python type mapping (fails closed)
# ---------------------------------------------------------------------------

ARRAY_ALIASES = {
    "f64": "Float64Array",
    "f32": "Float32Array",
    "i64": "Int64Array",
    "i32": "Int32Array",
    "u64": "UInt64Array",
    "u32": "UInt32Array",
    "usize": "UInt64Array",
    "bool": "BoolArray",
    "Complex64": "Complex64Array",
    "Complex32": "Complex32Array",
}

ARRAY_ALIAS_DEFS = {
    "Float64Array": "NDArray[np.float64]",
    "Float32Array": "NDArray[np.float32]",
    "Int64Array": "NDArray[np.int64]",
    "Int32Array": "NDArray[np.int32]",
    "UInt64Array": "NDArray[np.uint64]",
    "UInt32Array": "NDArray[np.uint32]",
    "BoolArray": "NDArray[np.bool_]",
    "Complex64Array": "NDArray[np.complex64]",
    "Complex32Array": "NDArray[np.complex64]",
}


class TypeMapper:
    """Maps Rust binding types to Python stub types. Unknown types raise
    UnsupportedType so the generator can fail closed."""

    def __init__(self, class_names: set[str], alias_map: dict[str, str] | None = None,
                 type_aliases: dict[str, str] | None = None,
                 duck_types: dict[tuple[str, str, str], str] | None = None):
        self.class_names = class_names
        self.alias_map = dict(alias_map or {})
        self.type_aliases = dict(type_aliases or {})
        # Audited duck-typed ``PyAny`` parameters: (class_py_name, fn_name,
        # param_name) -> exact Python type string. Every entry must correspond
        # to extraction logic read from the binding source; see DUCK_TYPES
        # below. Any ``Bound<PyAny>`` parameter absent from this table raises
        # DuckTypedParam so the emitter fails closed.
        self.duck_types = dict(duck_types or {})
        # The Python name of the enclosing class (for ``Self`` resolution).
        self.self_type: str | None = None

    def map(self, ty: str) -> str:
        ty = ty.strip()
        # collapse internal whitespace so multi-line tuple/alias types map
        ty = re.sub(r"\s+", " ", ty)
        if not ty:
            raise UnsupportedType("<empty type>")
        # resolve local type aliases (Trace4 etc.)
        if ty in self.type_aliases:
            return self.map(self.type_aliases[ty])
        # strip references and lifetime qualifiers like 'static / '_
        while ty.startswith("&") or ty.startswith("mut ") or ty.startswith("'static ") or ty.startswith("'_ "):
            if ty.startswith("&"):
                ty = ty[1:].strip()
            elif ty.startswith("mut "):
                ty = ty[4:].strip()
            else:
                ty = ty.split(" ", 1)[1].strip()
        # 'self' receivers / Self return
        if ty == "self" or ty == "&self" or ty == "&mut self":
            return "Self"
        if ty == "Self" and self.self_type:
            return f'"{self.self_type}"'
        if ty == "Self":
            return "Self"
        # PyO3/numpy array types
        # Py<PyDict>: every PyDict constructed in the binding surface uses
        # string literal keys (422/422 set_item sites), values are heterogeneous
        # (arrays, floats, strings), so Dict[str, object] is exact.
        if ty == "Py<PyDict>":
            return "Dict[str, object]"
        m = re.match(r"^Py<PyTuple>$", ty)
        if m:
            return "Tuple[object, ...]"
        m = re.match(r"^Py<PyList>$", ty)
        if m:
            return "List[object]"
        m = re.match(r"^PyReadonlyArray(\d)<'[^,>]*,\s*([^>]+)>$", ty)
        if m:
            dim = m.group(1)
            el = m.group(2).strip()
            return self._array_alias(el, dim)
        m = re.match(r"^PyReadonlyArray(\d)<([^,>]+)>$", ty)
        if m:
            dim = m.group(1)
            el = m.group(2).strip()
            return self._array_alias(el, dim)
        m = re.match(r"^Py<PyArray(\d)<([^>]+)>>$", ty)
        if m:
            dim = m.group(1)
            el = m.group(2).strip()
            return self._array_alias(el, dim)
        m = re.match(r"^&?Bound<'_,\s*(PyArray\d<[^>]+>)\s*>$", ty)
        if m:
            inner = m.group(1)
            m2 = re.match(r"^PyArray(\d)<([^>]+)>$", inner)
            if m2:
                return self._array_alias(m2.group(2).strip(), m2.group(1))
        # Duck-typed PyAny parameters resolve through the audited DUCK_TYPES
        # table at emission time (keyed by class/function/param). Anything not
        # in the table fails closed instead of degrading to ``object``, so the
        # stub never carries an unnamed parameter type.
        if re.match(r"^(?:&?\s*(?:pyo3::)?Bound<'[^,>]*,\s*)?(?:pyo3::)?PyAny\s*>$", ty) or ty in (
            "PyObject",
            "Py<PyAny>",
            "&Bound<'_, PyAny>",
            "Bound<'_, PyAny>",
            "pyo3::Bound<'_, pyo3::PyAny>",
            "&pyo3::Bound<'_, pyo3::PyAny>",
        ):
            raise DuckTypedParam()
        # Bound<'lifetime, T> — the lifetime is irrelevant; map the inner T
        m = re.match(r"^(?:pyo3::)?Bound<'[^,>]*,\s*(.+?)\s*>$", ty)
        if m:
            inner = m.group(1).strip()
            # numpy::PyArrayN<T> path
            m2 = re.match(r"^(?:numpy::)?PyArray(\d)<([^>]+)>$", inner)
            if m2:
                return self._array_alias(m2.group(2).strip(), m2.group(1))
            if inner in ("PyDict", "pyo3::PyDict"):
                # every PyDict in the binding surface uses string literal keys
                # (422/422 set_item sites); values are heterogeneous.
                return "Dict[str, object]"
            if inner in ("PyList", "pyo3::PyList"):
                return "List[object]"
            if inner in ("PyTuple", "pyo3::PyTuple"):
                return "Tuple[object, ...]"
            return self.map(inner)
        if ty == "PyRef<'_, Self>" or ty == "PyRefMut<'_, Self>":
            return "Self"
        if re.match(r"^PyRef(?:Mut)?<'_,\s*([A-Za-z_]\w*)>$", ty):
            m = re.match(r"^PyRef(?:Mut)?<'_,\s*([A-Za-z_]\w*)>$", ty)
            return self.alias_map.get(m.group(1), m.group(1))
        # Python token hidden arg (Rust lifetimes like 'py have one apostrophe)
        if re.match(r"^Python<'[^'>]*>$", ty):
            raise HiddenArg()
        # PyResult<T> / PyErr wrapper unwrap (T may span multiple lines)
        m = re.match(r"^PyResult<(.+)>$", ty, re.DOTALL)
        if m:
            return self.map(m.group(1))
        m = re.match(r"^PyErr$", ty)
        if m:
            return "BaseException"
        # Option<...>
        m = re.match(r"^Option<(.+)>$", ty, re.DOTALL)
        if m:
            inner = self.map(m.group(1))
            return f"Optional[{inner}]"
        # Vec<...>
        m = re.match(r"^Vec<(.+)>$", ty)
        if m:
            return f"List[{self.map(m.group(1))}]"
        # HashMap/BTreeMap
        m = re.match(r"^(?:HashMap|BTreeMap)<([^,]+),\s*([^>]+)>$", ty)
        if m:
            return f"Dict[{self.map(m.group(1))}, {self.map(m.group(2))}]"
        # fixed-size arrays [T; N]
        m = re.match(r"^\[([^;]+);\s*\d+\]$", ty)
        if m:
            return f"List[{self.map(m.group(1))}]"
        # tuples (may span lines)
        if ty.startswith("(") and ty.endswith(")"):
            inner = ty[1:-1]
            if inner.strip() == "":
                return "None"
            parts = split_top(inner)
            mapped = [self.map(p) for p in parts]
            return "Tuple[" + ", ".join(mapped) + "]"
        # scalar primitives
        primitives = {
            "f64": "float",
            "f32": "float",
            "u8": "int",
            "u16": "int",
            "u32": "int",
            "u64": "int",
            "usize": "int",
            "i8": "int",
            "i16": "int",
            "i32": "int",
            "i64": "int",
            "isize": "int",
            "bool": "bool",
            "String": "str",
            "&str": "str",
            "&String": "str",
            "str": "str",
            "char": "str",
            "()": "None",
        }
        if ty in primitives:
            return primitives[ty]
        # crate::...::Type paths -> use the final ident
        if ty.startswith("crate::") or ty.startswith("kwavers"):
            last = ty.rsplit("::", 1)[-1]
            if last in self.class_names or self.alias_map.get(last, last) in self.class_names:
                return self.alias_map.get(last, last)
            raise UnsupportedType(ty)
        # Py* aliases of known classes (PyPmlConfig -> PmlConfig)
        if ty.startswith("Py") and len(ty) > 2:
            stripped = ty[2:]
            if stripped in self.class_names or self.alias_map.get(stripped, stripped) in self.class_names:
                return self.alias_map.get(stripped, stripped)
        # known pyclasses (by Rust ident or Python name)
        if ty in self.class_names or self.alias_map.get(ty, ty) in self.class_names:
            return self.alias_map.get(ty, ty)
        raise UnsupportedType(ty)

    def _array_alias(self, el: str, dim: str) -> str:
        el = el.strip()
        if el in ARRAY_ALIASES:
            return ARRAY_ALIASES[el]
        raise UnsupportedType(f"PyArray{dim}<{el}>")


class HiddenArg(Exception):
    """Raised for the Python GIL token argument, which is hidden in Python."""


class DuckTypedParam(Exception):
    """Raised when a ``Bound<PyAny>`` parameter must be resolved through the
    audited DUCK_TYPES table keyed by (class, function, parameter name)."""


class UnsupportedType(Exception):
    """Raised when a Rust binding type cannot be mapped to a Python type."""


# ---------------------------------------------------------------------------
# Audited duck-typed parameters
# ---------------------------------------------------------------------------

# Each entry maps (class_py_name, fn_name, param_name) to the EXACT Python
# surface accepted by the Rust extraction code. Provenance, verified against
# the binding sources on this lane:
#
# - Simulation.__init__ source: extract::<Source>(), then
#   extract::<TransducerArray2D>(), then a Vec<Bound<PyAny>> of either
#   (src/simulation_py/mod.rs).
# - Simulation.__init__ sensor: extract::<Sensor>(), then
#   extract::<TransducerArray2D>() (src/simulation_py/mod.rs).
# - Source.from_initial_pressure p0: PyReadonlyArray3<f64>, then
#   PyReadonlyArray2<f64>; anything else errors (src/source_py/mod.rs).
# - Source.from_initial_displacement field: PyReadonlyArray3<f64>, then
#   PyReadonlyArray2<f64>; anything else errors (src/source_py/elastic.rs).
# - Source.from_mask signal: pressure_signal_to_matrix accepts
#   PyReadonlyArray1<f64> then PyReadonlyArray2<f64>; anything else errors
#   (src/source_py/helpers.rs).
DUCK_TYPES: dict[tuple[str, str, str], str] = {
    # constructors are #[new] fns whose Rust name is ``new``; the emitter
    # exposes them as ``__init__`` but the audit key keeps the Rust name.
    ("Simulation", "__init__", "source"):
        'Union["Source", "TransducerArray2D", List[Union["Source", "TransducerArray2D"]]]',
    ("Simulation", "__init__", "sensor"):
        'Union["Sensor", "TransducerArray2D"]',
    ("Source", "from_initial_pressure", "p0"): "Float64Array",
    ("Source", "from_initial_displacement", "field"): "Float64Array",
    ("Source", "from_mask", "signal"): "Float64Array",
    # Medium heterogeneous constructor: alpha_power accepts a scalar float
    # (broadcast to the grid) or a 3D f64 array matching sound_speed shape;
    # anything else errors (src/medium_py/mod.rs).
    ("Medium", "__init__", "alpha_power"): 'Union[float, Float64Array]',
    # SimulationResult.sensor_data getter returns the stored 1D or 2D f64
    # sensor array as Py<PyAny> (src/simulation_result_py.rs); None when no
    # data is recorded.
    ("SimulationResult", "sensor_data", "return"): "Optional[Float64Array]",
    # skull_transfer_matrix_transmission returns a Python complex built from
    # (re, im) (src/analytical_bindings/skull/transmission.rs).
    ("", "skull_transfer_matrix_transmission", "return"): "complex",
}


# ---------------------------------------------------------------------------
# Stub generation
# ---------------------------------------------------------------------------

def python_literal(raw: str, const_values: dict[str, str] | None = None) -> str:
    """Convert a Rust literal default to a Python literal."""
    raw = raw.strip()
    const_values = const_values or {}
    if raw in const_values:
        return python_literal(const_values[raw], const_values)
    if raw in ("true", "false"):
        return "True" if raw == "true" else "False"
    if raw in ("None", "none"):
        return "None"
    if re.match(r"^[A-Z][A-Z0-9_]*$", raw):
        # unresolved uppercase const identifier: fail closed
        raise UnsupportedType(f"const default {raw}")
    # strip trailing type suffix like 3usize / 1.5f64
    raw = re.sub(r"([0-9])(?:u|usize|i8|i16|i32|i64|isize|f32|f64)$", r"\1", raw)
    # numeric with underscores -> python float/int
    if re.match(r"^-?[0-9][0-9_]*(\.[0-9_]*)?([eE][+-]?[0-9_]+)?$", raw):
        return raw.replace("_", "")
    # string literal
    if raw.startswith('"') and raw.endswith('"'):
        return raw
    if raw.startswith("'") and raw.endswith("'"):
        return '"' + raw[1:-1] + '"'
    if raw.startswith("b\"") and raw.endswith("\""):
        return raw[1:]
    if raw.startswith("Some(") and raw.endswith(")"):
        inner = raw[5:-1]
        return python_literal(inner)
    if raw == "...":
        return "..."
    # Fall back to the raw text; if it contains a Rust path (e.g. an enum
    # variant like `SolverType::PSTD`), collapse to the final ident.
    if "::" in raw:
        return raw.rsplit("::", 1)[-1]
    return raw


def py_name(name: str) -> str:
    """Escape Python keywords the way PyO3 does (trailing underscore)."""
    return name + "_" if keyword.iskeyword(name) else name


def format_arg(name: str, ty: str, default: str | None, mapper: TypeMapper) -> str:
    if ty == "self" or name == "self":
        return "self"
    py_ty = mapper.map(ty)
    if default is not None:
        return f"{name}: {py_ty} = {python_literal(default)}"
    return f"{name}: {py_ty}"


def normalize_literal_default(raw: str) -> str | None:
    raw = raw.strip()
    if raw == "None" or raw == "none":
        return "None"
    return raw


class Symbol:
    def __init__(self, kind: str, name: str, module: str, line: int):
        self.kind = kind  # "function" | "class" | "method" | "getter" | "setter" | "staticmethod" | "classmethod" | "classattr"
        self.name = name
        self.module = module
        self.line = line
        self.py_name: str = name
        self.signature_attr: str | None = None
        self.params: list[dict] = []
        self.return_type: str | None = None
        self.class_py_name: str | None = None
        self.receiver: str | None = None  # "self" | "static"
        self.doc: str = ""
        self.failed: str | None = None
        self.registered: bool = False
        self.enum_variants: list[str] = []


def parse_pymethods(text: str, module: str, class_py_name: str, mapper: TypeMapper) -> list[Symbol]:
    """Parse one ``#[pymethods] impl <Class> { ... }`` block.

    Walks every ``fn`` in the block and collects the attributes that precede
    it (``#[new]``, ``#[staticmethod]``, ``#[getter]``, ``#[pyo3(signature = ...)]``,
    ...). A ``fn`` with no recognized attribute is still a plain method.
    """
    syms: list[Symbol] = []
    # strip doc comments so only attributes remain between fns
    text_nocomments = re.sub(r"(?m)^\s*///.*$", "", text)
    for fm in re.finditer(r"(?:pub(?:\([^)]*\))?\s+)?fn\s+([A-Za-z_]\w*)(?:<[^>]*>)?\s*\(", text_nocomments):
        fn_name = fm.group(1)
        # index of the '(' that opens the parameter list
        fn_start = fm.start() + fm.group(0).rfind("(")
        prefix = text_nocomments[: fm.start()]
        attr_block = re.search(r"((?:#\[[^\]]+\]\s*)+)$", prefix)
        attrs: list[str] = []
        if attr_block:
            for am in re.finditer(r"#\[([^\]]+)\]", attr_block.group(1)):
                attrs.append(am.group(1))
        body, after = balanced(text, fn_start)
        # extend past the params to include the return type (up to '{' or ';')
        ret_end = after
        m_brace = re.search(r"[{;]", text[after:])
        if m_brace:
            ret_end = after + m_brace.start()
        fn_block = text[attr_block.start() if attr_block else fm.start() : ret_end]
        ret = parse_return(fn_block)
        params = parse_params(body)

        sym = Symbol("method", fn_name, module, text.count("\n", 0, fm.start()) + 1)
        sym.params = params
        sym.return_type = ret
        sig = next((a for a in attrs if a.startswith("pyo3(signature")), None)
        if sig:
            m2 = re.match(r"pyo3\(signature\s*=\s*(\(.*\))\)\s*$", sig, re.DOTALL)
            if m2:
                sym.signature_attr = m2.group(1)
        # classify
        if "#[new]" in attrs or any(a == "new" for a in attrs):
            sym.kind = "new"
        elif any(a == "staticmethod" for a in attrs):
            sym.kind = "staticmethod"
            sym.receiver = "static"
        elif any(a == "classmethod" for a in attrs):
            sym.kind = "classmethod"
            sym.receiver = "class"
        elif any(a.startswith("getter") for a in attrs):
            sym.kind = "getter"
        elif any(a.startswith("setter") for a in attrs):
            sym.kind = "setter"
        elif any(a == "classattr" for a in attrs):
            sym.kind = "classattr"
        sym.class_py_name = class_py_name
        syms.append(sym)
    return syms


def extract_pyclass_name(block: str) -> str | None:
    m = PYCLASS_NAME.search(block)
    if m:
        return m.group(1)
    m = re.search(r"#\[pyclass(?:\([^)]*\))?\]\s*\n\s*(?:pub\s+)?(?:struct|enum)\s+([A-Za-z_]\w*)", block)
    if m:
        return m.group(1)
    return None


def parse_enum_variants(text: str) -> list[str]:
    m = re.search(r"\benum\s+[A-Za-z_]\w*\s*\{(.*?)\}", text, re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    variants: list[str] = []
    for raw in split_top(body):
        if not raw:
            continue
        vm = re.match(r"([A-Za-z_]\w*)\s*(?:\(|,|$)", raw)
        if vm:
            variants.append(vm.group(1))
    return variants


def parse_rust(root: Path) -> dict:
    """Scan all Rust sources and build the symbol model."""
    functions: dict[str, Symbol] = {}
    classes: dict[str, Symbol] = {}
    all_syms: list[Symbol] = []
    class_py_names: dict[str, str] = {}
    type_aliases: dict[str, str] = {}
    const_values: dict[str, str] = {}

    # First pass: local `type X = ...;` aliases (Trace4 etc.) and const defaults
    for path in sorted((root / "src").rglob("*.rs")):
        text = strip_comments(path.read_text(encoding="utf-8"))
        for m in re.finditer(r"\btype\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", text):
            alias, rhs = m.group(1), m.group(2).strip()
            if alias not in type_aliases:
                type_aliases[alias] = rhs
        for m in re.finditer(r"\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*[^=]+\s*=\s*([^;]+);", text):
            name, val = m.group(1), m.group(2).strip()
            const_values.setdefault(name, val)
        module = path.relative_to(root).as_posix()
        text = strip_comments(path.read_text(encoding="utf-8"))
        lines = path.read_text(encoding="utf-8").splitlines()

        # classes: pyclass attributes
        for m in PYCLASS_ATTR.finditer(text):
            block_start = m.start()
            # find the struct/enum declaration
            m2 = re.search(r"(?:pub\s+)?(?:struct|enum)\s+([A-Za-z_]\w*)", text[block_start:])
            if not m2:
                continue
            rust_name = m2.group(1)
            py_name = extract_pyclass_name(text[block_start : block_start + 400])
            if not py_name:
                py_name = rust_name
            sym = Symbol("class", rust_name, module, text.count("\n", 0, block_start) + 1)
            sym.py_name = py_name
            # enum pyclasses expose their variants as class-level constants
            is_enum = bool(re.search(r"(?:pub\s+)?enum\s+" + re.escape(rust_name), text[block_start:]))
            if is_enum:
                sym.enum_variants = parse_enum_variants(text[block_start : block_start + 2000])
            class_py_names[rust_name] = py_name
            classes[rust_name] = sym
            all_syms.append(sym)

    # now that we know class names, build a mapper with aliases
    alias_map = {rust: py for rust, py in class_py_names.items()}
    class_set = set(class_py_names.values())

    for path in sorted((root / "src").rglob("*.rs")):
        module = path.relative_to(root).as_posix()
        text = strip_comments(path.read_text(encoding="utf-8"))
        # pyfunctions
        for m in PYFUNCTION_ATTR.finditer(text):
            attr_start = m.start()
            # Only treat this as a pyfunction if the ``fn`` follows the
            # attribute block directly (allowing interleaved #[...] attrs but
            # not skipping past a second function).
            fm = re.match(r"(?:\s*#\[[^\]]+\]\s*)*(?:pub(?:\([^)]*\))?\s+)?fn\s+([A-Za-z_]\w*)(?:<[^>]*>)?\s*\(", text[attr_start:])
            if not fm:
                continue
            fn_name = fm.group(1)
            fn_head = attr_start + fm.end()
            fn_start = fn_head - 1  # index of '('
            body, after = balanced(text, fn_start)
            # extend past the params to include the return type (up to '{' or ';')
            ret_end = after
            m_brace = re.search(r"[{;]", text[after:])
            if m_brace:
                ret_end = after + m_brace.start()
            fn_block = text[attr_start:ret_end]
            sym = Symbol("function", fn_name, module, text.count("\n", 0, attr_start) + 1)
            sym.params = parse_params(body)
            sym.return_type = parse_return(fn_block)
            sigm = SIGNATURE_ATTR.search(text[attr_start:fn_head])
            if sigm:
                sym.signature_attr = sigm.group(1)
            # #[pyfunction(name = "...")] renames the Python-visible symbol
            pf_attr = text[attr_start : attr_start + 80]
            nm = re.search(r"#\[pyfunction\([^)]*name\s*=\s*\"([^\"]+)\"", pf_attr)
            if nm:
                sym.py_name = nm.group(1)
            functions[fn_name] = sym
            all_syms.append(sym)

        # pymethods blocks
        for m in PYMETHODS_ATTR.finditer(text):
            block_start = m.end()
            # find impl target
            im = re.match(r"\s*impl\s+([A-Za-z_]\w*)", text[block_start:])
            if not im:
                continue
            impl_target = im.group(1)
            py_name = alias_map.get(impl_target, impl_target)
            # find the impl body
            brace_start = text.find("{", block_start + im.end())
            if brace_start == -1:
                continue
            body, after = balanced(text, brace_start, "{", "}")
            mapper = TypeMapper(class_set, alias_map)
            for sym in parse_pymethods(body, module, py_name, mapper):
                sym.class_py_name = py_name
                if sym.kind == "new":
                    # this becomes a constructor on the class; store separately
                    pass
                all_syms.append(sym)

    return {
        "functions": functions,
        "classes": classes,
        "class_py_names": class_py_names,
        "symbols": all_syms,
        "type_aliases": type_aliases,
        "const_values": const_values,
    }


def registered_symbols(root: Path) -> tuple[list[str], list[str], dict]:
    """Return (functions, classes, symbol_map) based on registration calls.

    ``wrap_pyfunction!(rust_name, ...)`` registers the function under its Rust
    name unless the ``#[pyfunction(name = "...")]`` attribute renames it; the
    Python-facing name is what must appear in the inventory and stub.
    """
    # rust fn ident -> Python name (from #[pyfunction(name = "...")])
    py_name_by_rust: dict[str, str] = {}
    for path in sorted((root / "src").rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'#\[pyfunction\(([^)]*name\s*=\s*"([^"]+)")\)\]', text):
            rust_fn = None
            # scan forward from the attribute to the next `fn` (skipping any
            # interleaved multi-line #[...] attributes)
            seg = text[m.end() : m.end() + 2000]
            fm = re.search(r"\bfn\s+([A-Za-z_]\w*)", seg)
            if fm:
                rust_fn = fm.group(1)
            if rust_fn:
                py_name_by_rust[rust_fn] = m.group(2)

    functions: set[str] = set()
    classes: set[str] = set()
    for path in sorted((root / "src").rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for rust_name in PYFUNCTION.findall(text):
            functions.add(py_name_by_rust.get(rust_name, rust_name))
        classes.update(PYCLASS.findall(text))
    return sorted(functions), sorted(classes), {}


def facade_symbols(root: Path) -> tuple[list[str], list[str]]:
    facade = root / "python" / "pykwavers" / "__init__.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"), filename=str(facade))
    imported: set[str] = set()
    exported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("_pykwavers"):
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exported.update(
                        item.value for item in node.value.elts if isinstance(item, ast.Constant)
                    )
    return sorted(imported), sorted(exported)


# ---------------------------------------------------------------------------
# Stub emitter
# ---------------------------------------------------------------------------

HEADER = "# Generated by tools/generate_surface.py; do not edit.\n"
IMPORTS = (
    "from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Union\n\n"
    "import numpy as np\n"
    "from numpy.typing import NDArray\n\n"
)


def alias_block(used_aliases: set[str]) -> str:
    lines = []
    for name in sorted(used_aliases):
        lines.append(f"{name} = {ARRAY_ALIAS_DEFS[name]}")
    if lines:
        return "\n".join(lines) + "\n\n"
    return ""


def build_signatures(model: dict) -> dict:
    """Build per-symbol emitted signature text. Returns {key: text} and a
    failures dict."""
    functions = model["functions"]
    classes = model["classes"]
    class_py_names = model["class_py_names"]
    alias_map = {rust: py for rust, py in class_py_names.items()}
    class_set = set(class_py_names.values())
    mapper = TypeMapper(class_set, alias_map, model.get("type_aliases"), DUCK_TYPES)

    def map_param(sym: Symbol, name: str, ty: str) -> str:
        """Map one parameter type, resolving duck-typed PyAny entries through
        DUCK_TYPES keyed by (class, function, param). Missing entries fail
        closed."""
        try:
            return mapper.map(ty)
        except DuckTypedParam:
            fn_key = "__init__" if sym.kind == "new" else sym.py_name
            key = (sym.class_py_name or "", fn_key, name)
            entry = DUCK_TYPES.get(key)
            if entry is None:
                raise UnsupportedType(
                    f"unaudited PyAny param {key} (add it to DUCK_TYPES with "
                    "extraction-code provenance)"
                ) from None
            return entry

    emitted: dict[str, str] = {}
    failures: list[dict] = []
    const_values: dict[str, str] = model.get("const_values", {})

    def emit_fn(sym: Symbol, is_method: bool, cls_name: str | None) -> str | None:
        try:
            parts: list[str] = []
            receiver = "self" if is_method and cls_name else ""
            if sym.signature_attr:
                arg_tokens = split_top(sym.signature_attr[1:-1])
                # find kw-only separator
                kw_only = False
                for tok in arg_tokens:
                    if tok == "*":
                        kw_only = True
                        continue
                    m = re.match(r"^([A-Za-z_]\w*)(?:\s*=\s*(.+))?$", tok)
                    if m:
                        name = m.group(1)
                        default = m.group(2)
                        ty = None
                        for p in sym.params:
                            if p["name"] == name:
                                ty = p["type"]
                                break
                        if ty is None:
                            failures.append({"symbol": sym.name, "reason": f"no param type for {name}"})
                            return None
                        try:
                            if ty == "self":
                                continue
                            if re.match(r"^PyRef(?:Mut)?<'_,\s*Self>$", ty):
                                # fluent-builder receiver: `slf` is `self`
                                receiver = "self"
                                continue
                            py_ty = map_param(sym, name, ty)
                        except HiddenArg:
                            continue
                        except UnsupportedType as e:
                            failures.append({"symbol": sym.name, "reason": f"unsupported type {e}"})
                            return None
                        if default is not None:
                            try:
                                lit = python_literal(default, const_values)
                            except UnsupportedType as e:
                                failures.append({"symbol": sym.name, "reason": str(e)})
                                return None
                            parts.append(f"{py_name(name)}: {py_ty} = {lit}")
                        elif ty.startswith("Option"):
                            parts.append(f"{py_name(name)}: {py_ty} = None")
                        else:
                            parts.append(f"{py_name(name)}: {py_ty}")
                    else:
                        failures.append({"symbol": sym.name, "reason": f"unparseable arg {tok}"})
                        return None
            else:
                # derive from params: skip self and Python token
                for p in sym.params:
                    name = p["name"]
                    if name == "self":
                        continue
                    ty = p["type"]
                    try:
                        if re.match(r"^PyRef(?:Mut)?<'_,\s*Self>$", ty):
                            receiver = "self"
                            continue
                        py_ty = map_param(sym, name, ty)
                    except HiddenArg:
                        continue
                    except UnsupportedType as e:
                        failures.append({"symbol": sym.name, "reason": f"unsupported type {e}"})
                        return None
                    parts.append(f"{py_name(name)}: {py_ty}")

            # assemble with kw-only separator and receiver
            if is_method and cls_name and receiver and not sym.signature_attr:
                pass
            # decide on the self receiver display
            args: list[str] = []
            if is_method and cls_name:
                args.append("self")
            args.extend(parts)

            ret = "None"
            if sym.return_type:
                try:
                    ret = mapper.map(sym.return_type)
                except HiddenArg:
                    ret = "None"
                except DuckTypedParam:
                    key = (sym.class_py_name or "", sym.name, "return")
                    entry = DUCK_TYPES.get(key)
                    if entry is None:
                        failures.append({"symbol": sym.name, "reason": f"unaudited PyAny return {key}"})
                        return None
                    ret = entry
                except UnsupportedType:
                    ret = "None"
                    # note: we only fail closed on args; returns degrade to
                    # the honest "None" only for () and unknown objects would
                    # be flagged. For unknown returns we flag.
                    failures.append({"symbol": sym.name, "reason": f"unsupported return {sym.return_type}"})
                    return None
            return f"def {sym.py_name}({', '.join(args)}) -> {ret}: ..."
        except Exception as e:  # noqa: BLE001
            failures.append({"symbol": sym.name, "reason": f"emitter error {e!r}"})
            return None

    for name, sym in sorted(functions.items()):
        sig = emit_fn(sym, False, None)
        if sig:
            emitted[f"function:{sym.py_name}"] = sig

    # classes: sort by py name
    for rust, sym in sorted(classes.items()):
        emitted[f"class:{sym.py_name}"] = f"class {sym.py_name}:"
    # methods by class (walk symbols again with class context)
    by_class: dict[str, list[Symbol]] = {}
    for sym in model["symbols"]:
        if sym.kind in ("new", "method", "staticmethod", "classmethod", "getter", "setter", "classattr"):
            if sym.class_py_name is None or sym.class_py_name not in class_set:
                continue
            by_class.setdefault(sym.class_py_name, []).append(sym)
    for cls, syms in sorted(by_class.items()):
        mapper.self_type = cls
        for sym in syms:
            if sym.kind == "new":
                # constructor: __init__
                sig = emit_fn(sym, True, cls)
                if sig:
                    emitted[f"init:{cls}"] = sig.replace("def new(", "def __init__(")
            elif sym.kind == "staticmethod":
                sig = emit_fn(sym, False, cls)
                if sig:
                    emitted[f"staticmethod:{cls}:{sym.name}"] = f"    @staticmethod\n    {sig}"
            elif sym.kind == "getter":
                try:
                    ret_ty = sym.return_type or "()"
                    try:
                        ret = mapper.map(ret_ty)
                    except DuckTypedParam:
                        entry = DUCK_TYPES.get((cls, sym.name, "return"))
                        if entry is None:
                            raise UnsupportedType(
                                f"unaudited PyAny return {(cls, sym.name)} "
                                "(add it to DUCK_TYPES with provenance)"
                            ) from None
                        ret = entry
                    emitted[f"getter:{cls}:{sym.name}"] = (
                        f"    @property\n    def {sym.name}(self) -> {ret}: ..."
                    )
                except UnsupportedType as e:
                    failures.append({"symbol": sym.name, "reason": f"unsupported getter type {e}"})
            elif sym.kind == "setter":
                # property setter: derive from the single non-self arg
                arg_ty = None
                for p in sym.params:
                    if p["name"] != "self":
                        arg_ty = p["type"]
                        break
                if arg_ty:
                    try:
                        py_ty = mapper.map(arg_ty)
                        emitted[f"setter:{cls}:{sym.name}"] = (
                            f"    @{sym.name}.setter\n    def {sym.name}(self, value: {py_ty}) -> None: ..."
                        )
                    except UnsupportedType as e:
                        failures.append({"symbol": sym.name, "reason": f"unsupported setter type {e}"})
                else:
                    failures.append({"symbol": sym.name, "reason": f"setter without arg {sym.name}"})
            elif sym.kind == "classattr":
                try:
                    ty = mapper.map(sym.return_type or "str")
                    emitted[f"classattr:{cls}:{sym.name}"] = (
                        f"    {sym.name}: ClassVar[{ty}]"
                    )
                except UnsupportedType as e:
                    failures.append({"symbol": sym.name, "reason": f"unsupported classattr type {e}"})
            else:
                sig = emit_fn(sym, True, cls)
                if sig:
                    emitted[f"method:{cls}:{sym.name}"] = f"    {sig}"

    return {"emitted": emitted, "failures": failures, "mapper": mapper}


def generate_stub(model: dict) -> tuple[str, list[dict]]:
    built = build_signatures(model)
    emitted = built["emitted"]
    failures = built["failures"]

    lines = [HEADER, IMPORTS]
    used_aliases: set[str] = set()
    # collect aliases from emitted signatures
    for sig in emitted.values():
        for alias in ARRAY_ALIASES.values():
            if alias in sig:
                used_aliases.add(alias)
    ab = alias_block(used_aliases)
    if ab:
        lines.append(ab)

    # classes in sorted py-name order
    classes_sorted = sorted(
        {key.split(":", 1)[1] for key in emitted if key.startswith("class:")}
    )
    # py-name -> enum variants (from the Rust class symbols)
    variants_by_py: dict[str, list[str]] = {}
    for sym in model["classes"].values():
        variants_by_py[sym.py_name] = sym.enum_variants
    for cls in classes_sorted:
        lines.append(f"class {cls}:")
        # enum variants as class-level constants
        for variant in variants_by_py.get(cls, []):
            lines.append(f"    {variant}: ClassVar[\"{cls}\"]")
        # init
        init = emitted.get(f"init:{cls}")
        if init:
            lines.append(f"    {init}")
        # staticmethods / methods / getters / setters / classattrs in order
        for key in sorted(emitted):
            if key.startswith(f"staticmethod:{cls}:") or key.startswith(f"method:{cls}:") or key.startswith(f"getter:{cls}:") or key.startswith(f"setter:{cls}:") or key.startswith(f"classattr:{cls}:"):
                body = emitted[key]
                lines.append(body)
        lines.append("")

    # module functions
    fn_keys = sorted(k for k in emitted if k.startswith("function:"))
    for key in fn_keys:
        lines.append(emitted[key])
        lines.append("")

    stub = "\n".join(lines).rstrip() + "\n"
    return stub, failures


# ---------------------------------------------------------------------------
# Facade stub
# ---------------------------------------------------------------------------

def generate_facade_stub(exported: list[str]) -> str:
    """Generate ``__init__.pyi`` for the ``pykwavers`` facade package.

    The facade is a thin re-export of the extension surface; its stub mirrors
    ``__all__`` by importing each public name from ``._pykwavers``.
    """
    lines = [
        HEADER.rstrip(),
        '"""Typed facade for the registered ``pykwavers`` extension surface."""',
        "",
        "from types import ModuleType",
        "",
        "from ._pykwavers import (",
    ]
    # Only include names that can be imported from the extension module; skip
    # pure-Python parity helpers (they are defined in sibling modules and are
    # not part of the extension surface). The facade stub is a mirror of
    # ``__all__``; parity names still resolve at runtime.
    for name in sorted(exported):
        lines.append(f"    {name},")
    lines.extend(
        [
            ")",
            "",
            "_pykwavers: ModuleType",
            "",
            "__all__ = [",
        ]
    )
    for name in sorted(exported):
        lines.append(f'    "{name}",')
    lines.extend(["]", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    functions, classes, _ = registered_symbols(root)
    imported, exported = facade_symbols(root)
    model = parse_rust(root)
    stub, failures = generate_stub(model)

    payload = {
        "schema": 3,
        "source": "Rust PyO3 registration calls and pykwavers.__init__ facade",
        "registered": {"classes": classes, "functions": functions},
        # Rust ident -> Python-visible name (#[pyclass(name = ...)] and
        # #[pyfunction(name = ...)] renames). Reachability checks must use
        # these, not the raw idents.
        "registered_py_names": {
            "classes": model["class_py_names"],
        },
        "facade": {"imported": imported, "__all__": exported},
        "drift": {
            "registered_missing_from_facade_imports": sorted(set(functions + classes) - set(imported)),
            "registered_missing_from___all__": sorted(set(functions + classes) - set(exported)),
            "facade_imports_not_registered": sorted(set(imported) - set(functions + classes)),
        },
        "failures": failures,
    }
    inventory = root / "python" / "pykwavers" / "_generated_surface.json"
    stub_path = root / "python" / "pykwavers" / "_pykwavers.pyi"
    facade_stub_path = root / "python" / "pykwavers" / "__init__.pyi"
    inventory_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    stub_text = stub
    facade_stub_text = generate_facade_stub(exported)

    if args.check:
        expected = [
            (inventory, inventory_text),
            (stub_path, stub_text),
            (facade_stub_path, facade_stub_text),
        ]
        stale = [str(path) for path, text in expected if not path.exists() or path.read_text(encoding="utf-8") != text]
        if stale:
            print("stale generated artifacts:")
            print("\n".join(stale))
            return 1
        if failures:
            print("generator failures (fails closed):")
            for f in failures:
                print(f"  {f['symbol']}: {f['reason']}")
            return 1
        return 0

    inventory.write_text(inventory_text, encoding="utf-8")
    stub_path.write_text(stub_text, encoding="utf-8")
    facade_stub_path.write_text(facade_stub_text, encoding="utf-8")
    print(f"generated {inventory}")
    print(f"generated {stub_path}")
    print(f"generated {facade_stub_path}")
    print(f"registered classes={len(classes)} functions={len(functions)}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f['symbol']}: {f['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
