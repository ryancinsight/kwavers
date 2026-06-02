#!/usr/bin/env python3
"""Consolidate per-script parity metrics into a single comparative artifact.

Scans ``pykwavers/examples/output/*_metrics.txt`` produced by the
``*_compare.py`` and ``*_jl_compare.py`` parity drivers, classifies the
reference backend (k-wave-python, KWave.jl, k-wave MATLAB cache,
analytical / canonical), parses status + scalar parity metrics, and emits:

* ``output/parity_dashboard.png`` — comparative figure
* ``output/parity_dashboard.md``  — markdown summary table

This is a read-only aggregator. It does NOT re-run any parity script; run
``_run_parity_sweep.py`` or ``_run_julia_parity_sweep.py`` first to
regenerate the underlying metrics files.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXAMPLES_DIR, "output")

KWP_REF = "k-wave-python"
KJL_REF = "KWave.jl"
KWM_REF = "k-wave MATLAB"
CAN_REF = "analytical/canonical"

STATUS_PASS = {"PASS"}
STATUS_DIAG = {"DIAGNOSTIC", "DIAG", "INFO", "INFORMATIONAL"}
STATUS_FAIL = {"FAIL", "FAILED", "ERROR", "TIMEOUT"}

STATUS_COLOR = {
    "PASS": "#1f9d55",
    "DIAGNOSTIC": "#f2b134",
    "FAIL": "#d64545",
    "?": "#888888",
}


@dataclass
class ParityRecord:
    script: str
    backend: str
    status: str
    pearson: float | None = None
    psnr_db: float | None = None
    rms_ratio: float | None = None
    peak_ratio: float | None = None
    kwave_runtime_s: float | None = None
    pykw_runtime_s: float | None = None
    engine_ref: str | None = None
    raw: dict[str, str] = field(default_factory=dict)


_FLOAT = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"

_PATTERNS = {
    "pearson": re.compile(
        rf"\bpearson(?:_r(?:_min|_mean)?|_min)?(?![A-Za-z_])\s*[:=]\s*{_FLOAT}",
        re.M,
    ),
    "pearson_aligned": re.compile(
        rf"\bpearson_r_lag_aligned\s*[:=]\s*{_FLOAT}", re.M
    ),
    "psnr": re.compile(rf"\bpsnr_db(?:_min)?\s*[:=]\s*{_FLOAT}", re.M),
    "rms_ratio": re.compile(rf"\brms_ratio\s*[:=]\s*{_FLOAT}", re.M),
    "peak_ratio": re.compile(rf"\bpeak_ratio\s*[:=]\s*{_FLOAT}", re.M),
    "kwave_runtime": re.compile(rf"\bkwave_runtime_s\s*[:=]\s*{_FLOAT}", re.M),
    "pykw_runtime": re.compile(rf"\bpykwavers_runtime_s\s*[:=]\s*{_FLOAT}", re.M),
    "engine_ref": re.compile(r"^\s*engine_ref(?:\s*\(.*?\))?\s*[:=]\s*(.+)$", re.M),
}
_STATUS_RE = re.compile(
    r"^\s*(?:parity_status|status|checkpoint_status|procedure_status|"
    r"result|overall[^:=]*)\s*[:=]\s*(\S+)",
    re.M | re.IGNORECASE,
)


def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        v = float(s)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except ValueError:
        return None


def _classify_backend(script_stem: str, engine_ref: str | None) -> str:
    if script_stem.endswith("_jl"):
        return KJL_REF
    if engine_ref and "KWave.jl" in engine_ref:
        return KJL_REF
    if script_stem.startswith("canonical_"):
        return CAN_REF
    return KWP_REF


def parse_metrics_file(path: str) -> ParityRecord:
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    stem = os.path.basename(path)[: -len("_metrics.txt")]

    status_match = _STATUS_RE.search(text)
    status_raw = status_match.group(1).upper() if status_match else "?"
    if status_raw in STATUS_PASS:
        status = "PASS"
    elif status_raw in STATUS_DIAG:
        status = "DIAGNOSTIC"
    elif status_raw in STATUS_FAIL:
        status = "FAIL"
    else:
        status = status_raw or "?"

    def _first_match(rgx: re.Pattern) -> str | None:
        m = rgx.search(text)
        return m.group(1) if m else None

    def _last_match(rgx: re.Pattern) -> str | None:
        m = list(rgx.finditer(text))
        return m[-1].group(1) if m else None

    engine_ref = None
    er = _PATTERNS["engine_ref"].search(text)
    if er:
        engine_ref = er.group(1).strip()

    # Prefer lag-aligned Pearson when the script reports it (oscillatory
    # tone-burst comparisons), otherwise the first pearson_r in the file —
    # that is the headline / primary metric. Later pearson_r values are
    # typically per-row forward-trace pearsons that would otherwise displace
    # the headline metric.
    aligned = _first_match(_PATTERNS["pearson_aligned"])
    pearson_val = aligned if aligned is not None else _first_match(_PATTERNS["pearson"])

    return ParityRecord(
        script=stem + "_compare.py",
        backend=_classify_backend(stem, engine_ref),
        status=status,
        pearson=_safe_float(pearson_val),
        psnr_db=_safe_float(_last_match(_PATTERNS["psnr"])),
        rms_ratio=_safe_float(_last_match(_PATTERNS["rms_ratio"])),
        peak_ratio=_safe_float(_last_match(_PATTERNS["peak_ratio"])),
        kwave_runtime_s=_safe_float(_last_match(_PATTERNS["kwave_runtime"])),
        pykw_runtime_s=_safe_float(_last_match(_PATTERNS["pykw_runtime"])),
        engine_ref=engine_ref,
    )


def collect_records() -> list[ParityRecord]:
    paths = sorted(
        os.path.join(OUTPUT_DIR, n)
        for n in os.listdir(OUTPUT_DIR)
        if n.endswith("_metrics.txt")
    )
    return [parse_metrics_file(p) for p in paths]


def _status_counts(recs: list[ParityRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in recs:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def render_dashboard(recs: list[ParityRecord], out_png: str) -> None:
    backends = [KWP_REF, KJL_REF, CAN_REF]
    by_backend: dict[str, list[ParityRecord]] = {b: [] for b in backends}
    for r in recs:
        by_backend.setdefault(r.backend, []).append(r)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 1.6])

    # Panel 1: status counts per backend
    ax0 = fig.add_subplot(gs[0, 0])
    states = ["PASS", "DIAGNOSTIC", "FAIL", "?"]
    x = np.arange(len(backends))
    width = 0.2
    for i, st in enumerate(states):
        vals = [sum(1 for r in by_backend[b] if r.status == st) for b in backends]
        ax0.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=st,
            color=STATUS_COLOR.get(st, "#888"),
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(backends, rotation=10, fontsize=9)
    ax0.set_ylabel("script count")
    ax0.set_title("parity status by reference backend")
    ax0.legend(fontsize=8, loc="upper right")

    # Panel 2: Pearson histogram (all backends combined)
    ax1 = fig.add_subplot(gs[0, 1])
    pearsons = [r.pearson for r in recs if r.pearson is not None]
    ax1.hist(pearsons, bins=np.linspace(0.0, 1.001, 21), color="#3273dc",
             edgecolor="white")
    ax1.axvline(0.9, color="#d64545", ls="--", lw=1, label="0.9 acceptance")
    ax1.axvline(0.99, color="#1f9d55", ls="--", lw=1, label="0.99 high-parity")
    ax1.set_xlabel("Pearson r")
    ax1.set_ylabel("script count")
    ax1.set_title(f"Pearson distribution (n={len(pearsons)})")
    ax1.set_xlim(0.0, 1.02)
    ax1.legend(fontsize=8)

    # Panel 3: PSNR histogram
    ax2 = fig.add_subplot(gs[0, 2])
    psnrs = [r.psnr_db for r in recs if r.psnr_db is not None]
    if psnrs:
        finite = [p for p in psnrs if p < 400]
        ax2.hist(finite, bins=20, color="#9461c9", edgecolor="white")
        ax2.axvline(14, color="#d64545", ls="--", lw=1, label="14 dB floor")
        ax2.axvline(40, color="#1f9d55", ls="--", lw=1, label="40 dB high-parity")
    ax2.set_xlabel("PSNR (dB)")
    ax2.set_ylabel("script count")
    ax2.set_title(f"PSNR distribution (n={len(psnrs)})")
    ax2.legend(fontsize=8)

    # Panel 4: kwavers speed-up vs k-wave-python (where both runtimes captured)
    ax3 = fig.add_subplot(gs[1, 0])
    speedups = [
        (r.script, r.kwave_runtime_s / r.pykw_runtime_s)
        for r in recs
        if r.kwave_runtime_s and r.pykw_runtime_s
    ]
    if speedups:
        labels = [s.replace("_compare.py", "").replace("_", " ")[:32] for s, _ in speedups]
        vals = [v for _, v in speedups]
        order = np.argsort(vals)[::-1]
        ypos = np.arange(len(vals))
        ax3.barh(
            ypos,
            [vals[i] for i in order],
            color=["#1f9d55" if vals[i] >= 1 else "#d64545" for i in order],
        )
        ax3.set_yticks(ypos)
        ax3.set_yticklabels([labels[i] for i in order], fontsize=7)
        ax3.axvline(1.0, color="k", lw=0.6)
        ax3.set_xscale("log")
        ax3.set_xlabel("kwavers speed-up (×)")
        ax3.set_title(f"runtime parity (n={len(vals)})")
    else:
        ax3.text(0.5, 0.5, "no runtime pairs", ha="center", va="center")
        ax3.set_axis_off()

    # Panel 5: scatter Pearson vs PSNR
    ax4 = fig.add_subplot(gs[1, 1:])
    colors = {KWP_REF: "#3273dc", KJL_REF: "#f2b134", CAN_REF: "#9461c9"}
    for b in backends:
        bp = [(r.pearson, r.psnr_db) for r in by_backend[b]
              if r.pearson is not None and r.psnr_db is not None]
        if not bp:
            continue
        xs, ys = zip(*bp)
        ax4.scatter(xs, [min(y, 350) for y in ys], s=42, alpha=0.75,
                    label=f"{b} (n={len(bp)})", color=colors.get(b, "#666"))
    ax4.set_xlabel("Pearson r")
    ax4.set_ylabel("PSNR (dB, capped at 350)")
    ax4.set_title("parity quality landscape")
    ax4.axvline(0.9, color="#d64545", ls="--", lw=0.8)
    ax4.axhline(14, color="#d64545", ls="--", lw=0.8)
    ax4.legend(fontsize=8, loc="lower right")
    ax4.set_xlim(0.0, 1.02)
    ax4.grid(True, alpha=0.2)

    # Panel 6: per-script status matrix
    ax5 = fig.add_subplot(gs[2, :])
    ordered = sorted(recs, key=lambda r: (r.backend, r.status, r.script))
    n = len(ordered)
    cols = 3
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows, cols, 3))
    labels = [["" for _ in range(cols)] for _ in range(rows)]
    for idx, r in enumerate(ordered):
        rr, cc = idx % rows, idx // rows
        rgb = matplotlib.colors.to_rgb(STATUS_COLOR.get(r.status, "#888"))
        grid[rr, cc] = rgb
        tag = r.script.replace("_compare.py", "").replace("_jl", "·jl")
        if len(tag) > 36:
            tag = tag[:33] + "…"
        if r.pearson is not None:
            tag = f"{tag}  r={r.pearson:.3f}"
        labels[rr][cc] = tag
    ax5.imshow(grid, aspect="auto", interpolation="nearest")
    for rr in range(rows):
        for cc in range(cols):
            if labels[rr][cc]:
                ax5.text(cc, rr, labels[rr][cc], ha="center", va="center",
                         fontsize=6.5, color="white")
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.set_title(
        f"per-script status — {n} scripts  •  "
        + "  •  ".join(f"{k}={v}" for k, v in _status_counts(recs).items())
    )

    fig.suptitle(
        "kwavers / pykwavers parity vs k-wave-python, k-wave MATLAB, KWave.jl",
        fontsize=13, weight="bold",
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def render_markdown(recs: list[ParityRecord], out_md: str) -> None:
    counts = _status_counts(recs)
    by_backend: dict[str, list[ParityRecord]] = {}
    for r in recs:
        by_backend.setdefault(r.backend, []).append(r)

    lines: list[str] = []
    lines.append("# Parity Dashboard\n")
    lines.append(
        "Aggregated from `pykwavers/examples/output/*_metrics.txt`. "
        "Generated by `parity_dashboard.py`.\n"
    )
    lines.append(f"**Total scripts:** {len(recs)}  ")
    lines.append("  •  ".join(f"**{k}** {v}" for k, v in counts.items()) + "\n")

    pearsons = [r.pearson for r in recs if r.pearson is not None]
    psnrs = [r.psnr_db for r in recs if r.psnr_db is not None]
    if pearsons:
        lines.append(
            f"**Pearson** — median {np.median(pearsons):.4f}, "
            f"min {min(pearsons):.4f}, n={len(pearsons)}  "
        )
    if psnrs:
        lines.append(
            f"**PSNR (dB)** — median {np.median(psnrs):.2f}, "
            f"min {min(psnrs):.2f}, n={len(psnrs)}\n"
        )

    for backend in (KWP_REF, KJL_REF, CAN_REF):
        bk_recs = sorted(by_backend.get(backend, []), key=lambda r: (r.status, r.script))
        if not bk_recs:
            continue
        lines.append(f"\n## Reference backend: {backend} ({len(bk_recs)} scripts)\n")
        lines.append(
            "| Script | Status | Pearson | PSNR (dB) | rms_ratio | peak_ratio |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in bk_recs:
            def _f(v: float | None, fmt: str) -> str:
                return fmt.format(v) if v is not None else "—"
            lines.append(
                f"| `{r.script}` | {r.status} | "
                f"{_f(r.pearson, '{:.4f}')} | {_f(r.psnr_db, '{:.2f}')} | "
                f"{_f(r.rms_ratio, '{:.3f}')} | {_f(r.peak_ratio, '{:.3f}')} |"
            )

    open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    if not os.path.isdir(OUTPUT_DIR):
        print(f"missing output directory: {OUTPUT_DIR}", file=sys.stderr)
        return 2
    recs = collect_records()
    if not recs:
        print("no *_metrics.txt files found", file=sys.stderr)
        return 2

    out_png = os.path.join(OUTPUT_DIR, "parity_dashboard.png")
    out_md = os.path.join(OUTPUT_DIR, "parity_dashboard.md")
    render_dashboard(recs, out_png)
    render_markdown(recs, out_md)

    counts = _status_counts(recs)
    print(f"parity_dashboard: {len(recs)} scripts aggregated")
    for k, v in sorted(counts.items()):
        print(f"  {k:>10s}: {v}")
    pearsons = [r.pearson for r in recs if r.pearson is not None]
    psnrs = [r.psnr_db for r in recs if r.psnr_db is not None]
    if pearsons:
        print(f"  Pearson  : median={np.median(pearsons):.4f}  min={min(pearsons):.4f}  n={len(pearsons)}")
    if psnrs:
        print(f"  PSNR(dB) : median={np.median(psnrs):.2f}  min={min(psnrs):.2f}  n={len(psnrs)}")
    print(f"  figure   : {out_png}")
    print(f"  markdown : {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
