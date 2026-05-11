"""Chapter 22 figures — Simulation orchestration via the capability catalog.

Renders two diagrams that accompany the chapter:

1. ``capability_fanout.svg`` — a layered view of
   ``PhysicsConfig`` → ``PhysicsCatalog`` → enabled plugins, with the
   structured-error fork drawn explicitly so the reader sees what happens
   to a capability whose Plugin adapter is not yet wired.

2. ``field_dependency_dag.svg`` — the ``UnifiedFieldType`` dependency
   graph for a representative three-capability config, with the
   topologically-sorted execution order overlaid as node labels.

The figures are *logical* — they describe the architecture invariants,
not a live simulation. No compiled crate is required.

Run::

    python pykwavers/examples/book/ch22_simulation_orchestration.py

Outputs are written to ``docs/book/figures/ch22/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch22"


# ---------------------------------------------------------------------------
# Figure 1 — capability fan-out
# ---------------------------------------------------------------------------


def render_capability_fanout(out: Path) -> None:
    """Three-layer drawing: config models → catalog → outcomes.

    The ``unsupported`` branch is shown in red so the reader sees that
    every variant resolves either to a wired plugin or to a structured
    ``ConfigError``; there is no silent fallback path.
    """

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Column 1 — declared capabilities (left)
    capabilities = [
        ("LinearAcoustics{FDTD}", "ok"),
        ("LinearAcoustics{PSTD}", "ok"),
        ("LinearAcoustics{DG}", "err"),
        ("NonlinearAcoustics{KZK}", "ok"),
        ("NonlinearAcoustics{Westervelt}", "err"),
        ("ThermalDiffusion", "ok"),
        ("MechanicalStress", "ok"),
        ("BubbleDynamics", "err"),
        ("OpticalPropagation", "err"),
    ]
    cap_y = [7.4 - 0.78 * i for i in range(len(capabilities))]
    for (name, _), y in zip(capabilities, cap_y):
        ax.add_patch(
            plt.Rectangle((0.2, y - 0.27), 3.4, 0.54, fc="#eef3fb", ec="#3b6ea8", lw=1.2)
        )
        ax.text(1.9, y, name, ha="center", va="center", fontsize=9)

    ax.text(1.9, 7.95, "PhysicsConfig.models", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Column 2 — the catalog dispatcher
    ax.add_patch(plt.Rectangle((4.6, 2.5), 2.6, 3.0, fc="#fdf6e3", ec="#a07d2c", lw=1.4))
    ax.text(5.9, 5.10, "PhysicsCatalog", ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.text(5.9, 4.60, "match kind {", ha="center", va="center", fontsize=9, family="monospace")
    ax.text(5.9, 4.20, "  …concrete plugin…", ha="center", va="center",
            fontsize=9, family="monospace")
    ax.text(5.9, 3.80, "  …Err(unsupported)…", ha="center", va="center",
            fontsize=9, family="monospace")
    ax.text(5.9, 3.40, "}", ha="center", va="center", fontsize=9, family="monospace")
    ax.text(5.9, 2.85, "exhaustive (Thm. 22.1)", ha="center", va="center",
            fontsize=8, fontstyle="italic")

    # Column 3 — outcomes
    ok_box = plt.Rectangle((8.4, 4.6), 3.4, 1.4, fc="#e7f3e8", ec="#2e7d32", lw=1.4)
    err_box = plt.Rectangle((8.4, 2.0), 3.4, 1.4, fc="#fbe9e7", ec="#c0392b", lw=1.4)
    ax.add_patch(ok_box)
    ax.add_patch(err_box)
    ax.text(10.1, 5.6, "Box<dyn Plugin>",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(10.1, 5.1, "added to PluginManager", ha="center", va="center", fontsize=9)
    ax.text(10.1, 3.0, "ConfigError::InvalidValue",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(10.1, 2.5, "names variant + models[i]", ha="center", va="center", fontsize=9)

    # Edges
    for (name, status), y in zip(capabilities, cap_y):
        ax.annotate(
            "",
            xy=(4.6, 4.0),
            xytext=(3.6, y),
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.8),
        )
        target_y = 5.3 if status == "ok" else 2.7
        target_color = "#2e7d32" if status == "ok" else "#c0392b"
        ax.annotate(
            "",
            xy=(8.4, target_y),
            xytext=(7.2, 4.0),
            arrowprops=dict(arrowstyle="->", color=target_color, lw=0.9, alpha=0.55),
        )

    ax.set_title(
        "PhysicsConfig → PhysicsCatalog → outcomes\n"
        "(green = wired plugin; red = structured error, no silent fallback)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — field-dependency DAG
# ---------------------------------------------------------------------------


def representative_plugin_signatures() -> dict[str, dict[str, Iterable[str]]]:
    """Return a representative (provides, requires) map for three plugins.

    These signatures mirror the *kind* of dependencies the production
    plugins declare; they are not extracted from the live code at render
    time so the figure remains reproducible without a Rust toolchain.
    The ``required_fields``/``provided_fields`` declarations on the
    matching plugins (PSTDPlugin, ThermalDiffusionPlugin,
    ElasticWavePlugin) are the SSOT for the actual runtime graph.
    """

    return {
        "LinearAcoustics{PSTD}": {
            "requires": ["Density", "SoundSpeed"],
            "provides": ["Pressure", "VelocityX", "VelocityY", "VelocityZ"],
        },
        "ThermalDiffusion": {
            "requires": ["Pressure", "Density"],
            "provides": ["Temperature"],
        },
        "MechanicalStress": {
            "requires": ["Density", "Temperature"],
            "provides": ["StressXX", "StressYY", "StressZZ"],
        },
    }


def render_field_dependency_dag(out: Path) -> None:
    """Bipartite plugin↔field graph with the topological order overlaid."""

    sigs = representative_plugin_signatures()

    # Build a plugin-level DAG: edge P_i → P_j iff Prov(P_i) ∩ Req(P_j) ≠ ∅.
    plugin_dag = nx.DiGraph()
    plugin_dag.add_nodes_from(sigs.keys())
    for src, src_sig in sigs.items():
        for dst, dst_sig in sigs.items():
            if src == dst:
                continue
            if set(src_sig["provides"]) & set(dst_sig["requires"]):
                plugin_dag.add_edge(src, dst)

    # Topological order — exists by Theorem 22.2 because the example is acyclic.
    order = list(nx.topological_sort(plugin_dag))
    rank = {p: i for i, p in enumerate(order)}

    # Lay out: plugins on top row, fields they touch on bottom row.
    fields = sorted(
        {f for sig in sigs.values() for f in sig["provides"]}
        | {f for sig in sigs.values() for f in sig["requires"]}
    )
    plugin_x = {p: 1.0 + 4.0 * rank[p] for p in order}
    field_x = {f: 0.6 + (12.0 * i) / (len(fields) - 1) for i, f in enumerate(fields)}

    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 4.5)

    # Plugins (top)
    for plugin, x in plugin_x.items():
        ax.add_patch(
            plt.Rectangle((x - 1.6, 3.3), 3.2, 0.85,
                          fc="#dceaf7", ec="#2c5282", lw=1.4)
        )
        ax.text(x, 3.85, plugin, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x, 3.55, f"order π = {rank[plugin]}",
                ha="center", va="center", fontsize=8, fontstyle="italic")

    # Fields (bottom)
    for field, x in field_x.items():
        ax.add_patch(
            plt.Circle((x, 0.5), 0.32, fc="#f1f1f1", ec="#555", lw=1.0)
        )
        ax.text(x, -0.2, field, ha="center", va="center", fontsize=8, rotation=20)

    # Edges: requires (read) and provides (write)
    for plugin, sig in sigs.items():
        px = plugin_x[plugin]
        for f in sig["requires"]:
            ax.annotate(
                "",
                xy=(px, 3.3),
                xytext=(field_x[f], 0.82),
                arrowprops=dict(arrowstyle="->", color="#2e7d32",
                                lw=1.0, alpha=0.85),
            )
        for f in sig["provides"]:
            ax.annotate(
                "",
                xy=(field_x[f], 0.82),
                xytext=(px, 3.3),
                arrowprops=dict(arrowstyle="->", color="#c0392b",
                                lw=1.0, alpha=0.85),
            )

    # Legend
    ax.plot([], [], color="#2e7d32", lw=2.0, label="reads (required_fields)")
    ax.plot([], [], color="#c0392b", lw=2.0, label="writes (provided_fields)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    ax.set_title(
        "Field-dependency graph for an enabled three-capability config\n"
        "Plugin order π is the topological sort returned by PluginManager "
        "(Theorem 22.2)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fanout = FIG_DIR / "capability_fanout.svg"
    dag = FIG_DIR / "field_dependency_dag.svg"
    render_capability_fanout(fanout)
    render_field_dependency_dag(dag)
    print(f"wrote {fanout.relative_to(REPO_ROOT)}")
    print(f"wrote {dag.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
