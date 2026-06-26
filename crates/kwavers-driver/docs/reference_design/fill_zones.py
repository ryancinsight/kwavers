"""Cache zone fills in a .kicad_pcb using KiCad's own ZONE_FILLER.

The kicad-routing engine emits ground-pour and teardrop zones as *unfilled* outlines; KiCad's DRC
fills them on the fly, but exporters (gerbers, SVG, 3D render) plot only the *cached* fill. This
finalize step loads the board, runs the authoritative ZONE_FILLER (which carves design-rule
clearance around every foreign feature), and saves — so the delivered fabrication outputs contain
the poured copper. Run after the Rust engine emits the board, before any export.

Usage:  python fill_zones.py <board.kicad_pcb>
"""

import sys
import pcbnew


def main(path: str) -> int:
    board = pcbnew.LoadBoard(path)
    zones = board.Zones()
    n = len(zones)
    if n == 0:
        print(f"{path}: no zones to fill")
        return 0
    # Build connectivity before filling so island removal can resolve which fragments reach the net.
    # The zones keep their authored island-removal policy (ALWAYS-remove for both the plane pours and
    # the teardrops), so this cached fill and KiCad's own DRC refill converge on the same clean result
    # — no unconnected fill fragments (which DRC flags as isolated_copper).
    board.BuildConnectivity()
    filler = pcbnew.ZONE_FILLER(board)
    ok = filler.Fill(zones)
    pcbnew.SaveBoard(path, board)
    print(f"{path}: filled {n} zones (filler returned {ok})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
