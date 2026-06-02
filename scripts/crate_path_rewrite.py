#!/usr/bin/env python3
"""Grouped-import-aware path rewriter for kwavers crate extraction (ADR 009).

For a layer being extracted into its own crate, rewrites cross-crate paths,
correctly handling grouped/multiline `use crate::{ ... }` blocks that line-based
sed cannot (the gap that broke the first kwavers-domain attempt).

Rules (self=the layer being extracted, e.g. `domain`):
  crate::<self>::X   -> crate::X            (now crate root)
  crate::<dep>::X    -> <dep_crate>::X      (extracted dependency, e.g. core->kwavers_core)
  use crate::{ a::, <dep>::, <self>:: }     -> split: kept crate group + routed `use` stmts

Usage: crate_path_rewrite.py <src_dir> <self_layer> <dep1=crate1> [<dep2=crate2> ...]
  e.g. crate_path_rewrite.py crates/kwavers-domain/src domain core=kwavers_core math=kwavers_math
"""
import os, re, sys

def split_top_level(s):
    """Split a brace-inner string on top-level commas (respect nested {})."""
    items, depth, cur = [], 0, ""
    for ch in s:
        if ch == "{": depth += 1; cur += ch
        elif ch == "}": depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            if cur.strip(): items.append(cur.strip())
            cur = ""
        else: cur += ch
    if cur.strip(): items.append(cur.strip())
    return items

def find_matching_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return i
    return -1

def rewrite_use_crate_groups(text, selflayer, depmap):
    """Find `use crate::{ ... };` blocks and route their items by first segment."""
    out, i = [], 0
    # capture optional leading visibility (pub, pub(crate), pub(super), ...)
    pat = re.compile(r"(pub(?:\([^)]*\))?\s+)?use\s+crate::\{")
    while True:
        m = pat.search(text, i)
        if not m: out.append(text[i:]); break
        out.append(text[i:m.start()])
        vis = m.group(1) or ""
        brace_open = m.end() - 1
        brace_close = find_matching_brace(text, brace_open)
        if brace_close < 0:  # malformed; bail on this match
            out.append(text[m.start():m.end()]); i = m.end(); continue
        inner = text[brace_open+1:brace_close]
        # consume trailing `;`
        j = brace_close + 1
        while j < len(text) and text[j] in " \t\r\n": j += 1
        if j < len(text) and text[j] == ";": j += 1
        items = split_top_level(inner)
        kept, stmts = [], []
        for it in items:
            seg = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", it)
            head = seg.group(1) if seg else ""
            if head == selflayer:
                rest = it[len(head)+2:] if it.startswith(head+"::") else it
                stmts.append(f"{vis}use crate::{rest};")
            elif head in depmap:
                rest = it[len(head)+2:] if it.startswith(head+"::") else it
                stmts.append(f"{vis}use {depmap[head]}::{rest};")
            else:
                kept.append(it)
        rebuilt = []
        if kept:
            rebuilt.append(f"{vis}use crate::{{{', '.join(kept)}}};")
        rebuilt.extend(stmts)
        out.append("\n".join(rebuilt))
        i = j
    return "".join(out)

def simple_prefix(text, selflayer, depmap):
    # inline + single-use refs not inside a crate::{...} group
    text = text.replace(f"crate::{selflayer}::", "crate::")
    for dep, crate in depmap.items():
        text = text.replace(f"crate::{dep}::", f"{crate}::")
    return text

# kwavers crate-root re-exports (lib.rs): code that used `crate::<Ident>` relied on
# these. After extraction `crate::` is the layer root, so they must be re-pointed at
# the owning extracted crate. Only applied when that crate is an extracted dep.
ROOT_REEXPORTS = {
    "KwaversResult": ("core", "kwavers_core::error::KwaversResult"),
    "KwaversError": ("core", "kwavers_core::error::KwaversError"),
    "Grid": ("domain", "kwavers_domain::grid::Grid"),
    "Medium": ("domain", "kwavers_domain::medium::traits::Medium"),
}

def root_rewrite(text, selflayer, depmap):
    for ident, (dep, target) in ROOT_REEXPORTS.items():
        if dep == selflayer or dep in depmap:
            text = re.sub(rf"\bcrate::{ident}\b", target, text)
    return text

def main():
    src, selflayer = sys.argv[1], sys.argv[2]
    depmap = dict(kv.split("=") for kv in sys.argv[3:])
    changed = 0
    for root, _, files in os.walk(src):
        for fn in files:
            if not fn.endswith(".rs"): continue
            p = os.path.join(root, fn)
            orig = open(p, encoding="utf-8").read()
            t = rewrite_use_crate_groups(orig, selflayer, depmap)
            t = simple_prefix(t, selflayer, depmap)
            t = root_rewrite(t, selflayer, depmap)
            if t != orig:
                open(p, "w", encoding="utf-8", newline="").write(t)
                changed += 1
    print(f"rewrote {changed} files in {src} (self={selflayer}, deps={depmap})")

if __name__ == "__main__":
    main()
