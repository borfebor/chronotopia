"""
Generate `docs/reference/controls.md` from `docs.py`.

The point of this file is that there is no second copy of the text. Every tooltip
in the running app comes from `docs.SECTIONS`; this renders the same dictionary
into the manual, so the site and the app cannot drift apart. Editing the page by
hand is pointless — the next build overwrites it.

Run from the repository root, before `mkdocs build`:
    python docs/gen_control_reference.py
"""

from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import docs as tooltips           # docs.py, NOT the docs/ directory — see mkdocs.yml

OUT = os.path.join(REPO, "docs", "reference", "controls.md")

FRONT = """<!--
  GENERATED FILE — do not edit.
  Written by docs/gen_control_reference.py from the SECTIONS dictionary in
  docs.py, which is also what the app's tooltips read. Edit docs.py instead.
-->

"""

INTRO = """
!!! info "This page is the app's own tooltips"

    Every entry below is the exact text that appears when you hover the matching
    control in Chronotopia. It is generated from `docs.py` at build time, so the
    manual cannot say one thing while the app says another. The app can hand you
    the same document: **Download the control reference**.

"""


def main() -> None:
    body = tooltips.as_markdown(title="Control reference")
    # Slot the admonition in after the generated H1 and its lead paragraph.
    head, _, rest = body.partition("\n## ")
    text = FRONT + head + "\n" + INTRO + "\n## " + rest
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")
    n_controls = len(tooltips.HELP)
    n_sections = len(tooltips.SECTIONS)
    print(f"  controls.md — {n_controls} controls across {n_sections} sections")


if __name__ == "__main__":
    main()
