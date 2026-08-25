"""
Guard against the `docs.py` / `docs/` collision.

The repository contains both a module `docs.py` (every tooltip in the app) and a
directory `docs/` (this documentation site). Python resolves a regular module
ahead of a namespace package, so `import docs` in app.py finds `docs.py` and
everything works — but that is a property of import precedence, not of intent.
Adding `docs/__init__.py` would make `docs/` a regular package, flip the
resolution, and break the app at startup with an AttributeError that points
nowhere near the cause.

This check runs in CI. If it ever fails, the fix is either to delete the
`__init__.py` or to rename `docs.py` to something unambiguous (`tooltips.py`)
and update the import in `app.py` and the section-23 check in `verify.py`.

    python docs/check_docs.py
"""

from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

problems = []

init = os.path.join(REPO, "docs", "__init__.py")
if os.path.exists(init):
    problems.append(
        "docs/__init__.py exists — this turns the docs directory into a package "
        "that shadows docs.py, and `import docs` in app.py will no longer find "
        "the tooltip module."
    )

sys.path.insert(0, REPO)
try:
    import docs as _docs
except Exception as exc:                                  # pragma: no cover
    problems.append(f"`import docs` failed: {type(exc).__name__}: {exc}")
else:
    resolved = getattr(_docs, "__file__", None)
    expected = os.path.join(REPO, "docs.py")
    if resolved is None:
        problems.append(
            "`import docs` resolved to a namespace package (the docs/ directory) "
            "instead of docs.py."
        )
    elif os.path.abspath(resolved) != os.path.abspath(expected):
        problems.append(f"`import docs` resolved to {resolved}, expected {expected}")
    elif not hasattr(_docs, "as_markdown"):
        problems.append(
            "docs.py imported but has no as_markdown — app.py depends on it."
        )

if problems:
    for p in problems:
        print(f"FAIL  {p}")
    sys.exit(1)

print("PASS  `import docs` resolves to docs.py; the site directory is not shadowing it")
