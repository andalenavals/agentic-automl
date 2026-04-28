from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

project = "Agentic AutoML"
author = "andalenavals + Codex"
copyright = "2026, andalenavals + Codex"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_baseurl = "https://andalenavals.github.io/agentic-automl/"
html_static_path = ["_static"]
