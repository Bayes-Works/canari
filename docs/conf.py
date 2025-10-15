# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path
from sphinx.highlighting import lexers
from pygments.lexers.python import Python3Lexer

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

# Add the src directory to sys.path for autodoc
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

# ---------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------

project = "canari"
copyright = "2025, Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
author = "Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
release = "v0.2.0"

# ---------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_book_theme",
]

# Prevent notebook execution during build
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

autosummary_generate = False

# ---------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "collapse_navbar": False,
    "show_navbar_depth": 0,
    "max_navbar_depth": 5,
    "show_toc_level": 2,
    "includehidden": True,
    "titles_only": False,
    "repository_url": "https://github.com/Bayes-Works/canari.git",
    "use_repository_button": True,
    "use_download_button": False,
}
html_logo = "_static/canari_logo.png"
html_static_path = ["_static"]

# ---------------------------------------------------------------------
# Handle notebooks located outside docs/
# ---------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent
TUTORIAL_SRC = DOCS_DIR.parent / "examples" / "tutorial"
EXAMPLES_DST = DOCS_DIR / "examples"
EXAMPLES_DST.mkdir(exist_ok=True)

# Symlink notebooks from examples/tutorial → docs/examples/
if TUTORIAL_SRC.exists():
    for nb in TUTORIAL_SRC.glob("*.ipynb"):
        dst = EXAMPLES_DST / nb.name
        if not dst.exists():
            try:
                os.symlink(nb.resolve(), dst)
            except FileExistsError:
                pass
