import os
import sys

sys.path.insert(0, os.path.abspath("/Users/vuongdai/GitHub/canari/src"))

project = "canari"
copyright = "2025, Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
author = "Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
release = "v.0.0.2"

extensions = [
    "sphinx.ext.autodoc",  # Auto-import docstrings
    "sphinx.ext.napoleon",  # Google/NumPy style
    "sphinx.ext.viewcode",  # Add source links
    "sphinx.ext.autosummary",  # Summary tables (optional)
    "nbsphinx",
    "sphinx.ext.mathjax",
    # "sphinx_rtd_theme",
    "sphinx_book_theme",
]
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = []

language = "EN"

autosummary_generate = False
html_theme = "sphinx_book_theme"
# html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,  # don’t collapse everything
    "show_toc_level": 2,
    "sticky_navigation": True,
    "navigation_depth": 5,  # show up to 3 levels in the sidebar
    "includehidden": True,
    "titles_only": False,  # ← must be False to show per-page sections
}
html_title = "Canari Documentation"
