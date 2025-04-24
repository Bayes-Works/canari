import os
import sys

sys.path.insert(0, os.path.abspath("/Users/vuongdai/GitHub/canari/src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "canari"
copyright = "2025, Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
author = "Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
release = "v.0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-import docstrings
    "sphinx.ext.napoleon",  # Google/NumPy style
    "sphinx.ext.viewcode",  # Add source links
    "sphinx.ext.autosummary",  # Summary tables (optional)
]

templates_path = ["_templates"]
exclude_patterns = []

language = "EN"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True  # Enable if using autosummary
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
