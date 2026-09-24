"""Sphinx configuration."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from rwpytools._version import __version__  # noqa: E402


project = "rwpytools"
author = "Robot Wealth"
release = __version__
copyright = "%Y, Robot Wealth"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"rwpytools {release}"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "httpx": ("https://www.python-httpx.org", None),
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = False

myst_enable_extensions = ["colon_fence"]
