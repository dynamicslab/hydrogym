# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information

project = "Hydrogym"
copyright = "2024, Hydrogym Developers"
author = "The HydroGym Developers"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
autosummary_generate = True

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
autodoc_mock_imports = [
    "firedrake",
    "pyadjoint",
    "ufl",
    "mpi4py",
]

# -- Options for HTML output

html_theme = "sphinx_book_theme"

# Name of the image file
html_logo = "_static/imgs/logo.svg"
html_favicon = "_static/imgs/logo.svg"

# Custom paths for static files
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/dynamicslab/hydrogym",
    "use_repository_button": True,  # add a 'link to repository' button
    "use_issues_button": True,  # add an 'Open an Issue' button
    "path_to_docs": ("docs"),  # used to compute the path to launch notebooks in colab
    "prev_next_buttons_location": None,
    "show_navbar_depth": 1,
}

# -- Options for EPUB output
epub_show_urls = "footnote"
