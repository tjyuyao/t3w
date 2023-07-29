# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 't3w'
copyright = '2023, Yuyao Huang'
author = 'Yuyao Huang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',  # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    'sphinx.ext.autosectionlabel',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'  # https://sphinx-book-theme.readthedocs.io/en/stable/
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # https://sphinx-book-theme.readthedocs.io/en/stable/components/custom-css.html
html_theme_options = {
    "repository_url": "https://github.com/tjyuyao/t3w",
    "use_repository_button": True,
    "show_navbar_depth": 2,
}
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}