# -*- coding: utf-8 -*-

# See sphinx.builders.html for configuration builder
# http://www.sphinx-doc.org/en/stable/config.html

import datetime
import lerp

# -- General configuration ------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.imgconverter',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'nbsphinx',
    ]

author = 'gwin-zegal'
copyright = f'{datetime.datetime.now().year}, {author}'
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
project = 'lerp'
version = f'{lerp.__version__}'
release = ''
exclude_patterns = ['_build',
                    'Thumbs.db',
                    '.DS_Store',
                    '**.ipynb_checkpoints',
                    ]
pygments_style = 'sphinx'
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_path = ["_themes", ]
html_theme_options = {
    # "project_nav_name": "lerp",
    # 'typekit_id': '',
    # 'canonical_url': '',
    # 'analytics_id': '',
    'display_version': False,
    'prev_next_buttons_location': 'bottom',
    # 'style_external_links': False,
    # Toc options
    # 'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    # 'includehidden': True,
    # 'titles_only': False,
}
html_static_path = ['_static']
htmlhelp_basename = 'lerpdoc'
html_show_copyright = False
html_show_sphinx = False

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '\usepackage{svg}',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'lerp.tex', 'lerp Documentation',
     'ER', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'lerp', 'lerp Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'lerp', 'lerp Documentation',
     author, 'lerp', 'One line description of project.',
     'Miscellaneous'),
]


# http://www.sphinx-doc.org/en/stable/extdev/appapi.html
def setup(app):
    app.add_stylesheet('https://fonts.googleapis.com/css?family=Open+Sans:600,400,300,200|Inconsolata|Ubuntu+Mono:400,700')
    app.add_stylesheet('css/jupyter-notebook-custom.css')
    app.add_stylesheet('css/api.css')
