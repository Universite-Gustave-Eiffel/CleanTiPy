# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())





### Copy notebooks #### 
import os
source_notebooks = os.path.abspath('../../examples/')
destination_notebooks = os.path.abspath('./notebook/')

# create folder if not existing
if not os.path.exists(destination_notebooks):
    os.makedirs(destination_notebooks)

# Copy files
for file in os.listdir(source_notebooks):
    if file.endswith(".ipynb"):
        src = os.path.join(source_notebooks, file)
        dst = os.path.join(destination_notebooks, file)
        
        # Adapted to each OS
        if os.name == 'nt':  # Windows
            os.system(f'copy "{src}" "{dst}"')
        else:  # Linux/Mac
            os.system(f'cp "{src}" "{dst}"')







# def setup(app):
#     app.add_css_file('my_style.css')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CleanTiPy'
copyright = '2025, Raphaël LEIBA'
author = 'Raphaël LEIBA'
release = '0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     'sphinx_rtd_theme',
#     'sphinx.ext.duration',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx_design',
#     'nbsphinx',
# ]
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx_design',
    'nbsphinx',
]
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.ipynb': 'nbsphinx'
#     }
nbsphinx_execute = 'never'
nbsphinx_markup_language = "reStructuredText"
# nb_execution_mode = "off"




templates_path = ['_templates']
exclude_patterns = []
html_css_files = ["my_style.css"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = "_static/CLEAN-T_Logo_1_bw_dark_bg.svg"

pygments_style = 'sphinx'

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}

# toctree_only = True

# latex_engine = 'lualatex'
latex_logo = '_static/CLEAN-T_Logo_1_bw_white_bg.pdf'
# latex_engine = 'xelatex'
latex_elements = {
    'passoptionstopackages': r'''
\PassOptionsToPackage{svgnames}{xcolor}
''',
    'preamble': r'''
\AddToHook{cmd/section/before}{\clearpage}
\usepackage[titles]{tocloft}
''',
    # 'sphinxsetup': 'TitleColor=DarkGoldenrod',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}
latex_show_urls = 'footnote'
latex_toplevel_sectioning =  'section'
latex_theme = 'howto'

