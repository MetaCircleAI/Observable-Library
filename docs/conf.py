from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Observable Library"
author = "Jinxin"
copyright = "2026, Observable Library contributors"
version = "0.1.0"
release = version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "superpowers", "Thumbs.db", ".DS_Store"]
nitpicky = True

language = os.environ.get("DOCS_LANGUAGE", "en")
locale_dirs = ["locale/"]
gettext_compact = False
gettext_additional_targets = ["literal-block"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["css/tokens.css", "css/site.css"]
html_js_files = ["js/site.js"]
html_sidebars = {"**": []}
html_context = {"default_mode": "auto"}
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navbar_align": "left",
    "navbar_start": ["components/navbar-logo.html"],
    "navbar_center": ["components/navbar-nav.html"],
    "navbar_persistent": [
        "components/search-button-field.html",
        "components/theme-switcher.html",
    ],
    "navbar_end": [
        "components/language-switcher.html",
        "components/navbar-icon-links.html",
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MetaCircleAI/Observable-Library",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}
html_last_updated_fmt = None
html_show_sourcelink = False
html_copy_source = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
