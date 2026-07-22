from __future__ import annotations

from pathlib import Path

from babel.messages.pofile import read_po

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from scripts.build_docs import build_site


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
WEB_DOCS = [
    DOCS / "index.md",
    DOCS / "installation.md",
    DOCS / "quickstart.md",
    DOCS / "concepts.md",
    DOCS / "usage.md",
    DOCS / "api.md",
    DOCS / "limitations.md",
]
FORBIDDEN_DOCS_TEXT = [
    "ParameterNormObservable",
    "UpdateRatioObservable",
    "from observable_library import generate, observe",
    "BSD-3-Clause",
]
LOCALE_DIR = DOCS / "locale" / "zh_CN" / "LC_MESSAGES"


def test_docs_extra_declares_the_sphinx_toolchain() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    docs = metadata["project"]["optional-dependencies"]["docs"]

    for package in [
        "babel",
        "sphinx",
        "myst-parser",
        "pydata-sphinx-theme",
        "sphinx-design",
        "sphinx-copybutton",
        "sphinx-intl",
        "pytest",
    ]:
        assert any(item.lower().startswith(package) for item in docs)


def test_build_site_creates_one_documentation_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"

    build_site(ROOT, output_dir)

    for relative in ["index.html", "zh/index.html", "404.html"]:
        assert (output_dir / relative).is_file()


def test_web_docs_have_the_complete_top_level_navigation() -> None:
    assert all(path.is_file() for path in WEB_DOCS)
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    entries = [
        "installation",
        "quickstart",
        "concepts",
        "usage",
        "api",
        "limitations",
    ]

    positions = [index.index(f"\n{entry}\n") for entry in entries]
    assert positions == sorted(positions)


def test_web_docs_publish_only_the_real_0_1_0_contract() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in WEB_DOCS)

    for required in [
        "0.1.0",
        "torch>=2.4.1",
        "Apache-2.0",
        "Runtime.observe()",
        'selector="all"',
    ]:
        assert required in text
    for forbidden in FORBIDDEN_DOCS_TEXT:
        assert forbidden not in text


def test_zh_catalogs_translate_every_prose_message() -> None:
    catalogs = sorted(LOCALE_DIR.glob("*.po"))
    assert {path.stem for path in catalogs} == {
        "404",
        "api",
        "concepts",
        "index",
        "installation",
        "limitations",
        "quickstart",
        "usage",
    }

    for catalog_path in catalogs:
        with catalog_path.open(encoding="utf-8") as stream:
            catalog = read_po(stream, locale="zh_CN")
        assert catalog.locale_identifier == "zh_CN"
        for message in catalog:
            if not message.id:
                continue
            assert message.string
            assert not message.fuzzy
            if message.string == message.id:
                assert "\n" in message.id or (
                    message.id.startswith("`") and message.id.endswith("`")
                )


def test_rendered_homepages_load_the_approved_design_assets(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"
    build_site(ROOT, output_dir)

    english = (output_dir / "index.html").read_text(encoding="utf-8")
    chinese = (output_dir / "zh" / "index.html").read_text(encoding="utf-8")
    for html in [english, chinese]:
        for asset in ["css/tokens.css", "css/site.css", "js/site.js"]:
            assert asset in html
        assert "hero-live-code" in html
        assert "data-doc-language-switch" in html
        assert "ParameterNormObservable" not in html
        assert "UpdateRatioObservable" not in html
        assert "BSD-3-Clause" not in html

    assert "Generate and compute training observables" in english
    assert "生成并计算训练可观测量" in chinese


def test_responsive_header_hides_controls_outside_their_layout() -> None:
    css = (DOCS / "_static" / "css" / "site.css").read_text(encoding="utf-8")

    assert "#pst-header .drawer-header {\n  display: none !important;\n}" in css

    tablet_rules = css[css.index("@media (min-width: 901px) and (max-width: 1279px)") :]
    assert (
        "#pst-header .secondary-toggle {\n    display: none !important;\n  }"
        in tablet_rules
    )

    compact_header_rules = css[css.index("@media (max-width: 1279px)") :]
    assert (
        "#pst-header .navbar-header-items__center {\n    display: none !important;\n  }"
        in compact_header_rules
    )


def test_fragment_navigation_does_not_leave_a_persistent_heading_highlight() -> None:
    css = (DOCS / "_static" / "css" / "site.css").read_text(encoding="utf-8")

    assert ":target > :is(h1, .h1, h2, .h2, h3, .h3, h4, .h4, h5, .h5)" in css
    assert "background-color: transparent;" in css


def test_flat_navigation_is_not_duplicated_in_the_primary_sidebar() -> None:
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")

    assert 'html_sidebars = {"**": []}' in conf


def test_compact_header_aligns_actions_and_handles_a_320px_viewport() -> None:
    css = (DOCS / "_static" / "css" / "site.css").read_text(encoding="utf-8")

    compact_header_rules = css[css.index("@media (max-width: 1279px)") :]
    assert (
        "#pst-header .navbar-header-items__end {\n    margin-left: auto;\n  }"
        in compact_header_rules
    )

    narrow_header_rules = css[css.index("@media (max-width: 360px)") :]
    assert ".docs-wordmark__name {\n    max-width: 7.25rem;\n  }" in narrow_header_rules


def test_mobile_keeps_page_navigation_available_without_crowding_the_header() -> None:
    css = (DOCS / "_static" / "css" / "site.css").read_text(encoding="utf-8")

    tablet_rules = css[
        css.index("@media (max-width: 900px)") : css.index("@media (max-width: 767px)")
    ]
    assert ".bd-sidebar-secondary {\n    display: none;\n  }" not in tablet_rules

    mobile_rules = css[css.index("@media (max-width: 767px)") :]
    assert (
        "#pst-primary-sidebar-modal .drawer-header .docs-wordmark__version {\n"
        "    display: none;\n  }" in mobile_rules
    )

    narrow_rules = css[css.index("@media (max-width: 400px)") :]
    assert (
        "#pst-header .navbar-persistent--mobile:has(.theme-switch-container) {\n"
        "    display: none;\n  }" in narrow_rules
    )
    assert (
        "#pst-header .navbar-header-items__start .docs-wordmark__version {\n"
        "    display: none;\n  }" in narrow_rules
    )


def test_mobile_workflow_reflows_instead_of_requiring_horizontal_scroll() -> None:
    css = (DOCS / "_static" / "css" / "site.css").read_text(encoding="utf-8")

    mobile_rules = css[css.index("@media (max-width: 767px)") :]
    assert (
        ".workflow-steps {\n"
        "    grid-template-columns: minmax(0, 1fr);\n"
        "    min-width: 0;\n"
        "  }" in mobile_rules
    )
    assert (
        ".workflow-arrow {\n"
        "    justify-self: center;\n"
        "    transform: rotate(90deg);\n"
        "  }" in mobile_rules
    )
