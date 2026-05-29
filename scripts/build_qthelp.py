#!/usr/bin/env python3
"""Compile ``docs/HELP.md`` into a Qt Compressed Help (.qch + .qhc) bundle.

Splits the Markdown source on level-2 headings into one HTML file per
section, copies the image assets it references, writes a ``.qhp``
project and a ``.qhcp`` collection project, and runs ``qhelpgenerator``
to produce the binary archives consumed by the in-app help dialog.

Outputs ``sticker_creator/resources/help/{sticker_creator.qch,
sticker_creator.qhc}``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
HELP_MD = ROOT / "docs" / "HELP.md"
DOCS_DIR = ROOT / "docs"
OUT_DIR = ROOT / "sticker_creator" / "resources" / "help"
BUILD_DIR = ROOT / "build" / "qthelp"

QHELP_BIN_CANDIDATES = [
    "/usr/lib64/qt6/libexec/qhelpgenerator",
    "/usr/lib/qt6/libexec/qhelpgenerator",
    "qhelpgenerator",
]

NAMESPACE = "org.kde.stickercreator"
VIRTUAL_FOLDER = "sticker-creator"
COLLECTION_STEM = "sticker_creator"


def _slugify(text: str) -> str:
    t = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_-]+", "-", t)


def _find_qhelpgenerator() -> str:
    for cand in QHELP_BIN_CANDIDATES:
        path = shutil.which(cand) if "/" not in cand else cand
        if path and Path(path).exists():
            return path
    print("qhelpgenerator not found — install qt6-doctools.")
    sys.exit(1)


def _split_sections(md: str) -> list[tuple[str, str]]:
    """Split MD on ``^## `` lines. Returns [(title, body), …].

    The chunk before the first ``##`` is emitted as the ``"Overview"`` page
    so the first H1 (``# Sticker Creator — Help``) becomes its content.
    """
    lines = md.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current: list[str] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((current_title, current))
            current_title = line[3:].strip()
            current = []
        else:
            current.append(line)
    sections.append((current_title, current))
    return [(title, "\n".join(body)) for title, body in sections]


HTML_TEMPLATE = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{ font-family: sans-serif; max-width: 820px; margin: 1em auto;
                line-height: 1.5; }}
        h1 {{ padding-bottom: 4px; }}
        code, pre {{ border-radius: 4px; }}
        code {{ padding: 1px 5px; }}
        pre {{ padding: 10px 12px; overflow-x: auto; }}
        table {{ border-collapse: collapse; margin: 8px 0; }}
        th, td {{ padding: 6px 10px; text-align: left; }}
        img {{ max-width: 100%; height: auto; border-radius: 6px; }}
      </style>
    </head>
    <body>
    {body}
    </body>
    </html>
""")


def _convert_section(title: str, body_md: str) -> str:
    html_body = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="xhtml",
    )
    if not body_md.lstrip().startswith("#"):
        # Add an H1 if the section has no top-level heading already.
        html_body = f"<h1>{title}</h1>\n{html_body}"
    return HTML_TEMPLATE.format(title=title, body=html_body)


KEYWORDS: dict[str, list[str]] = {
    "Overview":               ["overview", "introduction", "what is"],
    "Quick tour":             ["quick tour", "tour", "demo"],
    "1. Open an image":       ["open", "image", "load", "paste", "drag and drop"],
    "2. Cut out the subject — segmentation":
                              ["segmentation", "cut out", "mask", "click",
                               "foreground", "background", "positive", "negative"],
    "3. Border and preview background":
                              ["border", "preview background", "white border"],
    "4. Speech balloon (optional)":
                              ["balloon", "speech", "thought", "shout", "whisper",
                               "text"],
    "5. Save the sticker":    ["save", "export", "png", "webp", "whatsapp"],
    "AI model: SAM 2":        ["SAM 2", "AI model", "segment anything",
                               "checkpoint", "hiera", "tiny", "small",
                               "base_plus", "large"],
    "Model manager":          ["model manager", "download", "switch", "remove"],
    "Keyboard shortcuts":     ["shortcuts", "keyboard", "hotkeys"],
    "Troubleshooting":        ["troubleshooting", "problem", "error",
                               "corrupt", "slow"],
}


def _qhp_xml(files: list[Path], sections: list[tuple[str, Path]]) -> str:
    file_entries = "\n".join(
        f"      <file>{p.relative_to(BUILD_DIR).as_posix()}</file>" for p in files
    )
    toc_entries = "\n".join(
        f'      <section title="{title}" ref="{path.name}"/>'
        for title, path in sections
    )
    kw_entries: list[str] = []
    for title, path in sections:
        for kw in KEYWORDS.get(title, []):
            kw_entries.append(
                f'      <keyword name="{kw}" ref="{path.name}"/>'
            )
    keywords_xml = "\n".join(kw_entries)

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<QtHelpProject version="1.0">\n'
        f'  <namespace>{NAMESPACE}</namespace>\n'
        f'  <virtualFolder>{VIRTUAL_FOLDER}</virtualFolder>\n'
        '  <filterSection>\n'
        '    <toc>\n'
        f'      <section title="Sticker Creator Help" ref="{sections[0][1].name}">\n'
        f'{toc_entries}\n'
        '      </section>\n'
        '    </toc>\n'
        '    <keywords>\n'
        f'{keywords_xml}\n'
        '    </keywords>\n'
        '    <files>\n'
        f'{file_entries}\n'
        '    </files>\n'
        '  </filterSection>\n'
        '</QtHelpProject>\n'
    )


def _qhcp_xml(qch_path: Path) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<QHelpCollectionProject version="1.0">\n'
        '  <assistant>\n'
        '    <title>Sticker Creator Help</title>\n'
        f'    <homePage>qthelp://{NAMESPACE}/{VIRTUAL_FOLDER}/index.html</homePage>\n'
        f'    <startPage>qthelp://{NAMESPACE}/{VIRTUAL_FOLDER}/index.html</startPage>\n'
        '    <enableFullTextSearchFallback>true</enableFullTextSearchFallback>\n'
        '  </assistant>\n'
        '  <docFiles>\n'
        '    <register>\n'
        f'      <file>{qch_path.name}</file>\n'
        '    </register>\n'
        '  </docFiles>\n'
        '</QHelpCollectionProject>\n'
    )


def main() -> int:
    if not HELP_MD.exists():
        print(f"missing source: {HELP_MD}")
        return 1
    qhg = _find_qhelpgenerator()
    print(f"using {qhg}")

    md_text = HELP_MD.read_text(encoding="utf-8")
    sections_md = _split_sections(md_text)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    written_html: list[tuple[str, Path]] = []
    referenced_assets: set[Path] = set()

    for i, (title, body_md) in enumerate(sections_md):
        if i == 0:
            file_name = "index.html"
        else:
            file_name = f"{i:02d}-{_slugify(title)}.html"
        html = _convert_section(title, body_md)
        path = BUILD_DIR / file_name
        path.write_text(html, encoding="utf-8")
        written_html.append((title, path))
        # Track image references for copy-in.
        for m in re.finditer(r'src="([^"]+)"', html):
            ref = m.group(1)
            if ref.startswith(("http://", "https://", "qthelp://")):
                continue
            referenced_assets.add(Path(ref))

    # Copy referenced images alongside the HTML files, preserving relative paths.
    copied_assets: list[Path] = []
    for rel in referenced_assets:
        src = DOCS_DIR / rel
        if not src.exists():
            print(f"  WARN: asset missing: {src}")
            continue
        dst = BUILD_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_assets.append(dst)

    all_files = [p for _, p in written_html] + copied_assets
    qhp_path = BUILD_DIR / f"{COLLECTION_STEM}.qhp"
    qhp_path.write_text(_qhp_xml(all_files, written_html), encoding="utf-8")

    qch_path = BUILD_DIR / f"{COLLECTION_STEM}.qch"
    print(f"building {qch_path.name} …")
    subprocess.run([qhg, str(qhp_path), "-o", str(qch_path)], check=True)

    qhcp_path = BUILD_DIR / f"{COLLECTION_STEM}.qhcp"
    qhcp_path.write_text(_qhcp_xml(qch_path), encoding="utf-8")
    qhc_path = BUILD_DIR / f"{COLLECTION_STEM}.qhc"
    print(f"building {qhc_path.name} …")
    subprocess.run([qhg, str(qhcp_path), "-o", str(qhc_path)], check=True)

    # Publish into the package resources directory.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (qch_path, qhc_path):
        dst = OUT_DIR / p.name
        shutil.copy2(p, dst)
        print(f"published {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
