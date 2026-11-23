"""Convert reports/Project_Report.md into a formatted Word document.

Usage (from the project root):
    python scripts/make_report_docx.py

Supports the Markdown used in the report: headings, paragraphs, **bold**,
`code`, bullet / numbered lists, block quotes, tables, code blocks and
images.  Images that do not exist yet are replaced by a grey "figure pending"
note, and <!-- RESULTS PENDING ... --> comments become a highlighted note so
nothing is silently missing.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "Project_Report.md"
OUT = ROOT / "reports" / "Project_Report.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x80, 0x80, 0x80)
INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]+\))")


def add_inline(paragraph, text: str) -> None:
    """Add text to a paragraph, honouring **bold**, *italic*, `code` and [links](url)."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Menlo"
            run.font.size = Pt(9)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            paragraph.add_run(part[1:-1]).italic = True
        elif part.startswith("["):
            label = part[1:part.index("]")]
            paragraph.add_run(label).underline = True
        else:
            paragraph.add_run(part)


def shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc: Document, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    n_cols = max(len(r) for r in cells)
    table = doc.add_table(rows=len(cells), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        for j in range(n_cols):
            cell = table.cell(i, j)
            para = cell.paragraphs[0]
            text = row[j] if j < len(row) else ""
            add_inline(para, text)
            for run in para.runs:
                run.font.size = Pt(8.5)
                if i == 0:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                if text == "PENDING":
                    run.font.color.rgb = GREY
                    run.italic = True
            if i == 0:
                shade(cell, "1F4E79")
            elif i % 2 == 0:
                shade(cell, "EAF1F8")
    doc.add_paragraph()


def add_image(doc: Document, alt: str, rel_path: str) -> None:
    path = (SRC.parent / rel_path).resolve()
    if path.exists():
        doc.add_picture(str(path), width=Cm(15))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(f"Figure: {alt}")
        run.italic = True
        run.font.size = Pt(9)
    else:
        para = doc.add_paragraph()
        run = para.add_run(f"[Figure pending: {alt} — {path.name}]")
        run.italic = True
        run.font.color.rgb = GREY


def add_pending_note(doc: Document, comment: str) -> None:
    para = doc.add_paragraph()
    run = para.add_run(f"[{comment}]")
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0xC0, 0x50, 0x00)


SPECIAL = re.compile(r"^\s*([-*] |\d+\. |#|\||>|!\[|```|<!--|---)")


def gather(lines: list[str], i: int, first: str) -> tuple[str, int]:
    """Join wrapped Markdown lines that belong to the same paragraph / list item."""
    text = first
    while i + 1 < len(lines) and lines[i + 1].strip() and not SPECIAL.match(lines[i + 1]):
        i += 1
        text += " " + lines[i].strip()
    return text, i


def convert(src: Path = SRC, out: Path = OUT) -> Path:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    for section in doc.sections:
        section.left_margin = section.right_margin = Cm(2.2)

    lines = src.read_text(encoding="utf-8").splitlines()
    i, first_heading = 0, True
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):                       # code block
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            para = doc.add_paragraph()
            run = para.add_run("\n".join(block))
            run.font.name = "Menlo"
            run.font.size = Pt(8.5)
        elif stripped.startswith("<!--"):                    # pending marker
            comment = stripped
            while "-->" not in comment and i + 1 < len(lines):
                i += 1
                comment += " " + lines[i].strip()
            text = comment.removeprefix("<!--").split("-->")[0].strip()
            if "PENDING" in text:
                add_pending_note(doc, text)
        elif stripped.startswith("#"):                       # heading
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if first_heading:
                title = doc.add_heading(text, level=0)
                title.alignment = WD_ALIGN_PARAGRAPH.CENTER
                first_heading = False
            else:
                heading = doc.add_heading(text, level=min(level - 1, 3))
                for run in heading.runs:
                    run.font.color.rgb = ACCENT
        elif stripped.startswith("|"):                        # table
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            add_table(doc, rows)
            continue
        elif m := re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped):
            add_image(doc, m.group(1), m.group(2))
        elif stripped in ("---", "***"):
            pass                                              # section rule: skip
        elif stripped.startswith(">"):
            text, i = gather(lines, i, stripped.lstrip("> ").strip())
            para = doc.add_paragraph()
            add_inline(para, text)
            para.paragraph_format.left_indent = Cm(1)
            for run in para.runs:
                run.italic = True
        elif re.match(r"^\s*[-*] ", line):
            indent = (len(line) - len(line.lstrip())) // 2
            para = doc.add_paragraph(style="List Bullet 2" if indent else "List Bullet")
            text, i = gather(lines, i, re.sub(r"^\s*[-*] ", "", line))
            add_inline(para, text)
        elif re.match(r"^\s*\d+\. ", line):
            para = doc.add_paragraph(style="List Number")
            text, i = gather(lines, i, re.sub(r"^\s*\d+\. ", "", line))
            add_inline(para, text)
        elif stripped:
            text, i = gather(lines, i, stripped)
            para = doc.add_paragraph()
            add_inline(para, text)
        i += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


if __name__ == "__main__":
    print(f"Saved {convert()}")
