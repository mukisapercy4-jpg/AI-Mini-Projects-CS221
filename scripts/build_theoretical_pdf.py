"""Typeset the consolidated Markdown theory draft as one accessible PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "theoretical" / "theoretical_part_master.md"
OUTPUT = ROOT / "theoretical" / "theoretical_solutions.pdf"
FONT_DIR = Path("C:/Windows/Fonts")


def inline_markup(text: str) -> str:
    """Convert common Markdown and LaTeX notation into safe reportlab markup."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)
    out = re.sub(r"\$\$(.+?)\$\$", r"\1", out)
    out = re.sub(r"\$([^$]+)\$", r"\1", out)
    out = re.sub(r"\\\((.+?)\\\)", r"\1", out)
    out = re.sub(r"\\\[(.+?)\\\]", r"\1", out)
    replacements = {
        r"\text{": "", r"\mathrm{": "", r"\mathbf{": "", r"\mathcal{": "",
        r"\operatorname{": "", r"\hat{": "", r"\bar{": "", r"\vec{": "",
        r"\begin{cases}": "cases: ", r"\end{cases}": "",
        r"\cdot": " dot ", r"\times": " times ", r"\nabla": "grad", r"\sum": "sum over",
        r"\prod": "product over", r"\max": "max", r"\min": "min", r"\argmax": "arg max",
        r"\argmin": "arg min", r"\log": "log", r"\exp": "exp", r"\sqrt": "√",
        r"\rightarrow": "to", r"\Rightarrow": "implies", r"\implies": "implies", r"\iff": "iff",
        r"\leq": "<=", r"\geq": ">=", r"\le": "<=", r"\ge": ">=", r"\neq": "!=", r"\approx": "approx",
        r"\in": "in", r"\notin": "not in", r"\forall": "for all", r"\exists": "there exists",
        r"\mid": "|", r"\cdots": "…", r"\ldots": "…", r"\infty": "∞",
        r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\lambda": "λ",
        r"\eta": "η", r"\theta": "θ", r"\phi": "φ", r"\pi": "π",
        r"\mu": "μ", r"\sigma": "σ", r"\epsilon": "ε", r"\Delta": "Δ",
        r"\mathbb{R}": "R", r"\text{nnz}": "nnz", r"\%": "%", r"\_": "_",
        r"\lceil": "ceil", r"\rceil": "ceil", r"\lfloor": "floor", r"\rfloor": "floor",
        r"\{": "{", r"\}": "}", r"\|": "‖", r"\,": " ", r"\;": " ",
        r"\!": "", r"\: ": " ", r"\quad": "  ", r"\qquad": "    ",
        r"\not\models": "does not entail", r"\models": "entails", r"\vdash": "|-",
        r"\land": " AND ", r"\lor": " OR ", r"\wedge": " AND ", r"\vee": " OR ",
        r"\neg": "NOT ", r"\lnot": "NOT ", r"\top": "TRUE", r"\bot": "FALSE",
        r"\notin": " not in ", r"\in": " in ", r"\not\equiv": "not equivalent to",
        r"\equiv": "equivalent to", r"\perp": "independent of", r"\left": "", r"\right": "",
        r"\leftrightarrow": "iff", r"\leftarrow": "from", r"\to": "to", r"\not": "not ",
        r"\square": "empty clause", r"\subseteq": "subset of or equal to", r"\subset": "subset of",
        r"\textbf{": "", r"\emph{": "",
    }
    for old in sorted(replacements, key=len, reverse=True):
        new = replacements[old]
        out = out.replace(old, new)
    out = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", out)
    out = out.replace(r"\\", "; ").replace("&amp;", " ")
    out = re.sub(r"([A-Za-z0-9)])_\{([^{}]+)\}", r"\1<sub>\2</sub>", out)
    out = re.sub(r"([A-Za-z0-9)])_([A-Za-z0-9])", r"\1<sub>\2</sub>", out)
    out = re.sub(r"([A-Za-z0-9)])\^\{([^{}]+)\}", r"\1<super>\2</super>", out)
    out = re.sub(r"([A-Za-z0-9)])\^([A-Za-z0-9])", r"\1<super>\2</super>", out)
    out = out.replace("**", "§B§").replace("*", "§I§")
    out = re.sub(r"§B§(.+?)§B§", r"<b>\1</b>", out)
    out = re.sub(r"§I§(.+?)§I§", r"<i>\1</i>", out)
    out = re.sub(r"`([^`]+)`", r"<font name='CourseMono'>\1</font>", out)
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("{", "").replace("}", "")
    return out


def parse_table(lines: list[str], styles: dict, available_width: float) -> Table:
    rows = []
    for row_index, line in enumerate(lines):
        if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", line):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        style = styles["table_header"] if row_index == 0 else styles["table"]
        rows.append([Paragraph(inline_markup(cell), style) for cell in cells])
    col_width = available_width / max(1, max(map(len, rows)))
    table = Table(rows, colWidths=[col_width] * max(map(len, rows)), repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F6F9")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D2DC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build() -> None:
    pdfmetrics.registerFont(TTFont("CourseSans", str(FONT_DIR / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("CourseSans-Bold", str(FONT_DIR / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("CourseSans-Italic", str(FONT_DIR / "ariali.ttf")))
    pdfmetrics.registerFont(TTFont("CourseMono", str(FONT_DIR / "consola.ttf")))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="title2", parent=styles["Title"], fontName="CourseSans-Bold", fontSize=19, leading=24, textColor=colors.HexColor("#17324D"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="subtitle2", parent=styles["Normal"], fontName="CourseSans", fontSize=11, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#536679"), spaceAfter=6))
    styles.add(ParagraphStyle(name="h1x", parent=styles["Heading1"], fontName="CourseSans-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#17324D"), spaceBefore=12, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name="h2x", parent=styles["Heading2"], fontName="CourseSans-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#225B83"), spaceBefore=9, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="h3x", parent=styles["Heading3"], fontName="CourseSans-Bold", fontSize=10.5, leading=14, textColor=colors.HexColor("#34799F"), spaceBefore=7, spaceAfter=3, keepWithNext=True))
    styles.add(ParagraphStyle(name="bodyx", parent=styles["BodyText"], fontName="CourseSans", fontSize=9, leading=13, alignment=TA_LEFT, spaceAfter=5))
    styles.add(ParagraphStyle(name="bulletx", parent=styles["BodyText"], fontName="CourseSans", fontSize=8.8, leading=12.5, leftIndent=14, firstLineIndent=-8, spaceAfter=3))
    styles.add(ParagraphStyle(name="table", parent=styles["BodyText"], fontName="CourseSans", fontSize=7.2, leading=9.5, textColor=colors.HexColor("#202B33")))
    styles.add(ParagraphStyle(name="table_header", parent=styles["BodyText"], fontName="CourseSans-Bold", fontSize=7.2, leading=9.5, textColor=colors.white))
    styles.add(ParagraphStyle(name="equation", parent=styles["BodyText"], fontName="CourseSans", fontSize=8.8, leading=12.5, leftIndent=14, rightIndent=10, backColor=colors.HexColor("#F2F6FA"), borderPadding=5, spaceBefore=3, spaceAfter=6))

    width, height = A4
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=21*mm, rightMargin=21*mm, topMargin=20*mm, bottomMargin=19*mm, title="AI Mini-Projects Theoretical Solutions", author="Makerere University AI Coursework Team")
    frame = Frame(doc.leftMargin, doc.bottomMargin, width-doc.leftMargin-doc.rightMargin, height-doc.topMargin-doc.bottomMargin, id="normal")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#C8D2DC"))
        canvas.line(21*mm, 14*mm, width-21*mm, 14*mm)
        canvas.setFont("CourseSans", 8)
        canvas.setFillColor(colors.HexColor("#536679"))
        canvas.drawString(21*mm, 9*mm, "Makerere University | CS221-aligned AI Mini-Projects")
        canvas.drawRightString(width-21*mm, 9*mm, f"Page {document.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="coursework", frames=frame, onPage=footer)])
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    first_heading_done = False
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            story.append(parse_table(table_lines, styles, width-doc.leftMargin-doc.rightMargin))
            story.append(Spacer(1, 6))
            continue
        if line == "$$" or line.startswith("$$") or line == r"\[" or line.startswith(r"\["):
            opening = "$$" if line.startswith("$$") else r"\["
            closing = "$$" if opening == "$$" else r"\]"
            first = line[len(opening):].strip()
            if first.endswith(closing):
                display_lines = [first[:-len(closing)].strip()]
            else:
                display_lines = [first] if first else []
                i += 1
                while i < len(lines) and closing not in lines[i].strip():
                    display_lines.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    display_lines.append(lines[i].strip().removesuffix(closing).strip())
            display = " ".join(display_lines)
            display = display.replace("&", " ")
            story.append(Paragraph(inline_markup(display), styles["equation"]))
            i += 1
            continue
        if line.startswith("# "):
            if first_heading_done:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(line[2:]), styles["title2"] if not first_heading_done else styles["h1x"]))
            first_heading_done = True
            i += 1
            continue
        if line.startswith("## "):
            if re.match(r"## (Section|Assignment)\s+[1-7]", line):
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(line[3:]), styles["h1x"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["h2x"]))
            i += 1
            continue
        if line.startswith("#### "):
            story.append(Paragraph(inline_markup(line[5:]), styles["h3x"]))
            i += 1
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), styles["equation"]))
            i += 1
            continue
        if line.startswith("- ") or re.match(r"\d+\. ", line):
            marker = "•" if line.startswith("- ") else ""
            content = line[2:] if marker else re.sub(r"^\d+\.\s+", "", line)
            story.append(Paragraph(f"{marker} {inline_markup(content)}", styles["bulletx"]))
            i += 1
            continue
        if line.startswith("```" ):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(Paragraph(inline_markup("<br/>".join(code_lines)), styles["equation"]))
            i += 1
            continue
        story.append(Paragraph(inline_markup(line), styles["bodyx"]))
        i += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes).")


if __name__ == "__main__":
    build()
