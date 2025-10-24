"""Convert Markdown files to styled PDF.

Features:
- Ask (or accept) an input Markdown file
- Convert Markdown -> HTML with sensible extensions
- Apply embedded print-friendly CSS
- Convert HTML -> PDF using WeasyPrint (preferred) or pdfkit (wkhtmltopdf) as fallback
- Save PDF with a clean filename next to the source file (or in an output path)

Usage:
  python markdown_to_pdf.py path/to/file.md
  python markdown_to_pdf.py  # will prompt for path

Notes:
 - WeasyPrint is the preferred renderer. On macOS you may need to install system libs:
   brew install cairo pango gdk-pixbuf libffi
 - If WeasyPrint isn't available the script will try pdfkit (wkhtmltopdf must be installed).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
	# markdown package (Python-Markdown)
	from markdown import Markdown
except Exception:
	Markdown = None


def clean_filename(name: str) -> str:
	"""Return a safe, clean filename (no spaces, lowercased, hyphens)."""
	name = name.strip()
	name = name.lower()
	# replace spaces and underscores with hyphens
	name = re.sub(r"[\s_]+", "-", name)
	# remove characters other than alnum, hyphen, dot
	name = re.sub(r"[^a-z0-9\-\.]+", "", name)
	# collapse multiple hyphens
	name = re.sub(r"-+", "-", name)
	return name


DEFAULT_PAGE_MARGIN_IN = 0.5  # inches; default margin to fit more text on a page


def generate_css(margin_in: float = DEFAULT_PAGE_MARGIN_IN, condensed: bool = False) -> str:
	"""Return the CSS string with the requested page margin in inches.

	If `condensed` is True, use a tighter layout (smaller font, smaller margins,
	reduced spacing) intended to help fit content into fewer pages.
	"""
	# Condensed defaults
	if condensed:
		margin_in = max(0.35, margin_in)  # don't go below 0.35in automatically
		body_font = "10pt"
		line_height = 1.12
		li_margin = "0 0 0.18em 0"
		heading_scale = {
			'h1': '18pt',
			'h2': '13pt',
			'h3': '11.5pt',
		}
		pre_padding = "8px"
	else:
		body_font = "11pt"
		line_height = 1.25
		li_margin = "0 0 0.35em 0"
		heading_scale = {
			'h1': '22pt',
			'h2': '16pt',
			'h3': '13pt',
		}
		pre_padding = "12px"

	margin = f"{margin_in}in"
	return f"""
@page {{ size: letter; margin: {margin} }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color: #222; font-size: {body_font}; line-height: {line_height}; }}
h1 {{ font-size: {heading_scale['h1']}; margin-bottom: 6px }}
h2 {{ font-size: {heading_scale['h2']}; margin-bottom: 4px }}
h3 {{ font-size: {heading_scale['h3']}; margin-bottom: 4px }}
article {{ max-width: 800px; margin: 0 auto }}
pre, code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", "Courier New", monospace; }}
pre {{ background: #f6f8fa; padding: {pre_padding}; border-radius: 6px; overflow: auto }}
code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px }}
img {{ max-width: 100%; height: auto }}
table {{ border-collapse: collapse; width: 100% }}
table, th, td {{ border: 1px solid #ddd }}
th, td {{ padding: 8px; text-align: left }}
blockquote {{ color: #555; border-left: 4px solid #ddd; padding: 0 1em }}
/* Improve list rendering: preserve bullets and proper wrapping/hanging indent */
ul, ol {{
	margin: 0 0 0.9em 1.0em;
	padding-left: 1.0em;
	list-style-position: outside;
}}
li {{
	margin: {li_margin};
	line-height: {line_height};
}}
/* When Markdown emits <p> inside <li>, remove extra paragraph margins so items stay contiguous */
li > p {{ margin: 0 }}
/* Ensure long lines wrap within list items and preserve hanging indent */
li {{ white-space: normal; }}
"""


def markdown_to_html(md_text: str, extensions: list[str] | None = None, margin: float = DEFAULT_PAGE_MARGIN_IN, condensed: bool = False) -> str:
	if Markdown is None:
		raise RuntimeError("The 'markdown' package is required. Install with: pip install markdown")

	# First: normalize excessive blank lines in the source markdown
	# Collapse 3+ consecutive newlines down to 2 (keeps a single blank line between blocks)
	md_text = re.sub(r"\n{3,}", "\n\n", md_text)

	# Normalize common Unicode bullet characters at start of lines to proper Markdown list markers
	# e.g. lines starting with '•', '·', '‣', '⁃', etc. -> '- '
	# This preserves documents where bullets were pasted from rich text editors.
	md_text = re.sub(r"(?m)^[ \t]*[\u2022\u2023\u25E6\u2219\u2043\u2027\u00B7]\s+", "- ", md_text)

	if extensions is None:
		extensions = [
			"fenced_code",
			"codehilite",
			"tables",
			"toc",
		]

	md = Markdown(extensions=extensions, output_format="html")
	html = md.reset().convert(md_text)

	# Post-process list HTML to remove extra blank paragraphs/lines inside lists
	def clean_list_blocks(html_in: str) -> str:
		# Find <ul>...</ul> and <ol>...</ol> blocks and sanitize their inner HTML
		def _clean_match(m: re.Match) -> str:
			open_tag = m.group(1)
			content = m.group(2)
			close_tag = m.group(3)

			# Remove empty paragraphs produced by some markdown inputs
			content = re.sub(r"<p>\s*(?:&nbsp;)?\s*</p>", "", content)
			# Replace adjacent paragraphs inside list items with a single <br/>
			content = re.sub(r"</p>\s*<p>", "<br/>", content)
			# Collapse multiple consecutive <br/> into a single one
			content = re.sub(r"(<br\s*/?>\s*){2,}", "<br/>", content)
			# Remove excessive whitespace between tags inside lists
			content = re.sub(r">\s+<", "><", content)

			return f"{open_tag}{content}{close_tag}"

		pattern = re.compile(r"(<(?:ul|ol)[^>]*>)(.*?)(</(?:ul|ol)>)", re.DOTALL | re.IGNORECASE)
		return pattern.sub(_clean_match, html_in)

	html = clean_list_blocks(html)

	# Wrap in basic HTML template
	full = f"""
<!doctype html>
<html>
<head>
	<meta charset="utf-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<title>Document</title>
	<style>{generate_css(margin, condensed)}</style>
</head>
<body>
	<article>
	{html}
	</article>
</body>
</html>
"""

	return full


def try_weasyprint_render(html: str, output_path: Path, base_url: str | None = None, css_string: str | None = None) -> bool:
	try:
		from weasyprint import HTML, CSS

		css = css_string if css_string is not None else generate_css()
		HTML(string=html, base_url=base_url).write_pdf(str(output_path), stylesheets=[CSS(string=css)])
		return True
	except Exception as e:
		print(f"WeasyPrint render failed: {e}")
		return False


def try_pdfkit_render(html: str, output_path: Path, base_url: str | None = None) -> bool:
	try:
		import pdfkit

		options = {
			"enable-local-file-access": None,
		}
		# If base_url provided, allow relative resources
		pdfkit.from_string(html, str(output_path), options=options)
		return True
	except Exception as e:
		print(f"pdfkit render failed: {e}")
		return False


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Convert Markdown file to PDF (styled)")
	parser.add_argument("input", nargs="?", help="Path to input markdown file")
	parser.add_argument("-o", "--output", help="Output PDF path (optional)")
	parser.add_argument("--condense", action="store_true", help="Try to condense layout to fit fewer pages (tighter margins and smaller font)")
	args = parser.parse_args(argv)

	input_path = args.input
	if not input_path:
		input_path = input("Path to markdown file: ").strip()

	if not input_path:
		print("No input provided. Exiting.")
		return 2

	infile = Path(input_path).expanduser()
	if not infile.exists() or not infile.is_file():
		print(f"Input file not found: {infile}")
		return 3

	text = infile.read_text(encoding="utf-8")
	try:
		html = markdown_to_html(text, condensed=args.condense)
	except RuntimeError as e:
		print(e)
		return 4

	# Determine output path
	if args.output:
		outpath = Path(args.output).expanduser()
		if outpath.is_dir():
			outpath = outpath / (clean_filename(infile.stem) + ".pdf")
	else:
		outname = clean_filename(infile.stem) + ".pdf"
		outpath = infile.parent / outname

	# Try WeasyPrint first, then pdfkit. If both unavailable, write HTML and instruct user.
	base_url = str(infile.parent)

	rendered = False
	if "weasyprint" in sys.modules or True:
		# we'll attempt import inside helper which will handle exceptions
		print("Trying WeasyPrint...")
		rendered = try_weasyprint_render(html, outpath, base_url=base_url, css_string=generate_css(condensed=args.condense))

	if not rendered:
		print("Trying pdfkit (wkhtmltopdf)...")
		rendered = try_pdfkit_render(html, outpath, base_url=base_url)

	if rendered:
		print(f"Saved PDF: {outpath}")
		return 0

	# fallback: write HTML to same directory and instruct user
	html_path = outpath.with_suffix(".html")
	html_path.write_text(html, encoding="utf-8")
	print("Could not render PDF automatically.")
	print(f"Wrote HTML to: {html_path}")
	print("Install WeasyPrint (and its system deps) or wkhtmltopdf and rerun to get PDF output.")
	return 5


if __name__ == "__main__":
	raise SystemExit(main())

