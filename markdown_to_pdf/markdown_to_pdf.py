import markdown
from weasyprint import HTML

def convert_markdown_to_pdf(markdown_text, output_filename="output.pdf"):
    """
    Converts Markdown text to a PDF file using WeasyPrint and Python-Markdown.

    Args:
        markdown_text (str): The Markdown text to convert.
        output_filename (str, optional): The name of the output PDF file. Defaults to "output.pdf".
    """

    try:
        # Convert Markdown to HTML
        html_text = markdown.markdown(markdown_text)

        # Wrap the HTML in a basic HTML document structure
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Markdown to PDF</title>
</head>
<body>
    {html_text}
</body>
</html>"""

        # Create a WeasyPrint HTML object
        html_obj = HTML(string=html)

        # Write the PDF to a file
        html_obj.write_pdf(output_filename)

        print(f"Successfully converted Markdown to PDF: {output_filename}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")


if __name__ == "__main__":
    # Example Usage (replace with your Markdown content)
    markdown_content = """
# My Awesome Document

This is a simple example of converting Markdown to PDF.

## A Subheading

Here's some text with **bold** and *italic* formatting.  You can also use lists:

- Item 1
- Item 2
- Item 3

And links like [example.com](https://www.example.com).
"""

    convert_markdown_to_pdf(markdown_content, "my_document.pdf")
