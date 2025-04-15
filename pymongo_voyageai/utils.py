import io
import urllib.request

from PIL import Image

DEFAULT_MODEL_NAME = "voyage-multimodal-3"
TIMEOUT = 15
INTERVAL = 1


def pdf_url_to_images(url: str, start: int | None =None, end: int|None=None, zoom: float = 1.0) -> list[Image.Image]:
    try:
        import fitz  # type:ignore[import-untyped]
    except ImportError:
        raise ValueError("pymongo-voyageai requires PyMuPDF to read pdf files") from None
    # Ensure that the URL is valid
    if not url.startswith("http") and url.endswith(".pdf"):
        raise ValueError("Invalid URL")

    # Read the PDF from the specified URL
    with urllib.request.urlopen(url) as response:
        pdf_data = response.read()
    pdf_stream = io.BytesIO(pdf_data)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")

    images = []

    # Loop through each page, render as pixmap, and convert to PIL Image
    mat = fitz.Matrix(zoom, zoom)
    start = start or 0
    end = end or pdf.page_count - 1
    for n in range(pdf.page_count):
        if n < start or n >= end:
            continue
        pix = pdf[n].get_pixmap(matrix=mat)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    # Close the document
    pdf.close()

    return images
