import io
import urllib.request
from typing import Any

from PIL import Image

from .document import ImageDocument

try:
    import fitz  # type:ignore[import-untyped]
except ImportError:
    fitz = None


DEFAULT_MODEL_NAME = "voyage-multimodal-3"
TIMEOUT = 15
INTERVAL = 1


def pdf_url_to_images(
    url: str, start: int | None = None, end: int | None = None, zoom: float = 1.0
) -> list[Image.Image]:
    if fitz is None:
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
    print("out of loop")

    # Close the document
    pdf.close()

    return images


def url_to_images(
    url: str,
    metadata: dict[str, Any] | None = None,
    start: int = 0,
    end: int | None = None,
    image_column: str | None = None,
    **kwargs: Any,
) -> list[ImageDocument]:
    images = []
    i = url.rfind("/") + 1
    basename = url[i:]
    i = basename.rfind(".")
    name = basename[:i]
    if url.endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError:
            raise ValueError("pymongo-voyageai requires pandas to read parquet files") from None
        if image_column is None:
            raise ValueError("Must supply and image field to read a parquet file")
        column = pd.read_parquet(url, **kwargs)[image_column][start:end]
        for idx, item in enumerate(column.tolist()):
            image = Image.open(io.BytesIO(item["bytes"]))
            images.append(
                ImageDocument(
                    image=image,
                    name=name,
                    source_url=url,
                    page_number=idx + start,
                    metadata=metadata,
                )
            )
    elif url.endswith(".pdf"):
        for idx, img in enumerate(pdf_url_to_images(url, start=start, end=end, **kwargs)):
            images.append(
                ImageDocument(
                    image=img,
                    name=name,
                    source_url=url,
                    page_number=idx + start,
                    metadata=metadata,
                )
            )
    else:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
        image = Image.open(io.BytesIO(image_data))
        if "transparency" in image.info and image.mode != "RGBA":
            image = image.convert("RGBA")
        images.append(ImageDocument(image=image, name=name, source_url=url, metadata=metadata))
    return images
