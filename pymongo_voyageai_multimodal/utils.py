import io
import ssl
import urllib.request
from typing import Any

import certifi
from PIL import Image

from .document import ImageDocument
from .storage import ObjectStorage, S3Storage

try:
    import fitz  # type:ignore[import-untyped]
except ImportError:
    fitz = None


DEFAULT_MODEL_NAME = "voyage-multimodal-3"
TIMEOUT = 60
INTERVAL = 1


def pdf_data_to_images(
    pdf_stream: io.BytesIO, start: int | None = None, end: int | None = None, zoom: float = 1.0
) -> list[Image.Image]:
    """Extract images from a pdf byte stream.

    Args:
        pdf_stream: The BytesIO object to load the images from.
        start: The start frame to use for the images.
        end: The end frame to use for the images.
        zoom: The zoom factor to apply to the images.

    Returns:
        A list of image objects.
    """
    if fitz is None:
        raise ValueError("pymongo-voyageai-multimodal requires PyMuPDF to read pdf files") from None

    # Read the PDF from the specified URL
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


def url_to_images(
    url: str,
    storage: ObjectStorage | None = None,
    metadata: dict[str, Any] | None = None,
    start: int = 0,
    end: int | None = None,
    image_column: str | None = None,
    **kwargs: Any,
) -> list[ImageDocument]:
    """Extract images from a url.

    Args:
        url: The url to load the images from.
        storage: The storage object which can be used to load data from custom urls.
        metadata: A set of metadata to associate with the images.
        start: The start frame to use for the images.
        end: The end frame to use for the images.
        image_column: The name of the column used to store the image data, for parquet files.

    Returns:
        A list of image document objects.
    """
    images = []
    i = url.rfind("/") + 1
    basename = url[i:]
    i = basename.rfind(".")
    name = basename[:i]

    source = None
    # Prefer to use our storage object to read the file data.
    if storage and storage.url_prefixes:
        for pattern in storage.url_prefixes:
            if url.startswith(pattern):
                source = storage.load_url(url)
                break
    # For parquet files that are not loaded by the storage object, let pandas handle the download.
    if source is None and url.endswith(".parquet"):
        source = url
    # For s3 files that are not loaded by the storage object, create a temp S3Storage object.
    if source is None and url.startswith("s3://"):
        storage = S3Storage("")
        source = storage.load_url(url)
        storage.close()
    # For all other files, use the native download.
    if source is None:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ssl_context) as response:
            source = io.BytesIO(response.read())

    if url.endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError:
            raise ValueError(
                "pymongo-voyageai-multimodal requires pandas to read parquet files"
            ) from None
        if image_column is None:
            raise ValueError("Must supply and image field to read a parquet file")
        column = pd.read_parquet(source, **kwargs)[image_column][start:end]
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
        for idx, img in enumerate(pdf_data_to_images(source, start=start, end=end, **kwargs)):
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
        image = Image.open(source)
        if "transparency" in image.info and image.mode != "RGBA":
            image = image.convert("RGBA")
        images.append(ImageDocument(image=image, name=name, source_url=url, metadata=metadata))
    return images
