from enum import Enum
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict


class DocumentType(int, Enum):
    """The type of document used by PyMongoVoyageAI."""

    storage = 1
    image = 2
    text = 3


class Document(BaseModel):
    """A document object used by PyMongoVoyageAI."""

    type: DocumentType
    metadata: dict[str, Any] | None = None


class ImageDocument(Document):
    """A document object containing image data and associated properties."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: DocumentType = DocumentType.image
    image: Image.Image
    name: str | None = None
    source_url: str | None = None
    page_number: int | None = None


class StoredDocument(Document):
    """A document object containing stored object data and associated properties."""

    type: DocumentType = DocumentType.storage
    root_location: str
    object_name: str
    name: str | None = None
    source_url: str | None = None
    page_number: int | None = None


class TextDocument(Document):
    """A document object containing text data."""

    type: DocumentType = DocumentType.text
    text: str
