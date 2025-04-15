from enum import Enum
from pydantic import BaseModel, ConfigDict
from PIL import Image
from typing import Any

class DocumentType(int, Enum):
    storage = 1
    image = 2
    text = 3

class Document(BaseModel):
    type: DocumentType
    metadata: dict[str, Any] | None = None

class ImageDocument(Document):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: DocumentType = DocumentType.image
    image: Image.Image
    name: str | None = None
    source_url: str | None = None
    page_number: int | None = None

class StoredDocument(Document):
    type: DocumentType = DocumentType.storage
    root_location: str
    object_name: str
    source_url: str | None
    page_number: int | None

class TextDocument(Document):
    type: DocumentType= DocumentType.text
    text: str