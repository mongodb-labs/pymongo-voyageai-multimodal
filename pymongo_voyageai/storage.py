import io

import boto3  # type:ignore[import-untyped]
import botocore  # type:ignore[import-untyped]
from bson import ObjectId
from PIL import Image

from .document import ImageDocument, StoredDocument


class ObjectStorage:
    """A class used store image documents."""

    root_location: str
    """The root location to use in the object store."""

    def save_image(self, image: ImageDocument) -> StoredDocument:
        """Save an image document to the object store."""
        raise NotImplementedError

    def load_image(self, document: StoredDocument) -> ImageDocument:
        """Load an image document from the object store."""
        raise NotImplementedError

    def delete_image(self, document: StoredDocument) -> None:
        """Remove an image document from the object store."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the object store."""
        raise NotImplementedError


class S3Storage(ObjectStorage):
    def __init__(
        self,
        bucket_name: str,
        client: botocore.client.BaseClient | None = None,
        region_name: str | None = None,
    ):
        """Create an S3 object store.

        Args:
            bucket_name: The s3 bucket name.
            client: An instantiated boto3 s3 client.
            region_name: The aws region name to use when creating a boto3 s3 client.
        """
        self.client = client or boto3.client("s3", region_name=region_name)
        self.root_location = bucket_name

    def save_image(self, image: ImageDocument) -> StoredDocument:
        object_name = str(ObjectId())
        fd = io.BytesIO()
        image.image.save(fd, "png")
        fd.seek(0)
        self.client.upload_fileobj(fd, self.root_location, object_name)
        return StoredDocument(
            root_location=self.root_location,
            object_name=object_name,
            page_number=image.page_number,
            source_url=image.source_url,
            name=image.name,
            metadata=image.metadata,
        )

    def load_image(self, document: StoredDocument) -> ImageDocument:
        buffer = io.BytesIO()
        self.client.download_fileobj(document.root_location, document.object_name, buffer)
        image = Image.open(buffer)
        return ImageDocument(
            image=image,
            source_url=document.source_url,
            page_number=document.page_number,
            metadata=document.metadata,
            name=document.name,
        )

    def delete_image(self, document: StoredDocument) -> None:
        self.client.delete_object(Bucket=document.root_location, Key=document.object_name)

    def close(self) -> None:
        self.client.close()
