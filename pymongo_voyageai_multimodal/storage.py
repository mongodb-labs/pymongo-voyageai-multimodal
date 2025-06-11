import io

import boto3  # type:ignore[import-untyped]
import botocore  # type:ignore[import-untyped]


class ObjectStorage:
    """A class used to store binary data."""

    root_location: str
    """The default root location to use in the object store."""

    url_prefixes: list[str] | None
    """The url prefixes used by the object store, for reading data from a url."""

    def save_data(self, data: io.BytesIO, object_name: str) -> None:
        """Save data to the object store."""
        raise NotImplementedError

    def read_data(self, object_name: str) -> io.BytesIO:
        """Read data from the object store."""
        raise NotImplementedError

    def load_url(self, url: str) -> io.BytesIO:
        """Load data from a url."""
        raise NotImplementedError

    def delete_data(self, object_name: str) -> None:
        """Delete data from the object store."""
        raise NotImplementedError

    def close(self):
        """Close the object store."""
        pass


class S3Storage(ObjectStorage):
    """An object store using an S3 bucket."""

    url_prefixes = ["s3://"]

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

    def save_data(self, data: io.BytesIO, object_name: str) -> None:
        """Save data to the object store."""
        self.client.upload_fileobj(data, self.root_location, object_name)

    def read_data(self, object_name: str) -> io.BytesIO:
        """Read data using the object store."""
        buffer = io.BytesIO()
        self.client.download_fileobj(self.root_location, object_name, buffer)
        return buffer

    def load_url(self, url: str) -> io.BytesIO:
        """Load data from a url."""
        bucket, _, object_name = url.replace("s3://", "").partition("/")
        buffer = io.BytesIO()
        self.client.download_fileobj(bucket, object_name, buffer)
        return buffer

    def delete_data(self, object_name: str) -> None:
        """Delete data from the object store."""
        self.client.delete_object(Bucket=self.root_location, Key=object_name)

    def close(self) -> None:
        self.client.close()


class MemoryStorage(ObjectStorage):
    """An in-memory object store"""

    url_prefixes = ["file://"]

    def __init__(self) -> None:
        self.root_location = "foo"
        self.storage: dict[str, io.BytesIO] = dict()

    def save_data(self, data: io.BytesIO, object_name: str) -> None:
        """Save data to the object store."""
        self.storage[object_name] = data

    def read_data(self, object_name: str) -> io.BytesIO:
        """Read data using the object store."""
        return self.storage[object_name]

    def load_url(self, url: str) -> io.BytesIO:
        """Load data from a url."""
        with open(url.replace("file://", ""), "rb") as fid:
            return io.BytesIO(fid.read())

    def delete_data(self, object_name: str) -> None:
        """Delete data from the object store."""
        self.storage.pop(object_name, None)
