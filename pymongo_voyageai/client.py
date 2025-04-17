from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from time import monotonic, sleep
from typing import Any

from bson import ObjectId
from langchain_mongodb.index import create_vector_search_index
from langchain_mongodb.pipelines import vector_search_stage
from langchain_mongodb.utils import make_serializable
from langchain_mongodb.vectorstores import DEFAULT_INSERT_BATCH_SIZE
from PIL import Image
from pymongo import MongoClient, ReplaceOne
from voyageai.client import Client

from .document import Document, DocumentType, ImageDocument, StoredDocument, TextDocument
from .storage import ObjectStorage, S3Storage
from .utils import DEFAULT_MODEL_NAME, INTERVAL, TIMEOUT, url_to_images

logger = logging.getLogger(__file__)


class PyMongoVoyageAI:
    def __init__(
        self,
        collection_name: str,
        database_name: str,
        s3_bucket_name: str | None = None,
        mongo_client: MongoClient[dict[str, Any]] | None = None,
        mongo_connection_string: str | None = None,
        voyageai_client: Client | None = None,
        voyageai_api_key: str | None = None,
        voyagai_model_name: str = DEFAULT_MODEL_NAME,
        storage_object: ObjectStorage | None = None,
        index_name: str = "vector_index",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        dimensions: int = 1024,
        auto_create_index: bool = True,
        auto_index_timeout: int = TIMEOUT,
        **kwargs: Any,
    ):
        """
        Args:
            collection_name: The name of the MongoDB collection to add the documents to.
            database_name: The name of the MongoDB database to use.
            s3_bucket_name: The name of the s3 bucket to use for storage.
            mongo_client: An instantiated MongoClient to use.
            mongo_connection_string: A MongoDB connection string that is used to create a
                MongoClient.  It must be provided if `mongo_client` is not provided.
            voyageai_client: An instantiated VoyageAI client to use.
            voyageai_api_key: An api key to use when creating a a VoyageAI Client object.
                It must be provided if `voyageai_client` is not provided.
            voyagai_model_name: The model name to use for VoyageAI embededdings.
            storage_object: The ObjectStorage object to use.  It can be used to provide alternate an
                alternate storage backend or an instantiated `S3Storage` object.
            index_name: The Atlas vector search index name to use for the collection.
            embedding_key: Field that will contain the embedding for each document.
            relevance_score_fn: The similarity score used for the index.
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'.
            dimensions: The dimensionality of the VoyageAI model.
            auto_create_index: Whether to automatically create the vector search index if needed.
            auto_index_timeout: Timeout in seconds to wait for an auto-created index.
               to be ready.
            kwargs: Additionally keyword args accepted for future use.
        """
        self._dimensions = dimensions  # the size of the VoyageAI model.
        self._vo = voyageai_client or Client(api_key=voyageai_api_key)
        # TODO: driver=DriverInfo(name="Langchain", version=version("langchain-mongodb")),
        self._mo = mongo_client or MongoClient(mongo_connection_string)
        self._index_name = index_name
        self._embedding_key = embedding_key
        self._relevance_score_fn = relevance_score_fn
        if storage_object:
            self._storage = storage_object
        elif s3_bucket_name:
            self._storage = S3Storage(s3_bucket_name)
        else:
            raise ValueError("Must provide an s3 bucket name or a storage object")
        self._vo_model_name = voyagai_model_name
        self._coll = coll = self._mo[database_name][collection_name]
        if auto_create_index and not any(
            [ix["name"] == self._index_name for ix in coll.list_search_indexes()]
        ):
            create_vector_search_index(
                collection=coll,
                index_name=self._index_name,
                dimensions=self._dimensions,
                path=self._embedding_key,
                similarity=self._relevance_score_fn,
                wait_until_completes=auto_index_timeout,
            )

    def image_to_storage(self, document: ImageDocument | Image.Image) -> StoredDocument:
        """Convert an image to a stored document.

        Args:
            document: The input document or image object.

        Returns:
            The stored document object.
        """
        if isinstance(document, Image.Image):
            document = ImageDocument(image=document)
        return self._storage.save_image(document)

    def storage_to_image(self, document: StoredDocument | str) -> ImageDocument:
        """Convert a stored document to an image document.

        Args:
            document: The input document or object name.

        Returns:
            The image document object.
        """
        if isinstance(document, str):
            document = StoredDocument(
                root_location=self._storage.root_location, object_name=document
            )
        return self._storage.load_image(document=document)

    def url_to_images(
        self,
        url: str,
        metadata: dict[str, Any] | None = None,
        start: int = 0,
        end: int | None = None,
        image_column: str | None = None,
        **kwargs: Any,
    ) -> list[ImageDocument]:
        """Extract images from a url.

        Args:
            url: The url to load the images from.
            metadata: A set of metadata to associate with the images.
            start: The start frame to use for the images.
            end: The end frame to use for the images.
            image_column: The name of the column used to store the image data, for parquet files.

        Returns:
            A list of image document objects.
        """
        return url_to_images(
            url, metadata=metadata, start=start, end=end, image_column=image_column, **kwargs
        )

    def add_documents(
        self,
        inputs: Sequence[str | Image.Image | Document | Sequence[str | Image.Image | Document]],
        ids: list[str] | None = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Add multimodal documents to the vectorstore.

        Args:
            inputs: List of inputs to add to the vectorstore, which are each a list of documents.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids in add_texts.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.
            kwargs: Additional keyword args for future expansion.

        Returns:
            A list documents with their associated input documents.
        """
        # Process the input documents, creating the metadata to write to the database,
        # as well as the inputs to the model.
        # Save images to storage along the way as appropriate.
        processed_inputs = []
        model_inputs = []
        for inp in inputs:
            processed_inner: list[Document] = []
            model_inner: list[str | Image.Image] = []
            if isinstance(inp, (str, Image.Image, Document)):  # noqa:UP038
                inp = [inp]
            for doc in inp:
                if isinstance(doc, str):
                    doc = TextDocument(text=doc)
                elif isinstance(doc, Image.Image):
                    doc = ImageDocument(image=doc)
                if isinstance(doc, ImageDocument):
                    processed_inner.append(self.image_to_storage(doc))
                    model_inner.append(doc.image)
                elif isinstance(doc, StoredDocument):
                    processed_inner.append(doc)
                    model_inner.append(self.storage_to_image(doc).image)
                elif isinstance(doc, TextDocument):
                    processed_inner.append(doc)
                    model_inner.append(doc.text)
                else:
                    raise ValueError(f"Cannot process item of type {type(doc)}")
            if processed_inner:
                processed_inputs.append(processed_inner)
                model_inputs.append(model_inner)

        # Create the embeddings for each set of processed model inputs.
        embeddings = self._vo.multimodal_embed(
            inputs=model_inputs, model=self._vo_model_name, input_type="document"
        ).embeddings

        # Write the embeddings and serialized inputs to the database in batches.
        # Use ReplaceOne to enable overwriting documents by _id.
        if ids:
            obj_ids = [ObjectId(i) for i in ids]
        else:
            obj_ids = [ObjectId() for _ in range(len(processed_inputs))]
        batch = []
        output_docs = []
        for idx, proc_inp in enumerate(processed_inputs):
            output_doc = {
                self._embedding_key: embeddings[idx],
                "inputs": proc_inp,
                "_id": obj_ids[idx],
            }
            output_docs.append(output_doc)
            pymongo_doc = {
                self._embedding_key: embeddings[idx],
                "inputs": [i.model_dump() for i in proc_inp],
                "_id": obj_ids[idx],
            }
            batch.append(pymongo_doc)
            if len(batch) == batch_size:
                operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in batch]
                self._coll.bulk_write(operations)
                batch = []
        if batch:
            operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in batch]
            self._coll.bulk_write(operations)
        return output_docs

    def delete_by_ids(
        self, ids: list[str | ObjectId], delete_stored_objects: bool = True, **kwargs: Any
    ) -> bool:
        """Delete documents by ids.

        Args:
            ids: List of ids to delete.
            delete_stored_objects: Whether to delete the associated stored objects.
            **kwargs: Other keyword arguments passed to delete_many().

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        oids = [ObjectId(str(i)) for i in ids]
        return self.delete_many(
            {"_id": {"$in": oids}}, delete_stored_objects=delete_stored_objects, **kwargs
        )

    def delete_many(
        self, filter: Mapping[str, Any], delete_stored_objects: bool = True, **kwargs: Any
    ) -> bool:
        """Delete documents using a filter.

        Args:
            ids: List of ids to delete.
            delete_stored_objects: Whether to delete the associated stored objects.
            **kwargs: Other keyword arguments passed to the collection's `delete_many` method.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        if delete_stored_objects:
            for obj in self._coll.find(filter):
                self._expand_doc(obj, False)
                for inp in obj["inputs"]:
                    if isinstance(inp, StoredDocument):
                        self._storage.delete_image(inp)
        return self._coll.delete_many(filter=filter, **kwargs).acknowledged

    def close(self) -> None:
        """Close the client, cleaning up resources."""
        self._coll.database.client.close()
        self._storage.close()

    def get_by_ids(
        self, ids: Sequence[str | ObjectId], extract_images: bool = True
    ) -> list[dict[str, Any]]:
        """Get a list of documents by id.

        Args:
            ids: List of ids to search for.
            extract_images: Whether to extract the stored documents into image documents.

        Returns:
            A list of matching documents, where the `inputs` is a list of stored documents
            or image documents.
        """
        docs = []
        oids = [ObjectId(i) for i in ids]
        for doc in self._coll.aggregate([{"$match": {"_id": {"$in": oids}}}]):
            self._expand_doc(doc, extract_images)
            docs.append(doc)
        return docs

    def wait_for_indexing(self, timeout: int = TIMEOUT, interval: int = INTERVAL) -> None:
        """Wait for the search index to update to account for newly added embeddings."""
        n_docs = self._coll.count_documents({})
        start = monotonic()
        while monotonic() - start <= timeout:
            if len(self.similarity_search("sandwich", k=n_docs, oversampling_factor=1)) == n_docs:
                return
            else:
                sleep(interval)

        raise TimeoutError(f"Failed to embed, insert, and index texts in {timeout}s.")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: dict[str, Any] | None = None,
        post_filter_pipeline: list[dict[str, Any]] | None = None,
        oversampling_factor: int = 10,
        include_scores: bool = False,
        include_embeddings: bool = False,
        extract_images: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:  # noqa: E501
        """Return documents most similar to the given query.

        Args:
            query: Input text of semantic query.
            k: The number of documents to return. Defaults to 4.
            pre_filter: List of MQL match expressions comparing an indexed field.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                to filter/process results after $vectorSearch.
            oversampling_factor: Multiple of k used when generating number of candidates
                at each step in the HNSW Vector Search.
            include_scores: If True, the query score of each result
                will be included in metadata.
            include_embeddings: If True, the embedding vector of each result
                will be included in metadata.
            extract_images: If True, the stored documents will be converted image documents.
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores, where the `inputs`
            is a list of stored documents or image documents.
        """
        query_vector = self._vo.multimodal_embed(
            inputs=[[query]],
            model=self._vo_model_name,
            input_type="document",  # type:ignore[arg-type]
        ).embeddings[0]

        # Atlas Vector Search, potentially with filter
        pipeline = [
            vector_search_stage(
                query_vector,
                self._embedding_key,
                self._index_name,
                k,
                pre_filter,
                oversampling_factor,
                **kwargs,
            )
        ]
        if include_scores:
            pipeline.append({"$set": {"score": {"$meta": "vectorSearchScore"}}})

        # Remove embeddings unless requested.
        if not include_embeddings:
            pipeline.append({"$project": {self._embedding_key: 0}})

        # Post-processing.
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        # Execution.
        cursor = self._coll.aggregate(pipeline)
        docs = []

        # Format and extract if necessary.
        for res in cursor:
            make_serializable(res)
            self._expand_doc(res, extract_images)
            docs.append(res)
        return docs

    def _expand_doc(self, obj: dict[str, Any], extract_images: bool = True) -> dict[str, Any]:
        for idx, inp in enumerate(list(obj["inputs"])):
            if inp["type"] == DocumentType.storage:
                doc = StoredDocument.model_validate(inp)
                if extract_images:
                    doc = self.storage_to_image(doc)  # type:ignore[assignment]
                obj["inputs"][idx] = doc
            elif inp["type"] == DocumentType.text:
                obj["inputs"][idx] = TextDocument.model_validate(inp)
        return obj
