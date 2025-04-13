from __future__ import annotations
from typing import Any, Sequence
from enum import Enum
from voyageai import Client
import urllib.request
from PIL import Image
from io import BytesIO
from pymongo import MongoClient, ReplaceOne
import logging
import fitz
from pydantic import BaseModel, ConfigDict
import boto3
from bson import ObjectId
import botocore.client
from langchain_mongodb.vectorstores import DEFAULT_INSERT_BATCH_SIZE
from langchain_mongodb.index import create_vector_search_index
from langchain_mongodb.pipelines import vector_search_stage
from langchain_mongodb.utils import make_serializable
import io

DEFAULT_MODEL_NAME = "voyage-multimodal-3"
logger = logging.getLogger(__file__)


def pdf_url_to_images(url: str, zoom: float = 1.0) -> list[Image.Image]:

    # Ensure that the URL is valid
    if not url.startswith("http") and url.endswith(".pdf"):
        raise ValueError("Invalid URL")

    # Read the PDF from the specified URL
    with urllib.request.urlopen(url) as response:
        pdf_data = response.read()
    pdf_stream = BytesIO(pdf_data)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")

    images = []

    # Loop through each page, render as pixmap, and convert to PIL Image
    mat = fitz.Matrix(zoom, zoom)
    for n in range(pdf.page_count):
        pix = pdf[n].get_pixmap(matrix=mat)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    # Close the document
    pdf.close()

    return images


class DocumentType(str, Enum):
    s3_object = 1
    image = 2
    url = 3
    text = 4

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

class URLDocument(Document):
    type: DocumentType = DocumentType.url
    url: str

class S3Document(Document):
    type: DocumentType = DocumentType.s3_object
    bucket_name: str
    object_name: str
    source_url: str | None
    page_number: int | None

class TextDocument(Document):
    type: DocumentType= DocumentType.text
    text: str


class PyMongoVoyageAI:

    def __init__(
        self,
        s3_bucket_name: str,
        collection_name: str,
        database_name: str,
        mongo_client: MongoClient | None = None,
        mongo_connection_string: str | None = None,
        voyageai_client: Client | None = None,
        voyageai_api_key: str| None = None,
        voyagai_model_name: str = DEFAULT_MODEL_NAME,
        s3_client: botocore.client.BaseClient | None = None,
        aws_region_name: str | None = None,
        index_name: str = "vector_index",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        dimensions: int = 1024,
        auto_create_index: bool = True,
        auto_index_timeout: int = 15,
        **kwargs: Any,
    ):
        """
        Args:
            collection: MongoDB collection to add the texts to
            text_key: MongoDB field that will contain the text for each document
            index_name: Existing Atlas Vector Search Index
            embedding_key: Field that will contain the embedding for each document
            vector_index_name: Name of the Atlas Vector Search index
            relevance_score_fn: The similarity score used for the index
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'
            auto_create_index: Whether to automatically create the vector search index if needed.
            auto_index_timeout: Timeout in seconds to wait for an auto-created index
               to be ready.
        """
        self._dimensions = dimensions  # the size of the VoyageAI model.
        self._vo = voyageai_client or Client(api_key=voyageai_api_key)
        # TODO: driver=DriverInfo(name="Langchain", version=version("langchain-mongodb")),
        self._mo = mongo_client or MongoClient(mongo_connection_string)
        self._index_name = index_name
        self._embedding_key = embedding_key
        self._relevance_score_fn = relevance_score_fn
        self._s3_client = s3_client or boto3.client('s3', region_name=aws_region_name)
        self._s3_bucket_name = s3_bucket_name
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

    def image_to_s3(self, document: ImageDocument | Image.Image) -> S3Document:
        if isinstance(document, Image.Image):
            document = ImageDocument(image=document)
        object_name = document.name or str(ObjectId())
        fd = io.BytesIO()
        document.image.save(fd, "png")
        fd.seek(0)
        self._s3_client.upload_fileobj(fd, self._s3_bucket_name, object_name)
        return S3Document(bucket_name=self._s3_bucket_name, object_name=object_name, page_number=document.page_number, source_url=document.source_url, metadata=document.metadata)

    def s3_to_image(self, document: S3Document | str) -> ImageDocument:
        if isinstance(document, str):
            document = S3Document(bucket_name=self._s3_bucket_name, object_name=document)
        buffer = io.BytesIO()
        self._s3_client.download_fileobj(document.bucket_name, document.object_name, buffer)
        image = Image.open(buffer)
        return ImageDocument(image=image, source_url=document.source_url, page_number=document.page_number, metadata=document.metadata, name=document.object_name)

    def url_to_images(self, document: URLDocument | str) -> list[ImageDocument]:
        if isinstance(document, str):
            document = URLDocument(url=document)
        url = document.url
        images = []
        i = url.rfind('/') + 1 
        basename = url[i:]
        i = basename.rfind('.')
        name = basename[:i]
        if url.endswith('.pdf'):
            for idx, img in pdf_url_to_images(url):
                images.append(ImageDocument(image=img, name=name, source_url=url, page_number=idx, metadata=document.metadata))
        else:
            with urllib.request.urlopen(url) as response:
                image_data = response.read()
            image = Image.open(BytesIO(image_data))
            images.append(ImageDocument(image=image, name=name, source_url=url, metadata=document.metadata))
        return images

    def add_documents(
        self,
        inputs: list[list[str | Image.Image | Document]],
        ids: list[str] | None = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Add documents to the vectorstore.

        Args:
            inputs: List of inputs to add to the vectorstore, which are each a list of documents.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids in add_texts.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.

        Returns:
            A list of the embedded documents, including embeddings, s3 locations if relevant, and any other metadata.
        """
        # Process the input documents, creating the metadata to write to the database, as well as the inputs to the model.
        # Save images to s3 along the way as appropriate.
        processed_inputs = []
        model_inputs = []
        for inp in inputs:
            processed_inner = []
            model_inner = []
            for doc in inp:
                if isinstance(doc, str):
                    processed_inner.append(TextDocument(text=doc))
                    model_inner.append(doc)
                elif isinstance(doc, Image.Image):
                    model_inner.append(doc)
                    doc = ImageDocument(image=doc)
                    processed_inner.append(self.image_to_s3(doc))
                elif isinstance(doc, URLDocument):
                    for doc in self.url_to_images(doc):
                        processed_inner.append(self.image_to_s3(doc))
                        model_inner.append(doc.image)
                elif isinstance(doc, ImageDocument):
                    processed_inner.append(self.image_to_s3(doc))
                    model_inner.append(doc.image)
                elif isinstance(doc, S3Document):
                    processed_inner.append(doc)
                    model_inner.append(self.s3_to_image(doc).image)
                elif isinstance(doc, TextDocument):
                    processed_inner.append(doc)
                    model_inner.append(doc.text)
                else:
                    raise ValueError(f"Cannot process item of type {type(doc)}")
            processed_inputs.append(processed_inner)
            model_inputs.append(model_inner)

        # Create the embeddings for each set of processed model inputs.
        embeddings = self._vo.multimodal_embed(
                inputs=model_inputs,
                model=self._vo_model_name,
                input_type="document"
            ).embeddings
        
        # Write the embeddings and serialized inputs to the database in batches.
        # Use ReplaceOne to enable overwriting documents by _id.
        if ids:
            ids = [ObjectId(i) for i in ids]
        else:
            ids = ids or [ObjectId() for _ in range(len(inputs))]
        batch = []
        output_docs = []
        for idx, inp in enumerate(processed_inputs):
            output_doc = {
                self._embedding_key: embeddings[idx],
                "inputs": inp,
                "_id": ids[idx]
            }
            output_docs.append(output_doc)
            doc = {
                self._embedding_key: embeddings[idx],
                "inputs": [i.model_dump() for i in inp],
                "_id": ids[idx]
            }
            batch.append(doc)
            if len(batch) == batch_size:
                operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in batch]
                self._coll.bulk_write(operations)
                batch = []
        if batch:
            operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in batch]
            self._coll.bulk_write(operations)
        return output_docs

    def delete(self, ids: list[str | ObjectId], delete_s3_objects: bool = True, **kwargs: Any) -> bool:
        """Delete documents by ids.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments passed to Collection.delete_many()

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        filter = {}
        oids = [ObjectId(str(i)) for i in ids]
        filter = {"_id": {"$in": oids}}
        if delete_s3_objects:
            for obj in self._coll.find(filter):
                self._expand_doc(obj, False)
                for inp in obj['inputs']:
                    if isinstance(inp, S3Document):
                        self._s3_client.delete_object(Bucket=inp.bucket_name, Key=inp.object_name)
        return self._coll.delete_many(filter=filter, **kwargs).acknowledged

    def _expand_doc(self, obj: dict[str, Any], extract_images: bool = True) -> dict[str, Any]:
        for idx, inp in enumerate(list(obj['inputs'])):
            if inp['type'] == DocumentType.s3_object:
                doc = S3Document.model_validate(inp)
                if extract_images:
                    doc = self.s3_to_image(doc)
                obj['inputs'][idx] = doc
            elif inp['type'] == DocumentType.text:
                obj['inputs'][idx] = TextDocument.model_validate(inp)

    def get_by_ids(self, ids: Sequence[str | ObjectId], extract_images: bool = True) -> list[Document]:
        docs = []
        oids = [ObjectId(i) for i in ids]
        for doc in self._coll.aggregate([{"$match": {"_id": {"$in": oids}}}]):
            self._expand_doc(doc, extract_images)
            docs.append(doc)
        return docs

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
        """Return MongoDB documents most similar to the given query.

        Atlas Vector Search eliminates the need to run a separate
        search system alongside your database.

         Args:
            query: Input text of semantic query
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: List of MQL match expressions comparing an indexed field
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                to filter/process results after $vectorSearch.
            oversampling_factor: Multiple of k used when generating number of candidates
                at each step in the HNSW Vector Search,
            include_scores: If True, the query score of each result
                will be included in metadata.
            include_embeddings: If True, the embedding vector of each result
                will be included in metadata.
            extract_images: If True, image data will be retrieved from
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores.
        """
        # $vectorSearch query followed by a subsequent lookup to the data stored in s3 along with the related metadata provided from Atlas.
        # Each element in the output of the query should contain the payload in Atlas, the singulated s3 image that was looked up, and (if relevant) the original pdf information.
        query_vector = self._vo.multimodal_embed(
                inputs=[[query]],
                model=self._vo_model_name,
                input_type="document"
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
        cursor = self._coll.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []

        # Format and extract if necessary.
        for res in cursor:
            make_serializable(res)
            self._expand_doc(res, extract_images)
            docs.append(res)
        return docs


def main():
    import os
    print("Hello from pymongo-voyageai!")

    import pandas as pd
    df = pd.read_parquet("hf://datasets/princeton-nlp/CharXiv/val.parquet")

    datas = df["image"].head(3).tolist()
    figures = [Image.open(BytesIO(d["bytes"])) for d in datas]
    documents = [[figures[n]] for n in range(len(figures))]
    conn_str = "mongodb://127.0.0.1:27017?directConnection=true"
    vo = PyMongoVoyageAI(voyageai_api_key=os.environ['VOYAGE_API_KEY'], mongo_connection_string=conn_str,
                         s3_bucket_name="pymongo-voyageai", collection_name="test", database_name="tests")
    resp = vo.add_documents(documents)
    import time
    time.sleep(3)
    print(len(resp))
    data = vo.similarity_search("3D loss landscapes for different training strategies", extract_images=True)
    print(data)
    print(len(vo.get_by_ids([d['_id'] for d in resp])))
    vo.delete([d['_id'] for d in resp])

if __name__ == "__main__":
    main()
