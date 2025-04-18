import os
from collections.abc import Generator
from typing import Any

import pytest

from pymongo_voyageai import MemoryStorage, PyMongoVoyageAI

# mypy: disable_error_code="no-untyped-def"


class MockVoyageAIResponse:
    def __init__(self, n):
        self.n = n

    @property
    def embeddings(self) -> list[float]:
        return [[0.1] * 1024 for i in range(self.n)]


class MockVoyageAI:
    def multimodal_embed(
        self,
        inputs: list[Any],
        model: str,
        input_type: str,
    ):
        return MockVoyageAIResponse(len(inputs))


@pytest.fixture
def client() -> Generator[PyMongoVoyageAI, None, None]:
    conn_str = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true")
    client = PyMongoVoyageAI(
        voyageai_client=MockVoyageAI(),
        storage_object=MemoryStorage(),
        mongo_connection_string=conn_str,
        collection_name="test",
        database_name="pymongo_voyageai_test_db",
    )
    client.delete_many({})
    yield client
    client.close()


def test_image_set(client: PyMongoVoyageAI):
    url = "hf://datasets/princeton-nlp/CharXiv/val.parquet"
    documents = client.url_to_images(url, image_column="image", end=3)
    resp = client.add_documents(documents)
    client.wait_for_indexing()
    query = "3D loss landscapes for different training strategies"
    data = client.similarity_search(query, extract_images=True)
    assert len(data[0]["inputs"][0].image.tobytes()) > 0
    client.delete_by_ids([d["_id"] for d in resp])


def test_text_and_images(client: PyMongoVoyageAI):
    text = "Voyage AI makes best-in-class embedding models and rerankers."
    images = client.url_to_images("https://www.voyageai.com/header-bg.png")
    image = images[0].image
    resp = client.add_documents(
        [
            [text],  # 0. single text
            [image],  # 1. single image
            [text, image],  # 2. text + image
            [image, text],  # 3. image + text
        ]
    )
    client.wait_for_indexing()
    assert len(client.get_by_ids([d["_id"] for d in resp])) == len(resp)
    client.delete_by_ids([d["_id"] for d in resp])


def test_pdf_pages(client: PyMongoVoyageAI):
    query = "The consequences of a dictator's peace"
    url = "https://www.fdrlibrary.org/documents/356632/390886/readingcopy.pdf"
    images = client.url_to_images(url)
    resp = client.add_documents(images)
    client.wait_for_indexing()
    data = client.similarity_search(query, extract_images=True)
    assert len(data[0]["inputs"][0].image.tobytes()) > 0
    assert len(client.get_by_ids([d["_id"] for d in resp])) == len(resp)
    client.delete_by_ids([d["_id"] for d in resp])


@pytest.mark.asyncio
async def test_image_set_async(client: PyMongoVoyageAI):
    url = "hf://datasets/princeton-nlp/CharXiv/val.parquet"
    documents = await client.aurl_to_images(url, image_column="image", end=3)
    resp = await client.aadd_documents(documents)
    await client.await_for_indexing()
    query = "3D loss landscapes for different training strategies"
    data = await client.asimilarity_search(query, extract_images=True)
    assert len(data[0]["inputs"][0].image.tobytes()) > 0
    ids = await client.aget_by_ids([d["_id"] for d in resp])
    assert len(ids) == len(resp)
    await client.adelete_by_ids([d["_id"] for d in resp])
    await client.adelete_many({})
    await client.aclose()
