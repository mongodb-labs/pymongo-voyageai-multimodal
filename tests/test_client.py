import numpy as np
import os
import pytest
from pymongo_voyageai import PyMongoVoyageAI


@pytest.fixure
def client() -> PyMongoVoyageAI:
    conn_str = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true")
    client = PyMongoVoyageAI(voyageai_api_key=os.environ['VOYAGE_API_KEY'], mongo_connection_string=conn_str,
                           s3_bucket_name="pymongo-voyageai", collection_name="test", database_name="tests")
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
    # The best match should be the third input image.
    assert data[0]['inputs'][0].image.tobytes() == documents[2].image.tobytes()
    client.delete_by_ids([d['_id'] for d in resp])


def test_text_and_images(client: PyMongoVoyageAI):
    text = "Voyage AI makes best-in-class embedding models and rerankers."
    images = client.url_to_images("https://www.voyageai.com/header-bg.png")
    image = images[0].image
    documents = [
        [text],         # 0. single text
        [image],        # 1. single image
        [text, image],  # 2. text + image
        [image, text],  # 3. image + text
    ]
    resp = client.add_documents(documents)
    client.wait_for_indexing()
    # The interleaved inputs should have different but similar embeddings.
    embeddings = [d['embedding'] for d in resp]
    assert embeddings[2] != embeddings[3]
    assert np.dot(embeddings[2], embeddings[3]) > 0.95
    client.delete_by_ids([d['_id'] for d in resp])


def test_pdf_pages(client: PyMongoVoyageAI):
    query = "The consequences of a dictator's peace"
    url = "https://www.fdrlibrary.org/documents/356632/390886/readingcopy.pdf"
    images = client.url_to_images(url)
    resp = client.add_documents(images)
    client.wait_for_indexing()
    data = client.similarity_search(query, extract_images=True)
    # We expect page 5 to be the best match.
    assert data[0]['inputs'][0].page_number == 5
    assert len(client.get_by_ids([d['_id'] for d in resp])) == len(resp)
    client.delete_by_ids([d['_id'] for d in resp])
