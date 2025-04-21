Examples
========

Querying Against PDF Pages
--------------------------

.. code-block:: python

    import os
    from pymongo_voyageai import PyMongoVoyageAI

    client = PyMongoVoyageAI(
        voyageai_api_key=os.environ["VOYAGEAI_API_KEY"],
        s3_bucket_name=os.environ["S3_BUCKET_NAME"],
        mongo_connection_string=os.environ["MONGODB_URI"],
        collection_name="test",
        database_name="test_db",
    )

    query = "The consequences of a dictator's peace"
    url = "https://www.fdrlibrary.org/documents/356632/390886/readingcopy.pdf"
    images = client.url_to_images(url)
    resp = client.add_documents(images)
    client.wait_for_indexing()
    data = client.similarity_search(query, extract_images=False)

    # We expect page 5 to be the best match.
    assert data[0]["inputs"][0].page_number == 5
    assert len(client.get_by_ids([d["_id"] for d in resp])) == len(resp)
    client.delete_by_ids([d["_id"] for d in resp])
    client.close()


Querying Against Parquet Data
-----------------------------

.. code-block:: python

    import os
    from pymongo_voyageai import PyMongoVoyageAI

    client = PyMongoVoyageAI(
        voyageai_api_key=os.environ["VOYAGEAI_API_KEY"],
        s3_bucket_name=os.environ["S3_BUCKET_NAME"],
        mongo_connection_string=os.environ["MONGODB_URI"],
        collection_name="test",
        database_name="test_db",
    )

    url = "hf://datasets/princeton-nlp/CharXiv/val.parquet"
    documents = client.url_to_images(url, image_column="image", end=3)
    resp = client.add_documents(documents)
    client.wait_for_indexing()
    query = "3D loss landscapes for different training strategies"
    data = client.similarity_search(query, extract_images=True)

    # The best match should be the third input image.
    assert data[0]["inputs"][0].image.tobytes() == documents[2].image.tobytes()
    client.delete_by_ids([d["_id"] for d in resp])
    client.close()


Combining Text and Images
-------------------------

.. code-block:: python

    import os
    from pymongo_voyageai import PyMongoVoyageAI

    client = PyMongoVoyageAI(
        voyageai_api_key=os.environ["VOYAGEAI_API_KEY"],
        s3_bucket_name=os.environ["S3_BUCKET_NAME"],
        mongo_connection_string=os.environ["MONGODB_URI"],
        collection_name="test",
        database_name="test_db",
    )

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

    # The interleaved inputs should have different but similar embeddings.
    embeddings = [d["embedding"] for d in resp]
    assert embeddings[2] != embeddings[3]
    assert np.dot(embeddings[2], embeddings[3]) > 0.95
    client.delete_by_ids([d["_id"] for d in resp])
    client.close()


Using Async API
---------------

.. code-block:: python

    import os
    from pymongo_voyageai import PyMongoVoyageAI

    client = PyMongoVoyageAI(
        voyageai_api_key=os.environ["VOYAGEAI_API_KEY"],
        s3_bucket_name=os.environ["S3_BUCKET_NAME"],
        mongo_connection_string=os.environ["MONGODB_URI"],
        collection_name="test",
        database_name="test_db",
    )

    url = "hf://datasets/princeton-nlp/CharXiv/val.parquet"
    documents = await client.aurl_to_images(url, image_column="image", end=3)
    resp = await client.aadd_documents(documents)
    await client.await_for_indexing()

    query = "3D loss landscapes for different training strategies"
    data = await client.asimilarity_search(query, extract_images=True)

    # The best match should be the third input image.
    assert data[0]["inputs"][0].image.tobytes() == documents[2].image.tobytes()
    ids = await client.aget_by_ids([d["_id"] for d in resp])
    assert len(ids) == len(resp)

    await client.adelete_by_ids([d["_id"] for d in resp])
    await client.adelete_many({})
    await client.aclose()
