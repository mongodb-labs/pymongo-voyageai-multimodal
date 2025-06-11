# PyMongo-VoyageAI-Multimodal

PyMongo integration with VoyageAI for multimodal embedding.

## Installation

Requires Python 3.10+.

```bash
pip install --pre pymongo-voyageai-multimodal
```

## Quickstart

Obtain an API key for [VoyageAI](https://docs.voyageai.com/docs/api-key-and-installation).

```python
from pymongo_voyageai_multimodal import PyMongoVoyageAI

# Create our client.
client = PyMongoVoyageAI(
    voyageai_api_key=os.environ["VOYAGE_API_KEY"],
    mongo_connection_string=os.environ["MONGODB_URI"],
    s3_bucket_name="<my-bucket-name>",
    collection_name="test",
    database_name="tests",
)

# Load data from a pdf url.
url = "https://www.fdrlibrary.org/documents/356632/390886/readingcopy.pdf"
images = client.url_to_images(url)
resp = client.add_documents(images)

# Wait for the vector search index to update.
client.wait_for_indexing()

# Query the embeddings, extracting the images.
query = "The consequences of a dictator's peace"
data = client.similarity_search(query, extract_images=True)

# Display the best image match.
data[0]["inputs"][0].image.show()

# Clean up and close the client.
client.delete_many({})
client.close()
```
