from PIL import Image

from pymongo_voyageai.utils import url_to_images


def test_url_to_images_parquet():
    url = "hf://datasets/princeton-nlp/CharXiv/val.parquet"
    documents = url_to_images(url, image_column="image", end=3)
    assert len(documents) == 3
    assert isinstance(documents[0].image, Image.Image)


def test_url_to_images_pdf():
    url = "https://www.fdrlibrary.org/documents/356632/390886/readingcopy.pdf"
    documents = url_to_images(url)
    assert len(documents) == 21
    assert isinstance(documents[0].image, Image.Image)


def test_url_to_images_png():
    documents = url_to_images("https://www.voyageai.com/header-bg.png")
    assert len(documents) == 1
    assert isinstance(documents[0].image, Image.Image)
