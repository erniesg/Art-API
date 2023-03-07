import csv
import os
from docarray import dataclass
from docarray import Document, DocumentArray
from docarray.typing import Image, Text


@dataclass
class Artwork:
    image: Image
    title: Text
    artist: Text
    dating: Text
    geo_reference: Text
    rights: Text


image_folder = '../raw_data/aws10k_sm'


def convert_to_docarray(filename):
    existing_images = os.listdir(image_folder)

    artworks = DocumentArray()

    with open(filename) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',')
        for row in csv_reader:
            image_file = f'{row["Accession No."]}.jpg'
            if image_file in existing_images:
                artwork = Artwork(
                    image=f'{image_folder}/{image_file}',
                    title=row['Title'],
                    artist=row['Artist/Maker'],
                    dating=row['Dating'],
                    geo_reference=row['Geo. Reference'],
                    rights=row['Rights']
                )
                doc = Document(artwork)
                artworks.append(doc)

    artworks.save_binary('artworks_ngs_all.docarray')
    artworks[:20].save_binary('artworks_ngs_small.docarray')


def push_all():
    da = DocumentArray.load_binary(f'artworks_ngs_all.docarray')
    da.push(name='artworks_ngs_all')


convert_to_docarray('../raw_data/csv/ngs_artplus.csv')
push_all()