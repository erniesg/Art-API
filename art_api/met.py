import sys
from PIL import Image
import io
import urllib.request
import urllib.parse
import requests

BASE_URI = "https://collectionapi.metmuseum.org/public/collection/v1/"

def emo_search(query):
    '''Look for a given work based on user query'''
    url = BASE_URI + "search?q=" + query
    response = requests.get(url).json()
    artwork_lst = response['objectIDs']
    for i, item in enumerate(artwork_lst):
        art_url = BASE_URI + "objects/" + str(item)
        art_response = requests.get(art_url).json()
        if art_response['constituents'] is not None and art_response['primaryImage']:
            print(f"{art_response['title']} by : {art_response['constituents'][0]['name']} at {art_response['primaryImage']}")
            img_response = requests.get('https://picsum.photos/seed/picsum/200/300')
            in_memory_file = io.BytesIO(img_response.content)
            im = Image.open(in_memory_file)
            im.show()

def main():
    '''Ask user for a keyword to be used as query'''
    query = input("Show me\n> ")
    emo = emo_search(query)

if __name__ == '__main__':
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print('\nGoodbye!')
        sys.exit(0)