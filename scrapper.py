import os
import pathlib

import requests
from bs4 import BeautifulSoup


def download_image(u, save_path):
    res = requests.get(u)
    if res.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(res.content)
        print(f"Downloaded {u} as {save_path}")
    else:
        print(f"Failed to download {u}")


# URL of the page to scrape
url = "https://dicegrimorium.com/free-rpg-map-library/"

# Send a request to the website
response = requests.get(url)

# Parse the content of the page
soup = BeautifulSoup(response.content, 'html.parser')

# Find all map thumbnails and titles
map_sites = soup.find_all('a', class_='custom-link no-lightbox')
image_data = {}

for map_site in map_sites:
    map_site = map_site['href']
    r = requests.get(map_site)
    s = BeautifulSoup(r.content, 'html.parser')
    main_table = s.find("figure", {'class': 'wp-block-image size-large'})
    if main_table is None:
        continue

    link = main_table.find("img")['src']
    title = map_site.split('/')[3]
    path = pathlib.Path("/media/kwoj/borrowed/Projekt_Uczenie_Maszyn/dnd_scrapped_maps", title + ".png")
    download_image(link, path)
