import os
import urllib.request
from zipfile import ZipFile

URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
DEST_DIR = "data/raw"

os.makedirs(DEST_DIR, exist_ok=True)
zip_path = os.path.join(DEST_DIR, "liar_dataset.zip")

print("Downloading...")
urllib.request.urlretrieve(URL, zip_path)

print("Unzipping...")
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DEST_DIR)

print("Done.")
