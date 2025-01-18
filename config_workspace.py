import os
import urllib.request
import zipfile
from tqdm import tqdm

# Définition des chemins
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Remonter au dossier parent de DOLOS
dataset_dir = os.path.join(base_dir, "dataset")
model_dir = os.path.join(base_dir, "model")

def create_directory(directory):
    """Crée un répertoire s'il n'existe pas."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Dossier créé : {directory}")
    else:
        print(f"Dossier déjà existant : {directory}")

# Création des dossiers dataset et model
create_directory(dataset_dir)
create_directory(model_dir)

# Téléchargement et extraction du dataset Tiny ImageNet
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
dataset_zip_path = os.path.join(dataset_dir, "tiny-imagenet-200.zip")

def download_file(url, dest_path):
    """Télécharge un fichier avec une barre de progression."""
    if not os.path.exists(dest_path):
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            file_size = int(response.info().get("Content-Length", -1))
            block_size = 1024
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    pbar.update(len(buffer))
        print(f"Téléchargement terminé : {dest_path}")
    else:
        print(f"Le fichier existe déjà : {dest_path}")

def extract_zip(zip_path, extract_to):
    """Extrait un fichier ZIP avec une barre de progression."""
    if not os.path.exists(os.path.join(extract_to, "tiny-imagenet-200")):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, unit='file', desc="Extraction") as pbar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, extract_to)
                    pbar.update(1)
        print(f"Extraction terminée dans : {extract_to}")
    else:
        print(f"Le dataset est déjà extrait dans : {extract_to}")

# Télécharger et extraire si nécessaire
download_file(url, dataset_zip_path)
extract_zip(dataset_zip_path, dataset_dir)
