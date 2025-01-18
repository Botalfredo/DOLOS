import os
import urllib.request
import zipfile

def download_and_extract_tiny_imagenet(destination_dir="tiny-imagenet"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_filename = "tiny-imagenet-200.zip"
    
    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    zip_path = os.path.join(destination_dir, zip_filename)
    
    # Télécharger le fichier si non existant
    if not os.path.exists(zip_path):
        print("Téléchargement de Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Téléchargement terminé.")
    else:
        print("Le fichier existe déjà.")
    
    # Extraire le fichier zip
    extracted_folder = os.path.join(destination_dir, "tiny-imagenet-200")
    if not os.path.exists(extracted_folder):
        print("Extraction du fichier ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        print("Extraction terminée.")
    else:
        print("Les fichiers sont déjà extraits.")
    
    print("Tiny ImageNet prêt à être utilisé dans", extracted_folder)

# Exécuter le script
download_and_extract_tiny_imagenet()
