from PIL import Image, ImageDraw
import os
import numpy as np
import shutil
from tqdm import tqdm

# Définition des chemins
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Remonter au dossier parent de DOLOS
dataset_dir = os.path.join(base_dir, "dataset", "tiny-imagenet-200")
processed_dataset_dir = os.path.join(base_dir, "dataset", "tiny-imagenet-processed")

def add_circular_mask(image):
    """Crée un masque circulaire irrégulier et l'applique à l'image."""
    mask = Image.new("L", (64, 64), 0)
    draw = ImageDraw.Draw(mask)
    
    # Dessiner un cercle de base
    draw.ellipse((2, 2, 62, 62), fill=255)
    
    # Ajouter des irrégularités
    for _ in range(20):  # 20 petites variations
        x_offset = np.random.randint(-3, 3)
        y_offset = np.random.randint(-3, 3)
        radius_variation = np.random.randint(-2, 2)
        draw.ellipse((4 + x_offset, 4 + y_offset, 60 + x_offset + radius_variation, 60 + y_offset + radius_variation), fill=255)
    
    # Appliquer le masque circulaire sur l'image
    img = Image.composite(image, Image.new("RGBA", (64, 64), (0, 0, 0, 255)), mask)
    
    return img

def add_random_noise(image, num_noise_pixels=64):
    """Ajoute du bruit aléatoire sur l'image."""
    noise = np.array(image)
    for _ in range(num_noise_pixels):
        x, y = np.random.randint(0, 64, size=2)
        noise[y, x] = np.random.randint(0, 256, size=4)  # Couleur aléatoire avec alpha
    return Image.fromarray(noise)

def process_and_copy_dataset(input_dir, output_dir):
    """Copie le dataset Tiny ImageNet en appliquant le masque et le bruit, en conservant la structure du dossier."""
    print("Début du traitement du dataset...")
    
    if os.path.exists(output_dir):
        print("Suppression de l'ancienne version du dataset traité...")
        shutil.rmtree(output_dir)  # Supprime l'ancienne version si elle existe
    
    print("Copie de la structure des dossiers...")
    shutil.copytree(input_dir, output_dir, ignore=shutil.ignore_patterns("*.JPEG"))
    
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".JPEG"):
                image_files.append(os.path.join(root, file))
    
    print("Traitement des images...")
    for input_path in tqdm(image_files, desc="Traitement des images", unit="image"):
        output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))
        
        # Charger et traiter l'image
        img = Image.open(input_path).convert("RGBA").resize((64, 64))
        img = add_circular_mask(img)
        img = add_random_noise(img)
        
        # Sauvegarder l'image traitée
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path.replace(".JPEG", ".png"))
    
    print(f"Traitement terminé ! {len(image_files)} images ont été traitées et enregistrées dans {output_dir}.")

# Exécution du script
process_and_copy_dataset(dataset_dir, processed_dataset_dir)
