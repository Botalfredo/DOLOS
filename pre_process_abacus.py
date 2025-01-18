from PIL import Image
import os
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Retire 2 pixels sur tout le contour de l'image."""
    img = Image.open(image_path)
    width, height = img.size
    
    # Recadrer l'image en enlevant 2 pixels de chaque côté
    cropped_img = img.crop((2, 2, width - 2, height - 2))
    return cropped_img

def split_image(image, output_dir, rows=3, cols=5):
    """Divise une image en sous-images de taille égale et retire 1 pixel sur chaque bord."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    width, height = image.size
    cell_width = width // cols
    cell_height = height // rows
    
    sub_images = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height
            
            # Découper l'image
            sub_image = image.crop((left, upper, right, lower))
            
            # Retirer 1 pixel sur chaque bord
            sub_image = sub_image.crop((1, 1, cell_width , cell_height))
            
            sub_images.append(sub_image)
            sub_image_path = os.path.join(output_dir, f"sub_image_{count}.png")
            sub_image.save(sub_image_path)
            count += 1
    
    print(f"Image divisée en {rows * cols} sous-images et enregistrée dans {output_dir}")
    return sub_images

def display_split_image(image_path, rows=3, cols=5):
    """Prétraite, divise l'image et affiche les sous-images."""
    preprocessed_img = preprocess_image(image_path)
    sub_images = split_image(preprocessed_img, "output_images", rows, cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    for ax, img in zip(axes.flatten(), sub_images):
        ax.imshow(img)
        ax.axis("off")
    plt.show()

# Exemple d'utilisation
display_split_image("test.png")
