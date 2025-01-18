import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image # type: ignore #

# Charger le modèle pré-entraîné
model_path = "tiny_imagenet_model.h5"
model = tf.keras.models.load_model(model_path)

# Charger les classes apprises
class_labels = {}
with open("classes_learned.txt", "r") as f:
    for line in f:
        parts = line.strip().split(": ")
        if len(parts) == 2:
            class_labels[parts[0]] = parts[1]

# Fonction pour classer une image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Redimensionner l'image
    img_array = image.img_to_array(img) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    class_id = list(class_labels.keys())[predicted_class]
    class_name = class_labels.get(class_id, "Unknown")
    
    print(f"L'image {img_path} est classée comme {class_name} (ID: {class_id})")
    return class_name

# Exemple d'utilisation
image_path = "images/Salamandra.jpg"  # Remplacez par votre image
classify_image(image_path)
