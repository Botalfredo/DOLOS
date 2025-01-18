import tensorflow as tf # type: ignore
import os
import numpy as np
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
import matplotlib.pyplot as plt

# Définition des paramètres
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Remonter au dossier parent de DOLOS
data_dir = os.path.join(base_dir, "dataset", "tiny-imagenet-processed")
model_dir = os.path.join(base_dir, "model")
num_classes = 10  # Utilisation des 10 premières classes
img_size = (64, 64)  # Taille des images de Tiny-ImageNet
batch_size = 64

# Générer un nom de dossier basé sur la date et l'heure
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = os.path.join(model_dir, f"model_{current_time}")
os.makedirs(model_save_path, exist_ok=True)

# Chargement du fichier de correspondance des classes
word_labels_path = os.path.join(data_dir, "words.txt")
word_labels = {}
if os.path.exists(word_labels_path):
    with open(word_labels_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                word_labels[parts[0]] = parts[1]

def get_first_n_classes(data_dir, n):
    """Sélectionne les N premières classes du dataset."""
    class_names = sorted(os.listdir(os.path.join(data_dir, "train")))[:n]
    return class_names

# Récupération des 10 premières classes
selected_classes = get_first_n_classes(data_dir, num_classes)
selected_classes_text = {cls: word_labels.get(cls, "Unknown") for cls in selected_classes}
print("Classes sélectionnées :", selected_classes_text)

# Création des générateurs de données
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalisation des pixels entre 0 et 1
    rotation_range=180,         # Rotation plus large (ex: 360°)
    shear_range=0.2,            # Distorsion affine (effet de cisaillement)
    zoom_range=0.3,             # Zoom avant/arrière aléatoire (30%)
    horizontal_flip=True,       # Symétrie horizontale
    vertical_flip=True,         # Symétrie verticale (utile pour certaines applications)
    brightness_range=[0.7, 1.3],# Variation de luminosité aléatoire
    fill_mode='nearest',        # Remplissage des zones vides après transformation
    validation_split=0.2        # Séparer 20% des images pour la validation
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    classes=selected_classes,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    classes=selected_classes,
    class_mode='categorical',
    subset='validation'
)

# Création du modèle basé sur MobileNetV2
base_model = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # On fige les poids du modèle pré-entraîné

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)  # Augmenter la taille des neurones
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_generator, validation_data=val_generator, epochs=50)

# Sauvegarde du modèle
model_path = os.path.join(model_save_path, "tiny_imagenet_model.h5")
model.save(model_path)

# Sauvegarde des classes apprises avec noms textuels
classes_file_path = os.path.join(model_save_path, "classes_learned.txt")
with open(classes_file_path, "w") as f:
    for class_id, class_name in selected_classes_text.items():
        f.write(f"{class_id}: {class_name}\n")

# Sauvegarde des paramètres d'apprentissage
params_file_path = os.path.join(model_save_path, "training_params.txt")
with open(params_file_path, "w") as f:
    f.write(f"Date et Heure: {current_time}\n")
    f.write(f"Dataset utilisé: {data_dir}\n")
    f.write(f"Nombre de classes: {num_classes}\n")
    f.write(f"Taille des images: {img_size}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Nombre d'images d'entraînement: {train_generator.samples}\n")
    f.write(f"Nombre d'images de validation: {val_generator.samples}\n")
    f.write("Classes apprises:\n")
    for class_id, class_name in selected_classes_text.items():
        f.write(f"{class_id}: {class_name}\n")

# Sauvegarde des graphiques
plt.figure(figsize=(12, 5))
loss_acc_path = os.path.join(model_save_path, "training_curves.png")

# Courbe de perte (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Courbe de la perte')

# Courbe de précision (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbe de la précision')

# Sauvegarde des graphiques
plt.savefig(loss_acc_path)
plt.show()

print(f"Modèle enregistré sous {model_path}")
print(f"Classes apprises enregistrées dans {classes_file_path}")
print(f"Paramètres d'entraînement enregistrés dans {params_file_path}")
print(f"Graphiques sauvegardés sous {loss_acc_path}")
