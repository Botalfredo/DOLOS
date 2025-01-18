import os
import numpy as np
import datetime
import json
import time
import tensorflow as tf                                                     # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator         # type: ignore
from tensorflow.keras.applications import MobileNetV2                       # type: ignore
from tensorflow.keras.models import Model                                   # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam                                # type: ignore
from tensorflow.keras.callbacks import EarlyStopping                        # type: ignore
import matplotlib.pyplot as plt

# Définition des paramètres
num_classes = 10  # Nombre de classes utilisées
batch_size = 64
img_size = (64, 64)  # Taille des images de Tiny-ImageNet

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Remonter au dossier parent de DOLOS
data_dir = os.path.join(base_dir, "dataset", "tiny-imagenet-processed")
model_dir = os.path.join(base_dir, "model")

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

# Récupération des 50 premières classes
selected_classes = get_first_n_classes(data_dir, num_classes)
selected_classes_text = {cls: word_labels.get(cls, "Unknown") for cls in selected_classes}
print("Classes sélectionnées :", selected_classes_text)

# Création des générateurs de données
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2
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
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callback Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time = time.time()

history = model.fit(train_generator, validation_data=val_generator, epochs=50, callbacks=[early_stopping])

# Calcul du temps d'entraînement
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))

# Sauvegarde du modèle
model_path = os.path.join(model_save_path, "tiny_imagenet_model.h5")
model.save(model_path)

params_file_path = os.path.join(model_save_path, "training_params.txt")
with open(params_file_path, "w") as f:
    f.write(f"Date et Heure: {current_time}\n")
    f.write(f"Dataset utilise: {data_dir}\n")
    f.write(f"Nombre de classes: {num_classes}\n")
    f.write(f"Taille des images: {img_size}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Nombre d'images d'entrainement: {train_generator.samples}\n")
    f.write(f"Nombre d'images de validation: {val_generator.samples}\n")
    f.write(f"Temps total d'entrainement: {total_time_str}\n\n")
    f.write("Paramètres du generateur de donnees:\n")
    f.write(f"Rotation range: {train_datagen.rotation_range}\n")
    f.write(f"Shear range: {train_datagen.shear_range}\n")
    f.write(f"Zoom range: {train_datagen.zoom_range}\n")
    f.write(f"Horizontal flip: {train_datagen.horizontal_flip}\n")
    f.write(f"Vertical flip: {train_datagen.vertical_flip}\n")
    f.write(f"Brightness range: {train_datagen.brightness_range}\n\n")
    f.write("Paramètres EarlyStopping:\n")
    f.write(f"Monitor: {early_stopping.monitor}\n")
    f.write(f"Patience: {early_stopping.patience}\n")
    f.write(f"Restore best weights: {early_stopping.restore_best_weights}\n\n")
    f.write("Classes apprises:\n")
    for class_id, class_name in selected_classes_text.items():
        f.write(f"{class_id}: {class_name}\n")

print(f"Paramètres d'entraînement enregistrés dans {params_file_path}")
