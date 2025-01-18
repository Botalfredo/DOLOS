import tensorflow as tf # type: ignore
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
import matplotlib.pyplot as plt

# Définition des paramètres
data_dir = "tiny-imagenet-processed"  # Assurez-vous que ce chemin est correct
num_classes = 10  # Utilisation des 10 premières classes
img_size = (64, 64)  # Taille des images de Tiny-ImageNet
batch_size = 64

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
    rescale=1./255,  # Normalisation des pixels entre 0 et 1
    rotation_range=180,  # Rotation plus large (ex: 360°)
    #width_shift_range=0.3,  # Décalage horizontal plus grand (30%)
    #height_shift_range=0.3,  # Décalage vertical plus grand (30%)
    shear_range=0.2,  # Distorsion affine (effet de cisaillement)
    zoom_range=0.3,  # Zoom avant/arrière aléatoire (30%)
    horizontal_flip=True,  # Symétrie horizontale
    vertical_flip=True,  # Symétrie verticale (utile pour certaines applications)
    brightness_range=[0.7, 1.3],  # Variation de luminosité aléatoire
    fill_mode='nearest',  # Remplissage des zones vides après transformation
    validation_split=0.2  # Séparer 20% des images pour la validation
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

# Réduire le learning rate (0.0001 au lieu de 0.001)
optimizer = Adam(learning_rate=0.0001)

# Compilation du modèle
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec sauvegarde de l'historique
history = model.fit(train_generator, validation_data=val_generator, epochs=50)

# Sauvegarde du modèle
model.save("tiny_imagenet_model.h5")

# Sauvegarde des classes apprises avec noms textuels
with open("classes_learned.txt", "w") as f:
    for class_id, class_name in selected_classes_text.items():
        f.write(f"{class_id}: {class_name}\n")

print("Modèle enregistré sous tiny_imagenet_processed_model.h5")
print("Classes apprises enregistrées dans classes_learned.txt")

# Affichage des courbes de perte et de précision
plt.figure(figsize=(12, 5))

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

# Afficher les graphiques
plt.show()