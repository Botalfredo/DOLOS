import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
# Définition des paramètres
data_dir = "tiny-imagenet-200"
img_size = (64, 64)
batch_size = 64

# Charger les classes apprises avec noms textuels
class_labels = {}
with open("classes_learned.txt", "r") as f:
    for line in f:
        parts = line.strip().split(": ")
        if len(parts) == 2:
            class_labels[parts[0]] = parts[1]

selected_classes = list(class_labels.keys())

# Définir ImageDataGenerator pour charger les données de validation
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    classes=selected_classes,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important pour garder l'ordre des étiquettes
)

# Charger le modèle entraîné
model = tf.keras.models.load_model("tiny_imagenet_model.h5")

# Générer les vraies étiquettes et les prédictions
true_labels = val_generator.classes  # Classes réelles des images de validation
class_names = [class_labels[cls] for cls in val_generator.class_indices.keys()]  # Noms des classes

# Prédictions sur le dataset de validation
y_pred_prob = model.predict(val_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# Créer la matrice de confusion
cm = confusion_matrix(true_labels, y_pred)

# Afficher la matrice de confusion avec seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prédictions")
plt.ylabel("Vraies classes")
plt.title("Matrice de Confusion")
plt.show()

# Afficher le rapport de classification
print("\nRapport de Classification:")
print(classification_report(true_labels, y_pred, target_names=class_names))
