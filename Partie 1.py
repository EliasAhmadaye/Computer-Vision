import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import pickle

path = r"C:\Users\zenab\PycharmProjects\CV Project\Data\dataset_example"


def get_file_list(path, ext):
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(ext.lower())
    ]

files = get_file_list(path, ".jpg")
#print("Fichiers trouvés :", files)

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Cannot read image")
    return img


#img = read_img(files[0])
#path_test = (r"C:\Users\zenab\PycharmProjects\CV Project\Data\dataset_example\dataset_example\dog\dog.999.jpg")

#img = read_img(path_test)

#print("Image lues avec succés")
#print(img.shape)

def read_all_image(filenames, dimensions, KeepRatio=True):
    target_w, target_h = dimensions
    images = []

    for filepath in filenames:
        img = read_img(filepath)  #on lit l'image

        if KeepRatio:
            h, w = img.shape[:2]

            scale = min(target_w / w, target_h / h)
            nh = int(h * scale)
            nw = int(w * scale)
            #redimensionner l'image
            resized = cv2.resize(img, (nw, nh))
            #creer un fond noir  plus centre l'image
            canvas = np.zeros((target_h, target_w, 3), dtype="uint8")
            x = (target_w - nw) // 2
            y = (target_h - nh) // 2
            canvas[y:y + nh, x:x + nw] = resized
            images.append(canvas) # on
        else:
            resized = cv2.resize(img, (target_w, target_h))
            images.append(resized)
    return images

imgs = read_all_image(files, (300, 300), KeepRatio=True)

#print("Nombre d'images dans le dataset :", len(imgs))
#print("Taille de la première image :", imgs[999].shape)


def display_image(image, title=None):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

#img = read_img(files[50])
#display_image(img, title="Première image du dataset")

def display_images(images, rows, columns):
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
    axes = axes.flatten()  # transforme en liste pour accéder facilement aux cases

    for i, ax in enumerate(axes):
        ax.axis("off")  # enlève les bordures

        if i < len(images):
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()

#imgs = read_all_image(files, (200, 200))
#display_images(imgs[:57], 2, 3)
def resize_image(images, ratio=None, max_size=None):
        def _resize(img):
            h, w = img.shape[:2]

            if ratio is not None:  # mode ratio
                new_w = int(w * ratio)
                new_h = int(h * ratio)

            elif max_size is not None:  # mode max_size
                max_w, max_h = max_size
                scale = min(max_w / w, max_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

            else:
                return img

            return cv2.resize(img, (new_w, new_h))

        if isinstance(images, np.ndarray):
            return _resize(images)

        return [_resize(img) for img in images]


"""""
resized_imgs = resize_image(imgs, max_size=(300, 300))

print("Images originales :", len(imgs))
resized_imgs = resize_image(imgs, max_size=(300, 300))
print("Images redimensionnées :", len(resized_imgs))
print("Taille exemple :", resized_imgs[0].shape)

"""""

def shuffle_dataset(dataset):
    d = list(dataset)     # On crée une copie du dataset
    random.shuffle(d)     # On mélange les éléments de manière aléatoire
    return d              # On retourne le dataset mélangé
"""
print("=== TEST SHUFFLE DATASET ===")

print("\nAvant mélange :")
for i in range(5):
    print(" -", files[i])

shuffled_files = shuffle_dataset(files)

print("\nAprès mélange :")
for i in range(5):
    print(" -", shuffled_files[i])
"""

def save_dataset(dataset, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)
save_dataset(imgs, "saved_imgs.pkl")
print("Dataset sauvegardé !")

def load_dataset(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

path = r"C:\Users\zenab\PycharmProjects\CV Project\Data\dataset_example\dataset_example\dog"  # ou cat

files = get_file_list(path, ".jpg")
print("➡ Nombre de fichiers trouvés dans le dossier :", len(files))

if len(files) == 0:
    print("⚠ AUCUN fichier .jpg trouvé, vérifie le chemin ou l'extension.")
# Lire toutes les images (seulement si on a trouvé des fichiers)
imgs = read_all_image(files, (300, 300), KeepRatio=True)
print("➡ Nombre d'images dans imgs après read_all_images :", len(imgs))

def save_dataset(dataset, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)

def load_dataset(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

print("➡ Nombre d'images dans imgs avant sauvegarde :", len(imgs))

save_path = r"C:\Users\zenab\PycharmProjects\CV Project\.venv\saved_imgs.pkl"
save_dataset(imgs, save_path)
print("Dataset sauvegardé !")

loaded = load_dataset(save_path)
print("➡ Nombre d'images chargées :", len(loaded))

if loaded:
    print("Taille de la première image :", loaded[0].shape)
else:
    print("⚠ Le dataset chargé est vide.")
