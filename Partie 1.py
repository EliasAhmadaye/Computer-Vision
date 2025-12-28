import os
import random
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt



# Lister les fichiers images

def get_file_list(path, ext):
    ext = ext.lower()
    files = []

    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(ext):
                files.append(os.path.join(root, name))

    return files


# Lire une image
def read_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(filepath)

    return image



# Lire plusieurs images
def read_all_images(filenames, dimensions, keepRatio=True):
    W, H = dimensions
    images = []

    for path in filenames:
        img = read_image(path)

        # Redimensionnement simple (déformation possible)
        if not keepRatio:
            images.append(cv2.resize(img, (W, H)))
            continue

        # Redimensionnement en gardant le ratio
        h, w = img.shape[:2]
        scale = min(W / w, H / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Fond noir + image centrée
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        x = (W - new_w) // 2
        y = (H - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized

        images.append(canvas)

    return images



# Afficher une image
def display_image(image, title=None):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()



# Afficher plusieurs images

def display_images(images, rows, columns):
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()



# Redimensionnement d’image(s)

def resize_image(images, ratio=1.0, max_size=(None, None)):
    max_w, max_h = max_size

    def resize_one(img):
        h, w = img.shape[:2]
        new_w, new_h = int(w * ratio), int(h * ratio)

        if max_w or max_h:
            scale = min(
                (max_w / new_w) if max_w else 1.0,
                (max_h / new_h) if max_h else 1.0
            )
            new_w, new_h = int(new_w * scale), int(new_h * scale)

        return cv2.resize(img, (new_w, new_h))

    # Une image
    if isinstance(images, np.ndarray):
        return resize_one(images)

    # Plusieurs images
    return [resize_one(img) for img in images]



# Mélanger un dataset
def shuffle_dataset(dataset):
    shuffled = list(dataset)
    random.shuffle(shuffled)
    return shuffled

# Diviser un dataset
def split_dataset(dataset, ratios):
    data = list(dataset)
    n = len(data)
    parts = []
    start = 0

    for r in ratios[:-1]:
        end = start + int(r * n)
        parts.append(data[start:end])
        start = end

    parts.append(data[start:])
    return parts



# Sauvegarder / charger dataset

def save_dataset(dataset, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)



# Augmentations d’images
def image_flip(image, horizontal=True, vertical=False):
    if horizontal and vertical:
        return cv2.flip(image, -1)
    if horizontal:
        return cv2.flip(image, 1)
    if vertical:
        return cv2.flip(image, 0)
    return image


def image_crop(image, crop_size, crop_center=True):
    crop_w, crop_h = crop_size
    h, w = image.shape[:2]

    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    if crop_center:
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
    else:
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)

    return image[y:y + crop_h, x:x + crop_w]


def adjust_brightness(image, brightness_factor):
    img = image.astype(np.float32)
    img = img * brightness_factor
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def augment_image(image, augmentations):
    result = image
    for aug in augmentations:
        result = aug(result)
    return result
