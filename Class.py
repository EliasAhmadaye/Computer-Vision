import os
import random
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageDataset:
    def __init__(self, folder=None, ext="jpg"):
        self.folder = folder
        self.ext = "." + ext.lower().lstrip(".")
        self.files = []
        self._cache = {}  # cache pour get_all_images

        if folder is None:
            print("Empty Dataset created.")
            return

        for f in os.listdir(folder):
            if f.lower().endswith(self.ext):
                self.files.append(os.path.join(folder, f))

        print(f"Dataset created. {len(self.files)} images found")

    # --- base ---
    def __len__(self):
        return len(self.files)

    def __iter__(self):
        # permet: sorted(dataset), list(dataset), for x in dataset
        return iter(self.files)

    def get_image(self, index):
        path = self.files[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def get_all_images(self, dimensions, keepRatio=True):
        key = (dimensions, keepRatio)
        if key in self._cache:
            return self._cache[key]

        W, H = dimensions
        N = len(self.files)
        images = np.zeros((N, H, W, 3), dtype=np.uint8)

        for i in range(N):
            img = self.get_image(i)
            h, w = img.shape[:2]

            if keepRatio:
                s = min(W / w, H / h)
                nw, nh = int(w * s), int(h * s)
                resized = cv2.resize(img, (nw, nh))

                canvas = np.zeros((H, W, 3), dtype=np.uint8)
                x = (W - nw) // 2
                y = (H - nh) // 2
                canvas[y:y + nh, x:x + nw] = resized
                images[i] = canvas
            else:
                images[i] = cv2.resize(img, (W, H))

        self._cache[key] = images
        return images

    # --- affichage ---
    def display_image(self, index, title=None):
        img = self.get_image(index)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.show()

    def display_images(self, indices, grid, size=(200, 200)):
        rows, columns = grid

        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
        axes = np.array(axes).reshape(-1)

        idx_list = list(indices)
        for ax, i in zip(axes, idx_list):
            img = self.get_image(i)
            img = cv2.resize(img, size)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Idx {i}")
            ax.axis("off")

        for ax in axes[len(idx_list):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # --- operations dataset ---
    def shuffle(self):
        random.shuffle(self.files)
        self._cache.clear()  # ordre changé => cache invalide
        return self

    def split_dataset(self, ratios):
        ratios = np.array(ratios, dtype=float)
        ratios = ratios / ratios.sum()  # accepte (0.6,0.2,0.2) etc.

        n = len(self.files)
        cuts = (ratios.cumsum() * n).astype(int)
        cuts[-1] = n

        start = 0
        parts = []
        for end in cuts:
            ds = ImageDataset.__new__(ImageDataset)  # évite __init__ + prints
            ds.folder = self.folder
            ds.ext = self.ext
            ds.files = self.files[start:end].copy()
            ds._cache = {}
            parts.append(ds)
            start = end

        return tuple(parts)

    def __add__(self, other):
        ds = ImageDataset.__new__(ImageDataset)
        ds.folder = self.folder or other.folder
        ds.ext = self.ext
        ds.files = self.files + other.files
        ds._cache = {}
        return ds

    def __eq__(self, other):
        # dataset == dataset : même ordre
        if isinstance(other, ImageDataset):
            return self.files == other.files

        # dataset == sorted(datasetX) : mêmes éléments, ordre ignoré
        if isinstance(other, (list, tuple)):
            return sorted(self.files) == list(other)

        return False

    # --- save / load ---
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Dataset object saved to the file {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            other = pickle.load(f)

        self.folder = other.folder
        self.ext = other.ext
        self.files = other.files
        self._cache = {}

        print(f"Dataset is loaded from {filepath}")
        return self
