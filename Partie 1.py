
import os, random, pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_file_list(path, ext):
    ext = ext.lower()
    return [os.path.join(r, f)
            for r, _, fs in os.walk(path)
            for f in fs if f.lower().endswith(ext)]


def read_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(filepath)
    return img


def read_all_images(filenames, dimensions, keepRatio=True):
    W, H = dimensions
    imgs = []
    for fp in filenames:
        img = read_image(fp)
        if not keepRatio:
            imgs.append(cv2.resize(img, (W, H)))
            continue
        h, w = img.shape[:2]
        s = min(W / w, H / h)
        nw, nh = int(w * s), int(h * s)
        r = cv2.resize(img, (nw, nh))
        canvas = np.zeros((H, W, 3), np.uint8)
        y, x = (H - nh) // 2, (W - nw) // 2
        canvas[y:y + nh, x:x + nw] = r
        imgs.append(canvas)
    return imgs


def display_image(image, title=None):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if title: plt.title(title)
    plt.axis("off")
    plt.show()


def display_images(images, rows, columns):
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


# resize_image(images, ratio, (max_width, max_height))
def resize_image(images, ratio=1.0, max_size=(None, None)):
    max_w, max_h = max_size

    def _one(img):
        h, w = img.shape[:2]
        w2, h2 = int(w * ratio), int(h * ratio)
        if max_w or max_h:
            s = min((max_w / w2) if max_w else 1.0, (max_h / h2) if max_h else 1.0)
            w2, h2 = int(w2 * s), int(h2 * s)
        return cv2.resize(img, (w2, h2))

    return _one(images) if isinstance(images, np.ndarray) else [_one(im) for im in images]


def shuffle_dataset(dataset):
    d = list(dataset)
    random.shuffle(d)
    return d


def split_dataset(dataset, ratios):
    d = list(dataset)
    n = len(d)
    cuts, acc = [], 0
    for r in ratios[:-1]:
        acc += int(round(r * n))
        cuts.append(acc)
    out, s = [], 0
    for c in cuts + [n]:
        out.append(d[s:c])
        s = c
    return out


def save_dataset(dataset, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def image_flip(image, horizontal=True, vertical=False):
    if horizontal and vertical: return cv2.flip(image, -1)
    if horizontal: return cv2.flip(image, 1)
    if vertical: return cv2.flip(image, 0)
    return image


def image_crop(image, crop_size, crop_center=True):
    cw, ch = crop_size
    h, w = image.shape[:2]
    cw, ch = min(cw, w), min(ch, h)
    if crop_center:
        x, y = (w - cw) // 2, (h - ch) // 2
    else:
        x, y = random.randint(0, w - cw), random.randint(0, h - ch)
    return image[y:y + ch, x:x + cw]


def adjust_brightness(image, brightness_factor):
    x = image.astype(np.float32) * float(brightness_factor)
    return np.clip(x, 0, 255).astype(np.uint8)


# (objectif) rotation — même si pas listé dans la table
def image_rotate(image, angle_degrees):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_degrees, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# (objectif) normalisation — utile pour ML
def normalize_image(image):
    return image.astype(np.float32) / 255.0


def augment_image(image, augmentations):
    out = image
    for aug in augmentations:
        out = aug(out)
    return out

































if __name__ == "__main__":
    path = r"C:\Users\zenab\PycharmProjects\CV Project\Data\dataset_example\dataset_example\cat"
    ext = ".jpg"
    files = get_file_list(path, ext)

    def need_files():
        if not files:
            raise ValueError("Aucune image trouvée : vérifie path/ext.")
        return files

    def one_image():
        f = need_files()[0]
        return read_image(f)

    while True:
        print("\n=== MENU TESTS ===")
        print("1) get_file_list")
        print("2) read_image")
        print("3) read_all_images (5 images)")
        print("4) display_image (1 image)")
        print("5) display_images (6 images)")
        print("6) resize_image (1 image)")
        print("7) shuffle_dataset (fichiers)")
        print("8) split_dataset (0.7/0.2/0.1)")
        print("9) save_dataset + load_dataset (6 images)")
        print("10) image_flip")
        print("11) image_crop")
        print("12) adjust_brightness")
        print("13) augment_image (pipeline)")
        print("0) quitter")

        c = input("Choix: ").strip()
        if c == "0":
            break

        if c == "1":
            print("Nb fichiers:", len(files))
            print("Exemples:", files[:5])

        elif c == "2":
            img = one_image()
            print("Image OK:", img.shape)

        elif c == "3":
            imgs = read_all_images(need_files()[:5], (300, 300), keepRatio=True)
            print("OK:", len(imgs), imgs[0].shape)

        elif c == "4":
            display_image(one_image(), "Original")

        elif c == "5":
            imgs = read_all_images(need_files()[:6], (300, 300), keepRatio=True)
            display_images(imgs, 2, 3)

        elif c == "6":
            r = resize_image(one_image(), ratio=0.5, max_size=(200, 200))
            print("Resize:", r.shape)
            display_image(r, "Resize")

        elif c == "7":
            s = shuffle_dataset(need_files())
            print("Avant:", need_files()[:5])
            print("Après :", s[:5])

        elif c == "8":
            tr, va, te = split_dataset(need_files(), (0.7, 0.2, 0.1))
            print("train/val/test:", len(tr), len(va), len(te))

        elif c == "9":
            imgs = read_all_images(need_files()[:6], (300, 300), keepRatio=True)
            save_dataset(imgs, "tmp.pkl")
            loaded = load_dataset("tmp.pkl")
            print("Chargé:", len(loaded), loaded[0].shape)

        elif c == "10":
            img = one_image()
            display_image(image_flip(img, horizontal=True), "Flip horizontal")

        elif c == "11":
            img = one_image()
            display_image(image_crop(img, (200, 200), crop_center=True), "Crop center")

        elif c == "12":
            img = one_image()
            display_image(adjust_brightness(img, 0.5), "Sombre")
            display_image(adjust_brightness(img, 1.5), "Clair")

        elif c == "13":
            img = one_image()
            aug = augment_image(img, [
                lambda x: image_flip(x, horizontal=True),
                lambda x: image_crop(x, (200, 200), crop_center=True),
                lambda x: adjust_brightness(x, 1.3),
            ])
            display_image(aug, "Pipeline")

        else:
            print("Choix invalide.")
