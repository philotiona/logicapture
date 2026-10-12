import numpy as np
from PIL import Image


def image_vector(image):
    return np.asarray(Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB"), dtype=np.uint8)


def preprocess(image):
    if not isinstance(image, Image.Image): image = Image.fromarray(image)
    return np.asarray(image.convert("L").resize((48, 48)), dtype=np.float32)[..., None]/255


def prototype_predict(support, labels, queries, embed=None):
    labels = np.asarray(labels, dtype=int)
    if set(labels) != {0, 1}: raise ValueError("Both prototype classes require support examples")
    x = np.stack([preprocess(image) for image in support])
    q = np.stack([preprocess(image) for image in queries])
    if embed is None:
        x, q = x.reshape(len(x), -1), q.reshape(len(q), -1)
    else:
        x, q = np.asarray(embed(x, training=False)), np.asarray(embed(q, training=False))
    prototypes = np.stack([x[labels == label].mean(axis=0) for label in (0, 1)])
    distance = ((q[:, None, :]-prototypes[None, :, :])**2).mean(axis=-1)
    return distance.argmin(axis=1).astype(bool)
