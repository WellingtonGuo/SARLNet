import os

import numpy as np
import torch
from PIL import Image


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def scandir(path):
    if path is None or not os.path.isdir(path):
        return []
    files = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(IMG_EXTENSIONS):
                files.append(os.path.join(root, name))
    return sorted(files)


def imread(path, mode="RGB"):
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return img


def pil_to_tensor(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_img(tensor):
    tensor = tensor.detach().float().clamp(0, 1).cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = tensor.permute(1, 2, 0).numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def save_img(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor_to_img(tensor).save(path)

