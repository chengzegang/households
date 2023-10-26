from typing import Callable, Tuple
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2.functional as TF
import os
import numpy as np


def find_paths(root: str, exts: Tuple[str, ...]) -> list[str]:
    paths = []
    exts = [ext.lower() for ext in exts]
    for dirname, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(exts):
                paths.append(os.path.join(dirname, filename))
    return paths


class Images(Dataset):
    def __init__(
        self,
        paths: list[str],
        image_size: int,
        exts: Tuple[str, ...] = ("jpg", "jpeg", "png"),
        transform: Callable | None = None,
        **kwargs
    ):
        self.paths = np.asanyarray(paths)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = read_image(path, mode=ImageReadMode.RGB)
        image = TF.resize(image, self.image_size, antialias=True)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "path": path,
        }

    def __add__(self, other: "Images") -> "Images":
        return Images(
            np.concatenate([self.paths, other.paths]),
            self.image_size,
            transform=self.transform,
        )

    def __radd__(self, other: "Images") -> "Images":
        return other + self


class ImageFolder(Images):
    def __init__(
        self,
        root: str,
        image_size: int,
        exts: Tuple[str, ...] = ("jpg", "jpeg", "png"),
        transform: Callable | None = None,
        **kwargs
    ):
        super().__init__(
            find_paths(root, exts), image_size, exts=exts, transform=transform, **kwargs
        )


class PathAnnotatedImages(ImageFolder):
    def __init__(
        self,
        root: str,
        image_size: int,
        exts: Tuple[str, ...] = ("jpg", "jpeg", "png"),
        separator: str = "_",
        transform: Callable | None = None,
        **kwargs
    ):
        super().__init__(
            find_paths(root, exts), image_size, exts=exts, transform=transform, **kwargs
        )
        self.separator = separator

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        path = data["path"]
        name = os.path.splitext(os.path.basename(path))[0]
        tags = name.split(self.separator)
        data["tags"] = tags
        return data


class VPRDataset(Images):
    coords: np.ndarray
    timestamps: np.ndarray
