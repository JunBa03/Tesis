"""Transformaciones de imagen.

Restriccion del dominio: NO se aplica flip horizontal. En una radiografia de torax
la inversion ubica el corazon a la derecha de la linea media, lo que corresponde a
situs inversus -- una condicion real ausente en CheXpert -- e introduce una senal de
entrenamiento incorrecta sobre la lateralidad cardiaca.

Las transformaciones admitidas son las que pueden ocurrir por variabilidad de
adquisicion sin alterar relaciones anatomicas: rotaciones pequenas (<10 grados),
recorte aleatorio con escalado y ajustes moderados de brillo y contraste.
"""
from __future__ import annotations

from torchvision import transforms

# Estadisticas de ImageNet: se usan porque los backbones vienen preentrenados en ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
