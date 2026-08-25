from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size=128):
    """
    Training augmentation using torchvision only.

    Mild color jitter is used because shirt color is important.
    """
    return T.Compose([
        T.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1)
        ),
        T.RandomRotation(degrees=10),
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.05,
            hue=0.0
        ),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
        T.RandomErasing(
            p=0.2,
            scale=(0.02, 0.10)
        )
    ])


def get_val_transform(image_size=128):
    """
    Validation / inference transform.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])