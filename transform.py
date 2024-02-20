from PIL import Image
from pathlib import Path
from torchvision import transforms, datasets


class TransformConfig:
    center_crop_size = (360, 260)
    resize = (256, 256)


def get_transform_compose():
    cfg = TransformConfig()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(size=cfg.center_crop_size),
        transforms.Resize(size=cfg.resize, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


if __name__ == '__main__':
    from pathlib import Path

    path = '~/content/projects/parametric-face-image-generator/data/output/img/0/0_1.png'
    path = Path(path).expanduser().resolve()
    transform = get_transform_compose()

    ## datasets.ImageFolder
    # dataset = datasets.ImageFolder(path, transform=transform)
    # dataset = [d[0] for d in dataset]
    # print(dataset[0].shape)

    ## convert to cv2 image
    # import cv2
    # img = cv2.imread(path.as_posix())
    # img = transform(img)
    # print(f"{type(img)}\t{img.shape}")
