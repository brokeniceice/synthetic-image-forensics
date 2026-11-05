"""独立 NPR+SRM 专家网络使用的图像预处理。"""

import torchvision.transforms as transforms


NPR_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NPR_NORMALIZE_STD = [0.229, 0.224, 0.225]


class DualExpertTransform:
    """Generate independent NPR/SRM and FOCAL inputs from one PIL image."""

    def __init__(self, npr_transform, focal_input_size=1024):
        self.npr_transform = npr_transform
        self.focal_transform = transforms.Compose(
            [
                transforms.Resize((focal_input_size, focal_input_size)),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image):
        return {
            "npr": self.npr_transform(image),
            "focal": self.focal_transform(image),
        }


def build_npr_transform(load_size=256, crop_size=224, training=False, no_crop=False):
    """构造官方 NPR+SRM 使用的预处理流程。"""
    operations = [transforms.Resize((load_size, load_size))]
    if not no_crop:
        crop = transforms.RandomCrop(crop_size) if training else transforms.CenterCrop(crop_size)
        operations.append(crop)
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NPR_NORMALIZE_MEAN, std=NPR_NORMALIZE_STD),
        ]
    )
    return transforms.Compose(operations)


def build_npr_focal_transform(
    load_size=256,
    crop_size=224,
    training=False,
    no_crop=False,
    focal_input_size=1024,
):
    """Build the dual-input transform used by NPR+SRM+FOCAL."""
    npr_transform = build_npr_transform(load_size, crop_size, training, no_crop)
    return DualExpertTransform(npr_transform, focal_input_size=focal_input_size)
