# preprocess_utils.py

import requests
from io import BytesIO
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Lambda(lambda img: TF.pad(img, (
        (abs(img.size[0] - img.size[1]) // 2, 0, abs(img.size[0] - img.size[1]) - abs(img.size[0] - img.size[1]) // 2, 0)
        if img.size[0] < img.size[1] else
        (0, abs(img.size[0] - img.size[1]) // 2, 0, abs(img.size[0] - img.size[1]) - abs(img.size[0] - img.size[1]) // 2)
    ), fill=0, padding_mode='constant') if img.size[0] != img.size[1] else img),
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.Lambda(lambda img: img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))),
    transforms.Lambda(lambda img: TF.adjust_contrast(img, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])

def preprocess_image_from_url(url: str, bbox: tuple) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    x, y, w, h = bbox
    cropped_img = img.crop((x, y, x + w, y + h))

    tensor = transform(cropped_img)

    for t, m, s in zip(tensor, CLIP_MEAN, CLIP_STD):
        t.mul_(s).add_(m)

    tensor = torch.clamp(tensor, 0, 1)
    output_img = TF.to_pil_image(tensor)
    return output_img
