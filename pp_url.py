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
        (
        abs(img.size[0] - img.size[1]) // 2, 0, abs(img.size[0] - img.size[1]) - abs(img.size[0] - img.size[1]) // 2, 0)
        if img.size[0] < img.size[1] else
        (
        0, abs(img.size[0] - img.size[1]) // 2, 0, abs(img.size[0] - img.size[1]) - abs(img.size[0] - img.size[1]) // 2)
    ), fill=0, padding_mode='constant') if img.size[0] != img.size[1] else img),
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.Lambda(lambda img: img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))),
    transforms.Lambda(lambda img: TF.adjust_contrast(img, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])

_temp = {}

def preprocess_cropped_region(url: str, bbox: tuple):
    """Download, crop bbox, apply preprocessing, and return tensor."""
    if url in _temp:
        img = _temp[url]
    else:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        _temp[url] = img

    x, y, w, h = bbox
    cropped_img = img.crop((x, y, x + w, y + h))
    tensor = transform(cropped_img)
    return tensor
