import requests
from io import BytesIO
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import os
import open_clip
import matplotlib.pyplot as plt

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
labels = [
    "Abyssocucumis abyssorum", "Acanthascinae", "Acanthoptilum", "Actinernus", "Actiniaria",
    "Actinopterygii", "Amphipoda", "Apostichopus leukothele", "Asbestopluma", "Asbestopluma monticola",
    "Asteroidea", "Benthocodon pedunculata", "Brisingida", "Caridea", "Ceriantharia",
    "Chionoecetes tanneri", "Chorilia longipes", "Corallimorphus pilatus", "Crinoidea", "Delectopecten",
    "Elpidia", "Farrea", "Florometra serratissima", "Funiculina", "Gastropoda",
    "Gersemia juliepackardae", "Heterocarpus", "Heterochone calyx", "Heteropolypus ritteri", "Hexactinellida",
    "Hippasteria", "Holothuroidea", "Hormathiidae", "Isidella tentaculum", "Isididae",
    "Isosicyonis", "Keratoisis", "Liponema brevicorne", "Lithodidae", "Mediaster aequalis",
    "Merluccius productus", "Metridium farcimen", "Microstomus pacificus", "Munidopsis", "Munnopsidae",
    "Mycale", "Octopus rubescens", "Ophiacanthidae", "Ophiuroidea", "Paelopatides confundens",
    "Pandalus amplus", "Pandalus platyceros", "Pannychia moseleyi", "Paragorgia", "Paragorgia arborea",
    "Paralomis multispina", "Parastenella", "Peniagone", "Pennatula phosphorea", "Porifera",
    "Psathyrometra fragilis", "Psolus squamatus", "Ptychogastria polaris", "Pyrosoma atlanticum",
    "Rathbunaster californicus",
    "Scleractinia", "Scotoplanes", "Scotoplanes globosa", "Sebastes", "Sebastes diploproa",
    "Sebastolobus", "Serpulidae", "Staurocalyptus", "Strongylocentrotus fragilis", "Terebellidae",
    "Tunicata", "Umbellula", "Vesicomyidae", "Zoantharia"
]

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

def preprocess_image(img) -> Image.Image:
    tensor = transform(img)
    return tensor

def visualize_samples(imgs, ids):
    plt.figure(figsize=(15, 6))
    for i, img in enumerate(imgs):
        plt.subplot(2, (len(imgs) + 1) // 2, i + 1)
        plt.imshow(img)
        plt.title(labels[ids[i]])
        plt.savefig("output.png")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

input_path = 'input_image'
image_extensions = ('.png', '.jpg', '.jpeg')
image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(image_extensions)]
images = [Image.open(path) for path in image_paths]

# Load the same model architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

# Load your trained weights
# Load the checkpoint
state_dict = torch.load("bioclip_finetuned_1.pth", map_location=device)

# Fix "module." prefix if present
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_k] = v

# Load into model
model.load_state_dict(new_state_dict)

model.to(device)

text_tokens = tokenizer(labels).to(device)
ids = []

for img in images:
    image_tensor = preprocess_image(img)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor.unsqueeze(0).to(device))
        text_features = model.encode_text(text_tokens)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T)
    pred_idx = similarity.argmax().item()
    ids.append(pred_idx)

visualize_samples(images, ids)
