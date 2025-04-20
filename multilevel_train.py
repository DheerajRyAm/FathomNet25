import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import json
import requests
from io import BytesIO
from tqdm import tqdm

# ==== Taxonomy Mapping ====
species_to_genus = {
    "Abyssocucumis abyssorum": "Abyssocucumis",
    # Add more species here
}

genus_to_family = {
    "Abyssocucumis": "Cucumarridae",
    # Add more genera here
}

species_list = sorted(set(species_to_genus.keys()))
genus_list = sorted(set(genus_to_family.keys()))
family_list = sorted(set(genus_to_family.values()))

species_to_idx = {s: i for i, s in enumerate(species_list)}
genus_to_idx = {g: i for i, g in enumerate(genera_list)}
family_to_idx = {f: i for i, f in enumerate(family_list)}

# ==== Dataset ====
class FathomTaxonomyDataset(Dataset):
    def __init__(self, json_path, preprocess):
        with open(json_path) as f:
            data = json.load(f)

        self.image_id_to_url = {img["id"]: img["coco_url"] for img in data["images"]}
        self.image_id_to_ann = {}
        for ann in data["annotations"]:
            if ann["category_id"] is not None:
                self.image_id_to_ann[ann["image_id"]] = ann["category_id"]

        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
        self.image_ids = list(self.image_id_to_ann.keys())
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label_id = self.image_id_to_ann[image_id]
        species = self.cat_id_to_name[label_id]
        genus = species_to_genus.get(species, "unknown")
        family = genus_to_family.get(genus, "unknown")

        try:
            url = self.image_id_to_url[image_id]
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_id}: {e}")

        return {
            "image": self.preprocess(image),
            "species": species_to_idx.get(species, 0),
            "genus": genus_to_idx.get(genus, 0),
            "family": family_to_idx.get(family, 0)
        }

# ==== Model ====
class MultiLevelClassifier(nn.Module):
    def __init__(self, base_model, dims):
        super().__init__()
        self.backbone = base_model
        self.fc_species = nn.Linear(512, dims['species'])
        self.fc_genus = nn.Linear(512, dims['genus'])
        self.fc_family = nn.Linear(512, dims['family'])

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone.encode_image(x)
        return {
            "species": self.fc_species(feats),
            "genus": self.fc_genus(feats),
            "family": self.fc_family(feats)
        }

# ==== Collate Function ====
def collate_fn(batch, device):
    images = torch.stack([item["image"] for item in batch]).to(device)
    labels = {
        "species": torch.tensor([item["species"] for item in batch]).to(device),
        "genus": torch.tensor([item["genus"] for item in batch]).to(device),
        "family": torch.tensor([item["family"] for item in batch]).to(device)
    }
    return images, labels

# ==== Training ====
def train():
    json_path = "datasets/dataset_train.json"
    epochs = 5
    batch_size = 8
    lr = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, preprocess_train = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')

    dims = {
        "species": len(species_to_idx),
        "genus": len(genus_to_idx),
        "family": len(family_to_idx)
    }

    model = MultiLevelClassifier(clip_model, dims).to(device)
    dataset = FathomTaxonomyDataset(json_path, preprocess_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    weights = {"species": 1.0, "genus": 0.5, "family": 0.25}

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            outputs = model(images)

            loss = sum(
                weights[level] * loss_fn(outputs[level], labels[level])
                for level in outputs
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "bioclip_multilevel.pth")
    print("Training complete! Saved as bioclip_multilevel.pth")

if __name__ == "__main__":
    train()
