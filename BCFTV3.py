import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import open_clip
from tqdm import tqdm

# --- Dataset Loading and Preprocessing ---
class TaxonomyDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_index = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, label_index

# Load COCO-style annotations
json_path = "datasets/dataset_train.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Build mappings
cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(cat_id_to_name))}
idx_to_name = {idx: cat_id_to_name[cat_id] for cat_id, idx in cat_id_to_idx.items()}

# Prepare samples
samples = []
for ann in data["annotations"]:
    img_id = ann["image_id"]
    ann_id = ann["id"]
    cat_id = ann["category_id"]
    filename = f"{img_id}_{ann_id}.png"
    image_path = os.path.join("data", "cleaned_all", filename)
    label_index = cat_id_to_idx[cat_id]
    samples.append((image_path, label_index))

# Shuffle and split
np.random.seed(42)
np.random.shuffle(samples)
split_idx = int(0.8 * len(samples))
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

# Load model and transforms
model_name = "hf-hub:imageomics/bioclip"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

# Create save directory
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If multiple GPUs available, use DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to(device)
model_module = model.module if isinstance(model, torch.nn.DataParallel) else model

# Prepare optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Load existing model checkpoint
checkpoint_path = "bioclip_epoch30_loss0.1182.pth"
if os.path.exists(checkpoint_path):
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found, starting from scratch.")

# Prepare category text embeddings once
category_texts = list(idx_to_name.values())
with torch.no_grad():
    tokenized = tokenizer(category_texts).to(device)
    category_embeddings = model_module.encode_text(tokenized, normalize=True)

# Dataloaders
train_dataset = TaxonomyDataset(train_samples, preprocess_train)
val_dataset = TaxonomyDataset(val_samples, preprocess_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# --- Training ---
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, label_indices in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        images = images.to(device)
        label_indices = label_indices.to(device)

        image_features = model_module.encode_image(images, normalize=True)
        logits = model_module.logit_scale.exp() * image_features @ category_embeddings.T
        loss = F.cross_entropy(logits, label_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

    # Save every 15 epochs
    if (epoch + 1) % 15 == 0:
        save_name = f"{save_dir}/bioclip_epoch{epoch+1}_loss{avg_loss:.4f}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved model at {save_name}")

# --- Evaluation ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, label_indices in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        label_indices = label_indices.to(device)
        image_features = model_module.encode_image(images, normalize=True)
        logits = model_module.logit_scale.exp() * image_features @ category_embeddings.T
        preds = logits.argmax(dim=1)
        correct += (preds == label_indices).sum().item()
        total += label_indices.size(0)

val_acc = 100.0 * correct / total
print(f"Validation Accuracy: {val_acc:.2f}%")

# Final Save
torch.save(model.state_dict(), f"{save_dir}/bioclip_finetuned_final.pth")
print(f"Saved final fine-tuned model to {save_dir}/bioclip_finetuned_final.pth")
