import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from collections import defaultdict

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

class BalancedCategorySampler(Sampler):
    def __init__(self, category_to_samples, batch_size_per_category=1, desired_batches=100):
        self.category_to_samples = category_to_samples
        self.batch_size_per_category = batch_size_per_category
        self.desired_batches = desired_batches
        self.categories = list(category_to_samples.keys())

    def __iter__(self):
        all_indices = []
        for cat in self.categories:
            samples = self.category_to_samples[cat]
            chosen_samples = np.random.choice(len(samples), self.batch_size_per_category)
            for idx in chosen_samples:
                all_indices.append(samples[idx])
        np.random.shuffle(all_indices)

        total_samples = len(all_indices)
        batch_size = max(1, total_samples // self.desired_batches)

        for i in range(0, total_samples, batch_size):
            yield all_indices[i:i+batch_size]

    def __len__(self):
        return self.desired_batches

json_path = "datasets/dataset_train.json"
with open(json_path, "r") as f:
    data = json.load(f)

cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(cat_id_to_name))}
idx_to_name = {idx: cat_id_to_name[cat_id] for cat_id, idx in cat_id_to_idx.items()}

samples = []
for ann in data["annotations"]:
    img_id = ann["image_id"]
    ann_id = ann["id"]
    cat_id = ann["category_id"]
    filename = f"{img_id}_{ann_id}.png"
    image_path = os.path.join("data", "cleaned_all", filename)
    label_index = cat_id_to_idx[cat_id]
    samples.append((image_path, label_index))

category_to_samples = defaultdict(list)
for image_path, label_index in samples:
    category_to_samples[label_index].append((image_path, label_index))

train_samples = []
val_samples = []
for label_index, items in category_to_samples.items():
    items = sorted(items)
    num_total = len(items)
    split_idx = int(0.9 * num_total)
    train_items = items[:split_idx]
    val_items = items[split_idx:]
    np.random.shuffle(train_items)
    train_samples.extend(train_items)
    val_samples.extend(val_items)

print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(val_samples)}")

model_name = "hf-hub:imageomics/bioclip"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

save_dir = "Saved_6"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to(device)
model_module = model.module if isinstance(model, torch.nn.DataParallel) else model

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
checkpoint_path = ""
#checkpoint_path = "Saved_6/bioclip_epoch1_loss0.000000100027704.pth"
if os.path.exists(checkpoint_path):
    print(f"Loading model weights from {checkpoint_path}")
    model_module = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_module
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found, starting from scratch.")

category_texts = list(idx_to_name.values())
with torch.no_grad():
    tokenized = tokenizer(category_texts).to(device)
    category_embeddings = model_module.encode_text(tokenized, normalize=True)

train_dataset = TaxonomyDataset(train_samples, preprocess_train)
val_dataset = TaxonomyDataset(val_samples, preprocess_val)

train_category_to_samples = defaultdict(list)
for idx, (image_path, label_index) in enumerate(train_samples):
    train_category_to_samples[label_index].append(idx)

balanced_sampler = BalancedCategorySampler(train_category_to_samples, batch_size_per_category=1, desired_batches=100)

train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

model.eval()
initial_total_loss = 0
initial_correct = 0
initial_total = 0

with torch.no_grad():
    for images, label_indices in tqdm(val_loader, desc="Initial Evaluation"):
        images = images.to(device)
        label_indices = label_indices.to(device)
        image_features = model_module.encode_image(images, normalize=True)
        logits = model_module.logit_scale.exp() * image_features @ category_embeddings.T
        loss = F.cross_entropy(logits, label_indices)
        initial_total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        initial_correct += (preds == label_indices).sum().item()
        initial_total += label_indices.size(0)

initial_avg_loss = initial_total_loss / initial_total
initial_val_acc = 100.0 * initial_correct / initial_total

print(f"Initial Validation Loss: {initial_avg_loss:.50f}")
print(f"Initial Validation Accuracy: {initial_val_acc:.2f}%")

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
    print(f"Epoch {epoch+1} - Average Training Loss: {avg_loss:.50f}")

    save_name = f"{save_dir}/bioclip_epoch{epoch+1}_loss{avg_loss:.15f}.pth"
    torch.save(model_module, save_name)
    print(f"Saved full model at {save_name}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, label_indices in tqdm(val_loader, desc="Final Validation"):
        images = images.to(device)
        label_indices = label_indices.to(device)
        image_features = model_module.encode_image(images, normalize=True)
        logits = model_module.logit_scale.exp() * image_features @ category_embeddings.T
        preds = logits.argmax(dim=1)
        correct += (preds == label_indices).sum().item()
        total += label_indices.size(0)

val_acc = 100.0 * correct / total
print(f"Final Validation Accuracy: {val_acc:.2f}%")

final_save_name = f"{save_dir}/bioclip_finetuned_final.pth"
torch.save(model_module, final_save_name)
print(f"Saved final full model at {final_save_name}")
