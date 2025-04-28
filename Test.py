import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import json
import requests
from io import BytesIO
import os
from tqdm import tqdm
import pandas as pd
from pp_url import preprocess_cropped_region as custom_preprocess

# --- Dataset Loader ---
class FathomDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.annotations = data["annotations"]
        self.image_id_to_url = {img["id"]: img["coco_url"] for img in data["images"]}
        self.annotation_id_to_bbox = {ann["id"]: ann["bbox"] for ann in self.annotations}
        self.annotation_id_to_image_id = {ann["id"]: ann["image_id"] for ann in self.annotations}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        ann_id = ann["id"]
        image_id = ann["image_id"]
        bbox = ann["bbox"]

        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                url = self.image_id_to_url[image_id]
                image_tensor = custom_preprocess(url, bbox)
                return image_tensor, ann_id

            except Exception as e:
                attempt += 1
                print(f"Warning: Failed to process annotation {ann_id} (Attempt {attempt}): {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Failed after {max_retries} attempts to load/process annotation {ann_id}")

def collate_fn(batch):
    images, ids = zip(*batch)
    return torch.stack(images), ids

# --- Main Predict ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    json_path = "datasets/dataset_test.json"
    model_path = "Final Model/bioclip_epoch1_loss0.0000.pth"

    # Load categories
    with open(json_path) as f:
        data = json.load(f)
    categories = data["categories"]
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Prepare ordered category names based on sorted category IDs
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    sorted_cat_names = [cat_id_to_name[cat_id] for cat_id in sorted_cat_ids]

    # Load model (full model, not state_dict!)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Tokenize and encode text features
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    with torch.no_grad():
        text_tokens = tokenizer(sorted_cat_names).to(device)
        text_features = model.module.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Dataset and loader
    test_dataset = FathomDataset(json_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_annotation_ids = []
    all_pred_concepts = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            image_features = model.module.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            preds_idx = logits.argmax(dim=-1)

            for ann_id, pred_idx in zip(ids, preds_idx):
                pred_cat_id = sorted_cat_ids[pred_idx.item()]
                pred_concept_name = cat_id_to_name[pred_cat_id]

                all_annotation_ids.append(ann_id)
                all_pred_concepts.append(pred_concept_name)

    # Save CSV
    df = pd.DataFrame({
        "annotation_id": all_annotation_ids,
        "concept_name": all_pred_concepts
    })
    df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

if __name__ == "__main__":
    main()
