import torch
from PIL import Image
import open_clip
import json
import requests
from io import BytesIO
import os


def main():
    output_dir = "predicted_images"
    os.makedirs(output_dir, exist_ok=True)

    # Load BioCLIP model and tokenizer
    model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Load dataset
    with open("datasets/dataset_train.json", "r") as f:
        rois = json.load(f)

    # Define biologically meaningful labels (customize if needed)
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

    text_tokens = tokenizer(labels).to(device)



    # Tokenize text labels
    text_tokens = tokenizer(labels)
    # Loop through a few image entries for demonstration
    for entry in rois["images"][:10]:  # Change or remove limit as needed
        url = entry["coco_url"]
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_tensor = preprocess_val(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                probs = (image_features @ text_features.T).softmax(dim=-1)
                pred_idx = probs.argmax().item()
                pred_label = labels[pred_idx]


                save_path = os.path.join(output_dir, f"{pred_label}_{entry['file_name']}")
                image.save(save_path)

            print(f"Name:{entry['file_name']}: Predicted class = {labels[pred_idx]}")
        except Exception as e:
            print(f"Failed to process image {entry['id']} from {url}: {e}")


if __name__ == "__main__":
    main()