import json
import os
import argparse
import requests
from PIL import Image
from io import BytesIO

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def find_annotation(annotations, annotation_id):
    for ann in annotations:
        if ann['id'] == annotation_id:
            return ann
    return None

def find_image(images, image_id):
    for img in images:
        if img['id'] == image_id:
            return img
    return None

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert('RGB')

def crop_and_save(image, bbox, output_path):
    x, y, width, height = bbox
    cropped = image.crop((x, y, x + width, y + height))
    cropped.save(output_path)

def run(mode, annotation_id):
    json_path = f'datasets/dataset_{mode}.json'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    data = load_json(json_path)
    annotations = data['annotations']
    images = data['images']

    annotation = find_annotation(annotations, annotation_id)
    if annotation is None:
        print(f"Error: annotation id {annotation_id} not found.")
        return

    image_info = find_image(images, annotation['image_id'])
    if image_info is None:
        print(f"Error: image id {annotation['image_id']} not found.")
        return

    img_url = image_info['coco_url']
    print(f"Downloading image from {img_url}...")
    image = download_image(img_url)

    output_dir = 'input_image'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'annotation_{annotation_id}.png')
    crop_and_save(image, annotation['bbox'], output_path)
    print(f"Cropped image saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="test", help="Run mode: train or test")
    parser.add_argument("--ann_id", type=int, required=True, help="Annotation ID to crop")
    args = parser.parse_args()

    run(args.mode, args.ann_id)
