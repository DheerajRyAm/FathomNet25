import os
from training.main import main as train_main

def train_bioclip():
    args = [
        "--train-data", "data/data.tsv",
        "--dataset-type", "csv",
        "--csv-img-key", "image_path",
        "--csv-caption-key", "text",
        "--pretrained", "hf-hub:imageomics/bioclip",
        "--model", "ViT-B-32",
        "--batch-size", "64",
        "--epochs", "10",
        "--lr", "1e-5",
        "--warmup", "100",
        "--logs", "logs/",
        "--save-frequency", "1",
        "--precision", "amp",
        "--report-to", "tensorboard"
    ]
    train_main(args)

if __name__ == "__main__":
    train_bioclip()
