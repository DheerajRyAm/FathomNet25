# FGVC Kaggle Competition 2025

Load Miniconda 24.1.2 with Python 3.10 environment:
```bash
module load miniconda3/24.1.2-py310
```

Install all the required libraries:
```bash
pip install -r requirements.txt
```
To run the model please use this command:
```bash
python run.py
```
It will predict all images in input_image folder and save the result as output.png. In case you want more image from the competition dataset, run this command:
```bash
python image.py --mode train --ann_id 1
```
--mode ["train", "test"] (test dataset has unseen images) | --ann_id for: train: 1-23700; test: 1-788
