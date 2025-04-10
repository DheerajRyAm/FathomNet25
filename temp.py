import torch
from PIL import Image
import open_clip
def main():
    # Load BioCLIP model and tokenizer
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load your image
    image = Image.open("your_image.jpg").convert("RGB")
    image_tensor = preprocess_val(image).unsqueeze(0).to(device)  # shape: [1, 3, H, W]

    # Define your classes
    labels = ["normal tissue", "cancerous tissue", "inflamed tissue"]

    # Tokenize text labels
    text_tokens = tokenizer(labels).to(device)

    # Encode image and text
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # Compute cosine similarity
        similarity = image_features @ text_features.T
        probs = similarity.softmax(dim=-1)

    # Print prediction
    predicted_index = probs.argmax().item()
    print(f"Predicted class: {labels[predicted_index]}")
    print(f"Probabilities: {probs.cpu().numpy()}")
