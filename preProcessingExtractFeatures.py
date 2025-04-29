# ACKNOWLEDGMENT
# Chat Generative Pre-trained Transformer 4o (OpenAI) was used for writing assistance by providing feedback on language and text structure, which was subsequently revised by critical appraisal from the author.

from datasets import load_dataset
from torchvision import models, transforms
from PIL import Image
import torch
import wandb
import psutil
import GPUtil
import os
import numpy as np
import json

# Initialize wandb
wandb.init(project="medical-image-captioning", name="feature-extraction_testrun2")

# Log configuration
wandb.config.update({
    "model": "ResNet50",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": (224, 224),
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225]
})

# Set GPU ID 4 as the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


ds = load_dataset("eltorio/ROCOv2-radiology")

# Extract images and captions from the dataset
images = [example["image"] for example in ds["train"]]
captions = [example["caption"] for example in ds["train"]]

# Load a pre-trained ResNet model and move it to the GPU
resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()

# Define a transformation pipeline for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image):
    # Ensure the image is in RGB format
    image = image.convert("RGB")
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
    
    # Extract features using the model
    with torch.no_grad():
        features = resnet(image_tensor)
    
    return features.squeeze().cpu().numpy()  # Move back to CPU for further processing

# Initialize lists to store features and captions
all_features = []
all_captions = []

# Extract features for all images and log to wandb
for idx, (image, caption) in enumerate(zip(images, captions)):
    features = extract_features(image)
    all_features.append(features)
    all_captions.append(caption)
    
    # Log features and captions to wandb
    wandb.log({
        "image_index": idx,
        "caption": caption,
        "features": features.tolist(),  # Log features as a list
        "image": wandb.Image(image, caption=caption)
    })

# Save features and captions to local files
np.save("features.npy", np.array(all_features))  # Save features as a NumPy array
with open("captions.json", "w") as f:
    json.dump(all_captions, f)  # Save captions as a JSON file

print("Features and captions saved locally.")

# Finish wandb run
wandb.finish()
