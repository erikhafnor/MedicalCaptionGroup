import numpy as np
import json
import torch
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from bleurt import score as bleurt_score
import wandb
# filepath: /home/ansatt/eriksh/bhome/DAT550/project/sentenceEmbedRandomForest.py
import os
import torch

# Set the CUDA device to GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize wandb
wandb.init(project="medical-image-captioning", name="evaluation_RandomForest3")

# Load pre-extracted features and captions
features = np.load("features.npy")
with open("captions.json", "r") as f:
    captions = json.load(f)

# Generate sentence embeddings for captions
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(captions)

# Perform K-means clustering
k = 30  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Prepare training data
X_train = features
y_train = clusters

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Load test dataset
ds = load_dataset("eltorio/ROCOv2-radiology")
test_images = [example["image"] for example in ds["test"]]
test_captions = [example["caption"] for example in ds["test"]]

# Define a transformation pipeline for test images
from torchvision import transforms, models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet model for feature extraction

resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()

def extract_features(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image_tensor)
    return features.squeeze().cpu().numpy()

# Evaluate on test dataset
results = []
examples_table = wandb.Table(columns=["Image", "Reference Caption", "Generated Caption", "BLEU", "BERT-Score", "ROUGE-1", "BLEURT"])

bleurt_scorer = bleurt_score.BleurtScorer("/home/ansatt/eriksh/bhome/DAT550/project/bleurt-20/BLEURT-20")

# Iterate over test images and captions
for idx, (image, caption) in enumerate(zip(test_images, test_captions)):
    # Extract features for the test image
    test_features = extract_features(image).reshape(1, -1)

    # Predict cluster
    predicted_cluster = clf.predict(test_features)[0]

    # Retrieve a representative caption from the predicted cluster
    generated_caption = captions[predicted_cluster]

    # Calculate BLEU score
    reference = [caption.split()]
    candidate = generated_caption.split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)

    # Calculate BERT-Score
    _, bert_recall, _ = bert_score([generated_caption], [caption], model_type="microsoft/deberta-base-mnli", idf=True, device="cpu")

    # Calculate ROUGE-1
    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(caption, generated_caption)
    rouge_1_fmeasure = rouge_scores["rouge1"].fmeasure

    # Calculate BLEURT score
    bleurt_score_value = bleurt_scorer.score(references=[caption], candidates=[generated_caption])[0]

    # Log to wandb
    wandb.log({
        "split": "test",
        "image_id": idx,
        "reference_caption": caption,
        "generated_caption": generated_caption,
        "bleu_score": bleu_score,
        "bert_score": bert_recall.mean().item(),
        "rouge_1": rouge_1_fmeasure,
        "bleurt": bleurt_score_value
    })

    # Add example to wandb.Table
    examples_table.add_data(
        wandb.Image(image),
        caption,
        generated_caption,
        bleu_score,
        bert_recall.mean().item(),
        rouge_1_fmeasure,
        bleurt_score_value
    )

    # Append to results
    results.append({
        "split": "test",
        "image_id": idx,
        "reference_caption": caption,
        "generated_caption": generated_caption,
        "bleu_score": bleu_score,
        "bert_score": bert_recall.mean().item(),
        "rouge_1": rouge_1_fmeasure,
        "bleurt": bleurt_score_value
    })

# Log the examples table to wandb
wandb.log({"Generated Captions Examples": examples_table})

# Finish wandb run
wandb.finish()

print("Evaluation completed.")
