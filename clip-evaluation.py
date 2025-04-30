import torch
from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor, CLIPModel
from collections import Counter
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import io
import requests # Required by datasets to load images from URLs sometimes

# --- Configuration ---
# DATASET_NAME = "eltorio/ROCOv2-radiology"
DATASET_NAME = "roco_updated_cuis"
MODEL_ID = "clip-finetuned-medical-cui-10000-10"
TRAIN_SUBSET_SIZE = 10000  # Number of training samples to determine candidate concepts
EVAL_SUBSET_SIZE = 4000   # Number of validation samples to evaluate on
NUM_CANDIDATE_CONCEPTS = 75 # Number of top frequent concepts to use as candidates
TOP_K_PREDICTIONS = 1      # Number of top concepts to predict per image

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 1. Load Dataset ---
print(f"Loading dataset {DATASET_NAME}...")
# Load a streaming dataset first to avoid downloading everything if not needed
# We will download specific splits later if necessary
try:
    # Load specific splits and select subsets
    full_dataset = load_from_disk(DATASET_NAME)
    train_ds = full_dataset['train'].shuffle(seed=42).select(range(TRAIN_SUBSET_SIZE)) # Shuffle and select first TRAIN_SUBSET_SIZE samples
    eval_ds = full_dataset['test'].shuffle(seed=42).select(range(EVAL_SUBSET_SIZE)) # Shuffle and select first EVAL_SUBSET_SIZE samples
    print(f"Loaded {len(train_ds)} training samples and {len(eval_ds)} evaluation samples.")
    # Make sure the necessary columns exist
    print(f"Dataset columns: {train_ds.column_names}")
    if 'cui' not in train_ds.column_names or 'image' not in train_ds.column_names:
         raise ValueError("Dataset does not contain required 'image' or 'cui' columns.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure the dataset name is correct and you have network connectivity.")
    exit()

# --- 2. Prepare Candidate Concepts ---
print("Determining candidate concepts...")
all_concepts_train = []

for concepts in tqdm(train_ds['cui'], desc="Scanning train concepts"):
    if concepts: # Check if the list is not None or empty
        all_concepts_train.extend(concepts)
concept_counts = Counter(all_concepts_train)
candidate_concepts = [concept for concept, count in concept_counts.most_common(NUM_CANDIDATE_CONCEPTS)]

if not candidate_concepts:
    print("Error: No candidate concepts found. Check the dataset or subset size.")
    exit()

print(f"Selected {len(candidate_concepts)} candidate concepts (CUIs). Top 10: {candidate_concepts[:10]}")

# --- 3. Load CLIP Model and Processor ---
print(f"Loading CLIP model: {MODEL_ID}...")
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(device)
model.eval() # Set model to evaluation mode

# --- 4. Preprocess Candidate Concepts (Text) ---
# We use the CUIs directly as text input as requested
print("Preprocessing candidate concepts for CLIP...")
# Handle potential empty list
if not candidate_concepts:
     print("Warning: Candidate concept list is empty.")
     text_inputs = None
else:
    text_inputs = processor(text=candidate_concepts, return_tensors="pt", padding=True).to(device)

# --- 5. Process Images and Predict Concepts ---
print(f"Predicting concepts for {len(eval_ds)} images...")
all_predictions = []
all_ground_truths = []

total_tp = 0
total_fp = 0
total_fn = 0

with torch.no_grad(): # Disable gradient calculations for inference
    for item in tqdm(eval_ds, desc="Evaluating images"):
        image_data = item['image']
        ground_truth_concepts = item['cui']

        # Ensure ground truth is a list (handle None or other types)
        if not isinstance(ground_truth_concepts, list):
            ground_truth_concepts = [] # Treat missing/invalid GT as empty list

        # Ensure image data is a PIL Image
        if isinstance(image_data, dict) and 'bytes' in image_data and image_data['bytes']:
             # Handle datasets that might store image bytes
             try:
                 image = Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
             except Exception as e:
                 print(f"Warning: Could not load image from bytes for an item. Skipping. Error: {e}")
                 continue # Skip this image
        elif isinstance(image_data, Image.Image):
             image = image_data.convert("RGB")
        elif isinstance(image_data, str) and image_data.startswith(('http://', 'https://')):
             # Handle image URLs if the dataset provides them
             try:
                 response = requests.get(image_data, stream=True)
                 response.raise_for_status()
                 image = Image.open(response.raw).convert("RGB")
             except Exception as e:
                 print(f"Warning: Could not load image from URL {image_data}. Skipping. Error: {e}")
                 continue # Skip this image
        else:
            print(f"Warning: Unsupported image data format: {type(image_data)}. Skipping.")
            continue # Skip this image

        # Preprocess the image
        try:
             image_inputs = processor(images=[image], return_tensors="pt").to(device)
        except Exception as e:
             print(f"Warning: Could not process image. Skipping. Error: {e}")
             continue # Skip this image if preprocessing fails

        # Get image and text features (check if text_inputs exist)
        if text_inputs is not None and len(candidate_concepts) > 0:
             # Calculate similarity scores
            outputs = model(**image_inputs, **text_inputs)
            logits_per_image = outputs.logits_per_image # This is the similarity score (image vs texts)

            # Get top-k predictions
            top_k_indices = torch.topk(logits_per_image, k=min(TOP_K_PREDICTIONS, len(candidate_concepts)), dim=1).indices.squeeze().tolist()

            # Handle case where k=1, topk returns a single int
            if isinstance(top_k_indices, int):
                top_k_indices = [top_k_indices]

            predicted_concepts = [candidate_concepts[i] for i in top_k_indices]
        else:
             # Handle case with no candidates or text inputs
             predicted_concepts = []


        # Store for evaluation
        all_predictions.append(predicted_concepts)
        all_ground_truths.append(ground_truth_concepts)

        # --- Calculate TP, FP, FN for this image ---
        pred_set = set(predicted_concepts)
        gt_set = set(ground_truth_concepts)

        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set.difference(gt_set))
        fn = len(gt_set.difference(pred_set))

        total_tp += tp
        total_fp += fp
        total_fn += fn

# Variables total_tp, total_fp, total_fn are already calculated in the prediction loop
# Add a counter for exact matches
exact_matches = 0
total_evaluated_images = 0

# Recalculate exact matches (can be done within the main loop or separately if needed)
print("Calculating Exact Match Ratio...")
for i in range(len(all_predictions)):
    # Ensure comparison is between sets for order independence
    pred_set = set(all_predictions[i])
    gt_set = set(all_ground_truths[i])

    if pred_set == gt_set:
        exact_matches += 1
    total_evaluated_images +=1 # Count images actually processed


# --- 6. Calculate Evaluation Metrics ---
print("Calculating evaluation metrics...")

# Avoid division by zero
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
exact_match_ratio = exact_matches / total_evaluated_images if total_evaluated_images > 0 else 0.0

# --- 7. Report Results ---
print("\n--- Concept Detection Results ---")
print(f"Method: CLIP ({MODEL_ID}) predicting Top-{TOP_K_PREDICTIONS} from {NUM_CANDIDATE_CONCEPTS} candidate CUIs.")
print(f"Evaluation Set Size: {total_evaluated_images} images processed") # Use count of processed images

print(f"\nMetrics:")
print(f"  Exact Match Ratio (Accuracy): {exact_match_ratio:.4f} \t(Fraction of predictions where the predicted set exactly matches the ground-truth set)")
print(f"  Precision (Micro-average):    {precision:.4f} \t(TP / (TP + FP))")
print(f"  Recall (Micro-average):       {recall:.4f} \t(TP / (TP + FN))")
print(f"  F1-Score (Micro-average):     {f1:.4f} \t(Harmonic mean of Precision and Recall)")

print(f"\nTotal Counts across all evaluated images:")
print(f"  Total True Positives (TP): {total_tp}")
print(f"  Total False Positives (FP): {total_fp}")
print(f"  Total False Negatives (FN): {total_fn}")
print(f"  Total Exact Matches:       {exact_matches}")


# --- Optional: Show some examples (code remains the same) ---
# ... (rest of the example printing code)

print("\n--- Script Finished ---")