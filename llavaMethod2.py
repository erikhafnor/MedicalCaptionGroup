import os
import io
import json
import requests
import base64
import string
from PIL import Image
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from bleurt import score as bleurt_score
import wandb

# Initialize wandb
wandb.init(project="llava-medical-image-evaluation", name="llava-relevance-evaluation")

# Preprocessing function for captions
def preprocess_caption(caption):
    caption = caption.lower()  # Convert to lowercase
    caption = ''.join(c for c in caption if c not in string.punctuation)  # Remove punctuation
    caption = ' '.join('number' if word.isdigit() else word for word in caption.split())  # Replace numbers
    return caption

# Function to compute image and caption similarity (placeholder for medical embedding model)
def compute_image_caption_similarity(image_embedding, caption_embedding):
    # Placeholder: Replace with actual embedding similarity calculation
    return float((image_embedding @ caption_embedding.T) / (image_embedding.norm() * caption_embedding.norm()))

# Function to process a single image and get the generated caption
def process_image(image, llava_url, base64_image):
    payload = {
        "model": "llava:13b",
        "messages": [
            {
                "role": "user",
                "content": "describe the image as a doctor would in one short sentence which includes the imaging modality type and latin words for orientations and anatomical locations, X-ray, CT, MRI.",
                "images": [base64_image]
            }
        ],
        "stream": True
    }
    
    try:
        response = requests.post(llava_url, json=payload, stream=True)
        if response.status_code == 200:
            full_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode('utf-8'))
                        if 'message' in json_line and 'content' in json_line['message']:
                            full_content += json_line['message']['content']
                    except json.JSONDecodeError:
                        continue
            return full_content
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Request exception: {e}")
        return None

# Load only the test set
print("Loading dataset...")
dataset = load_dataset("eltorio/ROCOv2-radiology", split="test")  # Load only the test split

# Define LLaVA API URL
LLAVA_URL = "https://ollama.ux.uis.no/api/chat"

# Prepare BLEURT scorer
bleurt_scorer = bleurt_score.BleurtScorer("bleurt-20")

# Prepare to store results
results = []

# Create a wandb.Table to log examples
examples_table = wandb.Table(columns=["Image", "Reference Caption", "Generated Caption", "BLEU Score", "BERT-Score", "ROUGE-1", "BLEURT", "Image-Caption Similarity"])

# Process each image in the test set
print("Processing test set...")
for example in dataset:
    image = example["image"]
    caption = preprocess_caption(example["caption"])  # Preprocess reference caption
    image_id = example.get("image_id", "sample_image")

    print(f"Processing image ID: {image_id}")
    print(f"Original caption: {caption}")

    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Get generated caption from LLaVA
    generated_caption = process_image(image, LLAVA_URL, base64_image)
    if generated_caption:
        generated_caption = preprocess_caption(generated_caption)  # Preprocess generated caption

        # Calculate BLEU score with smoothing
        reference = [caption.split()]  # Tokenized reference
        candidate = generated_caption.split()  # Tokenized candidate
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)

        # Calculate BERT-Score (Recall)
        _, bert_recall, _ = bert_score([generated_caption], [caption], model_type="microsoft/deberta-xlarge-mnli", idf=True)

        # Calculate ROUGE-1 (F-measure)
        rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge_scores = rouge_scorer_instance.score(caption, generated_caption)
        rouge_1_fmeasure = rouge_scores["rouge1"].fmeasure

        # Calculate BLEURT score
        bleurt_score_value = bleurt_scorer.score(references=[caption], candidates=[generated_caption])[0]

        # Placeholder for Image-Caption Similarity (replace with actual embedding model)
        image_embedding = ...  # Compute image embedding
        caption_embedding = ...  # Compute caption embedding
        image_caption_similarity = compute_image_caption_similarity(image_embedding, caption_embedding)

        # Log to wandb
        wandb.log({
            "split": "test",
            "image_id": image_id,
            "reference_caption": caption,
            "generated_caption": generated_caption,
            "bleu_score": bleu_score,
            "bert_score": bert_recall.mean().item(),
            "rouge_1": rouge_1_fmeasure,
            "bleurt": bleurt_score_value,
            "image_caption_similarity": image_caption_similarity
        })

        # Add example to wandb.Table
        examples_table.add_data(
            wandb.Image(image),  # Log the image
            caption,  # Reference caption
            generated_caption,  # Generated caption
            bleu_score,  # BLEU score
            bert_recall.mean().item(),  # BERT-Score
            rouge_1_fmeasure,  # ROUGE-1
            bleurt_score_value,  # BLEURT
            image_caption_similarity  # Image-Caption Similarity
        )

        # Append to results
        results.append({
            "split": "test",
            "image_id": image_id,
            "reference_caption": caption,
            "generated_caption": generated_caption,
            "bleu_score": bleu_score,
            "bert_score": bert_recall.mean().item(),
            "rouge_1": rouge_1_fmeasure,
            "bleurt": bleurt_score_value,
            "image_caption_similarity": image_caption_similarity
        })

# Log the examples table to wandb
wandb.log({"Generated Captions Examples": examples_table})

# Save results to a JSON file
output_file = "llava_evaluation_results_test.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")

# Finish wandb run
wandb.finish()