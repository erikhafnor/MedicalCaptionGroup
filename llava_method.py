import os
import io
import json
import requests
import base64
from PIL import Image
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import wandb

# Initialize wandb
wandb.init(project="llava-medical-image-evaluation", name="llava-bleu-evaluation")

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

# Prepare to store results
results = []

# Create a wandb.Table to log examples
examples_table = wandb.Table(columns=["Image", "Reference Caption", "Generated Caption", "BLEU Score"])

# Process each image in the test set
print("Processing test set...")
for example in dataset:
    image = example["image"]
    caption = example["caption"]
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
        # Calculate BLEU score
        reference = [caption.split()]  # Tokenized reference
        candidate = generated_caption.split()  # Tokenized candidate
        bleu_score = sentence_bleu(reference, candidate)

        # Log to wandb
        wandb.log({
            "split": "test",
            "image_id": image_id,
            "reference_caption": caption,
            "generated_caption": generated_caption,
            "bleu_score": bleu_score
        })

        # Add example to wandb.Table
        examples_table.add_data(
            wandb.Image(image),  # Log the image
            caption,  # Reference caption
            generated_caption,  # Generated caption
            bleu_score  # BLEU score
        )

        # Append to results
        results.append({
            "split": "test",
            "image_id": image_id,
            "reference_caption": caption,
            "generated_caption": generated_caption,
            "bleu_score": bleu_score
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