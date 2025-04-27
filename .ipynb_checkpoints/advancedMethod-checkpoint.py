import os
import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from PIL import Image
import wandb

# Initialize W&B
wandb.init(project="medical-image-captioning", name="clip-fine-tuning5")

# Step 1: Load the dataset
ds = load_dataset("eltorio/ROCOv2-radiology")

# Step 2: Preprocess the dataset
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_data(example):
    try:
        # Use the image directly as it's already a PIL.Image object
        image = example["image"].convert("RGB")  # Ensure it's in RGB format
        
        caption = example["caption"]  # Extract the caption
        inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True)
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # Remove batch dimension
            "input_ids": inputs["input_ids"].squeeze(0)  # Remove batch dimension
        }
    except Exception as e:
        print(f"Error processing example: {e}")
        return None  # Return None for invalid examples

# Apply preprocessing to each subset
train_ds = ds["train"].map(preprocess_data, remove_columns=ds["train"].column_names)
val_ds = ds["validation"].map(preprocess_data, remove_columns=ds["validation"].column_names)

# Filter out invalid examples
train_ds = train_ds.filter(lambda x: x is not None and "pixel_values" in x and "input_ids" in x)
val_ds = val_ds.filter(lambda x: x is not None and "pixel_values" in x and "input_ids" in x)

# Step 3: Create DataLoaders
def collate_fn(batch):
    try:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    except KeyError as e:
        print(f"KeyError in batch: {e}")
        print(f"Batch: {batch}")
        raise

train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)

# Step 4: Load the pretrained CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Step 5: Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: Fine-tune the model with validation monitoring
model.train()
for epoch in range(3):  # Number of epochs
    train_loss = 0.0
    val_loss = 0.0

    # Training loop
    for batch in train_dataloader:
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss  # CLIP provides a contrastive loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            val_loss += outputs.loss.item()
    model.train()

    # Calculate average losses
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)

    # Log losses to W&B
    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

# Step 7: Save the fine-tuned model
model.save_pretrained("./fine_tuned_clip")
processor.save_pretrained("./fine_tuned_clip")
wandb.save("./fine_tuned_clip/*")

# Step 8: Finish W&B run
wandb.finish()