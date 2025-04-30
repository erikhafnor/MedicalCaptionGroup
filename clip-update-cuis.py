import pandas as pd
from datasets import load_dataset, DatasetDict, Features, Sequence, Value
import os
from tqdm.auto import tqdm # For progress bars

# --- Configuration ---
DATASET_NAME = "eltorio/ROCOv2-radiology"
CSV_FILES = {
    "train": "train_concepts.csv",
    "validation": "valid_concepts.csv",
    "test": "test_concepts.csv",
}
OUTPUT_DIR = "roco_updated_cuis" # Directory to save the updated dataset

# --- Helper Function to Parse CUIs ---
def parse_cuis(cui_string):
    """Splits a semicolon-separated string into a list of CUIs, handling NaNs."""
    if pd.isna(cui_string) or not cui_string:
        return [] # Return empty list for missing or empty CUI strings
    # Split and filter out any potential empty strings resulting from split
    return [cui.strip() for cui in cui_string.split(';') if cui.strip()]

# --- 1. Load the Original Dataset ---
print(f"Loading original dataset: {DATASET_NAME}...")
try:
    original_ds = load_dataset(DATASET_NAME)
    print("Original dataset loaded successfully.")
    print("Original dataset structure:")
    print(original_ds)
    # Optional: Inspect original features, especially 'cui'
    # print("\nOriginal features:")
    # print(original_ds['train'].features)

except Exception as e:
    print(f"Error loading dataset {DATASET_NAME}: {e}")
    exit()

# --- 2. Load CUI Data from CSVs ---
print("\nLoading CUI data from CSV files...")
cui_dataframes = {}
try:
    for split, filename in CSV_FILES.items():
        if not os.path.exists(filename):
             raise FileNotFoundError(f"CSV file not found: {filename}. Please ensure it's in the correct directory.")
        # Read CSV, using 'ID' as index for faster lookups
        df = pd.read_csv(filename, index_col='ID')
        cui_dataframes[split] = df
        print(f"Loaded {filename} for split '{split}'.")
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit()

# --- 3. Update the Dataset Splits ---
print("\nUpdating dataset splits with new CUIs...")
updated_splits = {}

# Define the expected features for the updated dataset
original_features = original_ds['train'].features.copy()
# Update the 'cui' feature type
original_features['cui'] = Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
updated_features = Features(original_features)


for split_name in original_ds.keys():
    print(f"\nProcessing split: '{split_name}'...")
    if split_name not in cui_dataframes:
        print(f"Warning: No CUI data found for split '{split_name}'. Skipping update for this split.")
        updated_splits[split_name] = original_ds[split_name]
        continue

    current_split_ds = original_ds[split_name]
    cui_df = cui_dataframes[split_name]

    # Function to apply with .map()
    def update_cui_batch(batch):
        new_cuis = []
        missing_ids = []
        ids_in_batch = batch['image_id'] # Get all IDs in the current batch

        # Attempt efficient lookup using the DataFrame index
        cui_series = cui_df.loc[ids_in_batch, 'CUIs']
        # Need to handle cases where some IDs might be missing in the lookup result
        for id_val in ids_in_batch:
            try:
                cui_string = cui_series.at[id_val]
                new_cuis.append(parse_cuis(cui_string))
            except KeyError:
                # ID from dataset batch not found in the CSV index
                missing_ids.append(id_val)
                new_cuis.append([]) # Append empty list for missing IDs


        if missing_ids:
            print(f"Warning: IDs from dataset split '{split_name}' not found in {CSV_FILES[split_name]}: {missing_ids[:5]}... (showing first 5)")

        # Return dictionary with the column to update
        return {'cui': new_cuis}

    updated_splits[split_name] = current_split_ds.map(
        update_cui_batch,
        batched=True,
        features=updated_features, # Apply the defined features
        desc=f"Updating CUIs for '{split_name}' split" # Progress bar description
    )
    print(f"Split '{split_name}' updated.")


# --- 4. Create the Final DatasetDict ---
final_ds = DatasetDict(updated_splits)
print("\nFinal dataset structure:")
print(final_ds)
print("\nSample CUI data from updated 'train' split:")
if 'train' in final_ds:
    print(final_ds['train'][0]['image_id'], final_ds['train'][0]['cui'])
    print(final_ds['train'][1]['image_id'], final_ds['train'][1]['cui'])


# --- 5. Save the Updated Dataset ---
print(f"\nSaving updated dataset to disk: '{OUTPUT_DIR}'...")
try:
    final_ds.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved successfully to '{OUTPUT_DIR}'.")
    print("\nYou can load it later using:")
    print(f"from datasets import load_from_disk")
    print(f"loaded_ds = load_from_disk('{OUTPUT_DIR}')")
except Exception as e:
    print(f"Error saving dataset to disk: {e}")

print("\nScript finished.")