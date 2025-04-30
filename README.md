# 📚 Medical Image Captioning - Project

This project aims to generate captions for medical images using various models including CNN+Transformer, BLIP, and evaluation metrics like BERTScore and MedBERTScore.

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MedicalCaptionGroup
```

> **Important**: Please make sure you are on the `development` branch to access the latest updates.

---

### 2. Install Dependencies

We recommend using Python 3.10 or higher.  
First, install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, you can manually install key packages:

```bash
pip install torch torchvision transformers datasets tqdm bert_score evaluate rouge_score
```

---

### 3. Prepare Data

The datasets used (ROCOv2) must be downloaded manually if not already available.


> ⚡ **Note**: Large files like models and outputs are **ignored** in GitHub using `.gitignore` rules. You need to download or generate them locally if needed.

---

### 4. Running the Notebooks

Each major part of the project has its own Python script or Jupyter Notebook:

- `preprocess_extract_features.py` → Data preparation and feature extraction
- `sentenceEmbedRandomForest.py` → Baseline method with Sentence Embedding + K-Means clustering and RandomForest Classification with evaluations on dataset split test
- `llavaMethod2.py` → Advanced method with Ollama LLaVa with evaluations on dataset split test
- `reportGeneration.ipynb` → Get statistics from evaluations on dataset split test for baseline method and LLaVa method
- `blip_captioning.ipynb` → Fine-tuning and evaluating BLIP model
- `cnn_transformer_captioning.ipynb` → CNN feature extraction + Transformer decoder captioning
- `evaluation.ipynb` → Evaluation of results using BERTScore, MedBERTScore, ROUGE

You can open them using:

```bash
jupyter notebook
```

Or inside VS Code or Jupyter Lab.

---

### 5. Evaluation Metrics

We used:

- **BERTScore**: General language caption quality
- **MedBERTScore**: Specialized for medical domain
- **BLEU**: Focus on Precision (n-gram overlap)
- **ROUGE-1**: Recall (unigram overlap)
- **BLEURT**: Semantic similarity

Final scores are available in:

- `evaluation_results.txt`

---

⚠️ Important Notes
-Model checkpoint files (like .safetensors) are large and not included in the repository.

-Please download the model manually or train it if you prefer.

-Make sure you have enough GPU memory (recommended: 8GB+) for faster training and inference.

- Model Checkpoint Download
You can download the pretrained model checkpoint from the following link:
[Download model.safetensors](https://drive.google.com/file/d/16tVF6YRXLbiMFVj1uVrwK8dGPBUS2fSp/view?usp=drive_link)

After downloading, please place the file next to your BLIP-related code files, or update the code path accordingly if needed.



# 📄 License

This project is for educational purposes at University of Stavanger (UiS) - Data Mining Course 2025.
