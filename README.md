
# 🩺 Chest X-Ray Classification using Vision Transformers (ViT)

This project fine-tunes a pre-trained Vision Transformer (ViT) on the NIH Chest X-ray dataset to perform multi-label classification of thoracic diseases. It uses Google Cloud TPUs for acceleration and integrates with Google Cloud Storage for dataset access and model checkpointing.

---

## 🧠 Model Architecture

- **Model**: `google/vit-base-patch16-384`
  - Pre-trained on ImageNet-1k
  - Input Resolution: `384x384`
- **Normalization**
  - `mean`: `[0.485, 0.456, 0.406]`
  - `std`: `[0.229, 0.224, 0.225]`

---

## ⚙️ Hyperparameters

| Parameter            | Value     |
|----------------------|-----------|
| `IMG_SIZE`           | 384       |
| `BATCH_SIZE_PER_CORE`| 16        |
| `LEARNING_RATE`      | 2e-4      |
| `WEIGHT_DECAY`       | 0.01      |
| `NUM_EPOCHS`         | 4         |
| `NUM_WORKERS`        | 8         |
| `USE_SUBSET_DATA`    | None (use full dataset) |

---

## 📁 Project Structure

```
.
├── ViT-Training.py        # Main training script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/                  # (Optional) Local data folder if not using GCS
└── outputs/               # Saved models and logs
```

---

## ☁️ Cloud TPU & GCS Setup

To train on a TPU VM with access to NIH Chest X-rays on Google Cloud:

1. **Upload NIH dataset to Google Cloud Storage**:
   ```bash
   gsutil -m cp -r ./data gs://<your-bucket-name>/nih-chest-xray
   ```

2. **Launch TPU VM**:
   ```bash
   gcloud compute tpus tpu-vm create vit-tpu        --zone=us-central1-b        --accelerator-type=v3-8        --version=tpu-ubuntu2204-base
   ```

3. **SSH into TPU VM**:
   ```bash
   gcloud compute tpus tpu-vm ssh vit-tpu --zone=us-central1-b
   ```

4. **Clone repo & install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip -y
   pip install -r requirements.txt
   ```

5. **Run training**:
   ```bash
   python ViT-Training.py
   ```

---

## 📝 Notes

- Training uses **multi-label classification** (e.g., a single image can have multiple diseases).
- For faster testing, set `USE_SUBSET_DATA = 1000` in the script.
- Save logs and checkpoints automatically in `./outputs/`.

---

## ✅ TODO

- [ ] Add evaluation metrics (AUC, F1)
- [ ] Visualize attention maps (Grad-CAM)
- [ ] Export trained model for inference
