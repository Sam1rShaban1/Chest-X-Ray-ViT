# Chest X-Ray Classification using Vision Transformers (ViT)

This project fine-tunes a pre-trained Vision Transformer (ViT) model to classify chest X-ray images from the NIH Chest X-ray dataset using PyTorch. The training is performed on TPUs with data loading from Google Cloud Storage (GCS).

## 📁 Project Structure

```
.
├── ViT-Training.py              # Main training script
├── requirements.txt             # Dependencies for the project
├── utils/
│   └── data_utils.py            # Custom dataset, preprocessing, and data loaders
├── config.yaml (optional)      # For hyperparameters and environment configs
└── README.md                    # Project overview and instructions
```

## 🚀 Features

- ✅ Fine-tuning HuggingFace ViT model (`google/vit-base-patch16-384`)
- ✅ TPU training with `torch_xla`
- ✅ Efficient GCS integration with `gcsfs`
- ✅ Multiclass classification with NIH Chest X-ray labels
- ✅ Mixed precision support
- ✅ AUROC metric tracking

---

## 📦 Installation

### 1. Clone this repo

```bash
git clone https://github.com/Sam1rShaban1/Chest-X-Ray-ViT.git
cd Chest-X-Ray-ViT
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

For TPU:
```bash
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

---

## ☁️ Google Cloud Setup

### 1. Upload your NIH dataset to a GCS bucket

Example structure:
```
gs://your-bucket/chest-xray-data/images/
gs://your-bucket/chest-xray-data/Data_Entry_2017.csv
```

### 2. Set your Google Cloud credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"
```

---

## 🧠 Training the Model

Run training on a TPU:

```bash
python ViT-Training.py \
  --bucket_name your-bucket \
  --data_dir chest-xray-data \
  --batch_size 32 \
  --epochs 10 \
  --lr 3e-5
```

### Optional Flags

- `--image_size`: Resize input image (default: 384)
- `--precision`: Enable mixed precision (default: False)
- `--tpu`: Enable TPU training (default: True)

---

## 📊 Evaluation

- The script logs AUROC per label on the validation and test sets.
- You can modify the script to save the best model weights to either GCS or local disk.

---

## 📝 NIH Labels

Supported 15 chest pathology labels:
```
['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
```

Multi-label one-hot vectors are used as ground truth.

---

## 📈 Results

You can expect AUROC values per label. Save results using:

```python
torch.save(model.state_dict(), "vit_chestxray_best.pth")
```

or to GCS:

```python
fs = gcsfs.GCSFileSystem()
with fs.open('gs://your-bucket/models/vit_chestxray_best.pth', 'wb') as f:
    torch.save(model.state_dict(), f)
```

---

## 🛠️ TODO

- [ ] Add confusion matrix visualization
- [ ] Implement learning rate scheduler
- [ ] Add TensorBoard/XLA metrics
- [ ] Improve image augmentations

---

## 📚 References

- [NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/data)
- [Hugging Face Transformers - ViT](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [PyTorch/XLA TPU Docs](https://pytorch.org/xla/)

---

## 👨‍⚕️ Authors

- **Samir Shabani** 
