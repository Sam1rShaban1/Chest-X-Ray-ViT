# --- Library Imports (Keep your existing ones) ---
import os
import site

# CRITICAL TPU ENVIRONMENT SETUP - MUST BE FIRST
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_USE_BF16"] = "1"  # Enable bfloat16 for TPU
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"  # Prevent memory issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TPU-specific environment variables
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_PJRT"] = "1"

# --- Explicitly set LD_LIBRARY_PATH for libtpu.so ---
libtpu_directory = "/home/ss31514/tpu_matched_env/lib/python3.10/site-packages/libtpu"

if os.path.exists(os.path.join(libtpu_directory, 'libtpu.so')):
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{libtpu_directory}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = libtpu_directory
    print(f"Main process: LD_LIBRARY_PATH set to: {os.environ['LD_LIBRARY_PATH']}")
else:
    print(f"Main process: WARNING: libtpu.so not found at expected path: {libtpu_directory}")

import io
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt # Not used in final training, but keeping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from google.cloud import storage
from tqdm import tqdm

# Hugging Face Transformers
from transformers import ViTForImageClassification, ViTImageProcessor
# ADD THESE IMPORTS FOR TRAINER
from transformers import Trainer, TrainingArguments
# For multi-label metrics with Trainer, we'll need this helper
# from evaluate import load # We'll use sklearn.metrics directly

print("All necessary libraries imported.")

# Check TPU availability (These lines are fine)
#print(f"TPU devices available: {xm.torch_xla.device_count()}")
#print(f"Current XLA device: {xm.xla_device()}")

# --- Configuration (Keep your existing ones) ---
GCP_PROJECT_ID = "affable-alpha-454813-t8"
GCS_BUCKET_NAME = "chest-xray-samir"
GCS_IMAGE_BASE_PREFIX = ""
GCS_BBOX_CSV_PATH = "BBox_List_2017.csv"
GCS_DATA_ENTRY_CSV_PATH = "Data_Entry_2017.csv"
GCS_TRAIN_VAL_LIST_PATH = "train_val_list.txt"
GCS_TEST_LIST_PATH = "test_list.txt"

OUTPUT_DIR = os.path.expanduser("~/vit_finetune_results/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ViT Model & Training Hyperparameters ---
MODEL_NAME = 'google/vit-base-patch16-384'
IMG_SIZE = 384
VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD = [0.229, 0.224, 0.225]

# Adjusted batch size for TPU
# BATCH_SIZE_PER_CORE is now per_device_train_batch_size in TrainingArguments
BATCH_SIZE_PER_CORE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 4
# NUM_WORKERS = 0 # Trainer usually handles this, or it can be set in DataLoaders. For XLA, 0 is often safe.

USE_SUBSET_DATA = None

print("Configuration set.")

# --- Global Variables (Keep these for now, as _mp_fn needs them as args) ---
bbox_df = None
data_entry_df = None
mlb = None
unique_labels_list = []
NUM_CLASSES = 0
gcs_blob_map_names = {}

# --- Load Metadata (Keep your existing code for metadata loading) ---
print("\n--- Loading Metadata ---")
try:
    _temp_storage_client = storage.Client(project=GCP_PROJECT_ID)
    _temp_bucket = _temp_storage_client.bucket(GCS_BUCKET_NAME)

    print(f"Downloading {GCS_BBOX_CSV_PATH}...")
    bbox_blob = _temp_bucket.blob(GCS_BBOX_CSV_PATH)
    csv_bytes = bbox_blob.download_as_bytes()
    bbox_df = pd.read_csv(io.BytesIO(csv_bytes))

    bbox_df.columns = bbox_df.columns.str.replace('[\[\]]', '', regex=True)
    bbox_df.columns = bbox_df.columns.str.replace(' ', '_', regex=False)
    bbox_df = bbox_df.loc[:, ~bbox_df.columns.str.contains('^Unnamed')]

    print("BBox_List_2017.csv loaded successfully.")
    print(f"BBox DataFrame shape: {bbox_df.shape}")

    bbox_dict = {}
    for index, row in bbox_df.iterrows():
        img_idx = row['Image_Index']
        bbox_info = {
            'label': row['Finding_Label'],
            'x': row['Bbox_x'],
            'y': row['y'],
            'w': row['w'],
            'h': row['h']
        }
        if img_idx not in bbox_dict:
            bbox_dict[img_idx] = []
        bbox_dict[img_idx].append(bbox_info)
    print(f"Created bbox_dict with {len(bbox_dict)} unique image entries.")

except Exception as e:
    print(f"ERROR: Failed to load BBox_List_2017.csv: {e}")
    bbox_df = None
    bbox_dict = {}

try:
    print(f"Downloading {GCS_DATA_ENTRY_CSV_PATH}...")
    data_entry_blob = _temp_bucket.blob(GCS_DATA_ENTRY_CSV_PATH)
    csv_bytes_data_entry = data_entry_blob.download_as_bytes()
    data_entry_df = pd.read_csv(io.BytesIO(csv_bytes_data_entry))

    data_entry_df['Finding Labels'] = data_entry_df['Finding Labels'].apply(
        lambda x: x.replace('No Finding', '').strip() if '|' in x else x
    )
    data_entry_df['Finding Labels'] = data_entry_df['Finding Labels'].apply(
        lambda x: 'No Finding' if not x else x
    )

    all_labels_str = "|".join(data_entry_df['Finding Labels'].tolist())
    unique_labels_list = sorted(list(set([lbl for lbl in all_labels_str.split('|') if lbl])))

    if 'No Finding' not in unique_labels_list:
        unique_labels_list.append('No Finding')
    unique_labels_list = sorted(unique_labels_list)

    mlb = MultiLabelBinarizer(classes=unique_labels_list)
    mlb.fit(data_entry_df['Finding Labels'].apply(lambda x: x.split('|')))
    NUM_CLASSES = len(unique_labels_list)

    print("Data_Entry_2017.csv loaded successfully.")
    print(f"Total unique labels: {unique_labels_list}")
    print(f"Number of classes: {NUM_CLASSES}")

except Exception as e:
    print(f"ERROR: Failed to load Data_Entry_2017.csv: {e}")
    data_entry_df = None
    mlb = None
    unique_labels_list = []
    NUM_CLASSES = 0

if NUM_CLASSES == 0:
    print("FATAL: NUM_CLASSES is 0. Exiting.")
    exit(1)

del _temp_storage_client
del _temp_bucket

print("\nMetadata loaded successfully.")

# --- Helper Functions (Keep your existing ones) ---
def pad_to_square(pil_img, padding_value=0):
    w, h = pil_img.size
    if w == h:
        return pil_img
    mode = pil_img.mode
    if w > h:
        new_img = Image.new(mode, (w, w), padding_value)
        new_img.paste(pil_img, (0, (w - h) // 2))
        return new_img
    else:
        new_img = Image.new(mode, (h, h), padding_value)
        new_img.paste(pil_img, ((h - w) // 2, 0))
        return new_img

def crop_and_pad_from_bbox(pil_img, bbox_coords, padding_value=0):
    x, y, w, h = int(bbox_coords['x']), int(bbox_coords['y']), int(bbox_coords['w']), int(bbox_coords['h'])
    img_w, img_h = pil_img.size
    left = max(0, x)
    upper = max(0, y)
    right = min(img_w, x + w)
    lower = min(img_h, y + h)

    if right <= left or lower <= upper or w <= 0 or h <= 0:
        return pad_to_square(pil_img, padding_value)

    cropped_pil = pil_img.crop((left, upper, right, lower))
    return pad_to_square(cropped_pil, padding_value)

roi_preprocess_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert('RGB')), # Ensure 3 channels for ViT
])

# --- Build GCS Image Path Map (Keep your existing ones) ---
print("\n--- Building GCS Image Path Map ---")
_temp_storage_client_map = storage.Client(project=GCP_PROJECT_ID)
_temp_bucket_map = _temp_storage_client_map.bucket(GCS_BUCKET_NAME)

image_subfolders = [f"images_{i:03}" for i in range(1, 13)]
base_img_prefix = GCS_IMAGE_BASE_PREFIX
if base_img_prefix and not base_img_prefix.endswith('/'):
    base_img_prefix += '/'

for subfolder in image_subfolders:
    current_prefix = f"{base_img_prefix}{subfolder}/images/"
    try:
        blobs_in_folder = list(_temp_bucket_map.list_blobs(prefix=current_prefix))
        for blob_obj in blobs_in_folder:
            if not blob_obj.name.endswith('/'):
                gcs_blob_map_names[os.path.basename(blob_obj.name)] = blob_obj.name
    except Exception as e:
        print(f"Warning: Error listing blobs from {current_prefix}: {e}")

print(f"Built GCS blob map with {len(gcs_blob_map_names)} unique image filenames.")
del _temp_storage_client_map
del _temp_bucket_map

# --- Dataset Class (Keep, but modify to return dict expected by Trainer) ---
class NIHChestDataset(Dataset):
    def __init__(self, df, image_filenames_list, bbox_dict, label_binarizer, transform=None, image_processor=None, gcs_blob_map_names=None, use_subset=None):
        self.transform = transform
        self.image_processor = image_processor
        self.bbox_dict = bbox_dict
        self.label_binarizer = label_binarizer
        self.gcs_blob_names_for_dataset = gcs_blob_map_names

        self.df_filtered = df[df['Image Index'].isin(image_filenames_list)].copy()
        self.df_filtered.set_index('Image Index', inplace=True)
        self.image_filenames = self.df_filtered.index.tolist()

        if use_subset:
            self.image_filenames = self.image_filenames[:use_subset]

        self.encoded_labels = self.label_binarizer.transform(
            self.df_filtered.loc[self.image_filenames, 'Finding Labels'].apply(lambda x: x.split('|'))
        )
        print(f"Dataset initialized with {len(self.image_filenames)} images.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        label_vector = torch.FloatTensor(self.encoded_labels[idx])

        blob_name_to_download = self.gcs_blob_names_for_dataset.get(img_name)

        # Create GCS client for this worker (within __getitem__ is fine for XLA/multi-process)
        worker_storage_client = storage.Client(project=GCP_PROJECT_ID)
        worker_bucket = worker_storage_client.bucket(GCS_BUCKET_NAME)

        if blob_name_to_download:
            try:
                worker_blob = worker_bucket.blob(blob_name_to_download)
                image_bytes = worker_blob.download_as_bytes()
                original_pil = Image.open(io.BytesIO(image_bytes)).convert('L')
                del image_bytes
            except Exception as e:
                print(f"Warning: Could not download image {img_name}: {e}. Returning dummy image.")
                original_pil = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
        else:
            print(f"Warning: Image {img_name} not found in GCS map. Returning dummy image.")
            original_pil = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)

        # Process bounding box
        if img_name in self.bbox_dict and self.bbox_dict[img_name]:
            bbox_coords = self.bbox_dict[img_name][0] # Assuming one bbox per image or taking the first
            cropped_padded_pil_image = crop_and_pad_from_bbox(original_pil, bbox_coords, padding_value=0)
        else:
            cropped_padded_pil_image = pad_to_square(original_pil, padding_value=0)

        if self.transform:
            cropped_padded_pil_image = self.transform(cropped_padded_pil_image)

        if self.image_processor:
            # Use image_processor on the PIL image
            processed_output = self.image_processor(images=cropped_padded_pil_image, return_tensors="pt")
            image_tensor = processed_output.pixel_values.squeeze(0) # Remove batch dimension added by processor
        else:
            image_tensor = cropped_padded_pil_image

        return {'pixel_values': image_tensor, 'labels': label_vector}


# --- NEW: Data Collator for Trainer ---
# This is directly from the blog post, adapted slightly for our outputs
def collate_fn(batch):
    # 'pixel_values' are already tensors from the image_processor
    # 'labels' are already tensors from the dataset
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]) # Use stack for multi-label
    }

# --- NEW: Compute Metrics for Trainer (AUROC) ---
# This will be defined within _mp_fn to access unique_labels_list via closure
def compute_metrics_fn(eval_pred, unique_labels_list):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # Convert logits to probabilities using sigmoid for multi-label classification
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels_np = labels # Already numpy array from Trainer

    # Calculate AUROC per class
    auroc_per_class = {}
    valid_classes = 0
    total_auroc = 0

    for i, label_name in enumerate(unique_labels_list):
        try:
            # roc_auc_score requires at least two unique class values
            if len(np.unique(labels_np[:, i])) > 1:
                class_roc_auc = roc_auc_score(labels_np[:, i], probs[:, i])
                auroc_per_class[label_name] = class_roc_auc
                total_auroc += class_roc_auc
                valid_classes += 1
            else:
                # If only one class is present, AUROC is undefined. Assign NaN or skip.
                auroc_per_class[label_name] = np.nan # Or print a warning and skip
        except ValueError as e:
            # Catch cases where roc_auc_score might fail (e.g., all predictions are same)
            auroc_per_class[label_name] = np.nan
            # print(f"Warning: Could not calculate AUROC for class {label_name}: {e}")

    avg_auroc = total_auroc / valid_classes if valid_classes > 0 else 0.0

    # Trainer expects a dictionary of metrics
    metrics = {"avg_auroc": avg_auroc}
    # Optionally, include per-class AUROC for more detailed logging
    # for k, v in auroc_per_class.items():
    #     if not np.isnan(v):
    #         metrics[f"auroc_{k}"] = v
    return metrics


# --- Main Training Function (REFRACTORED to use Trainer) ---
def _mp_fn(rank, data_entry_df, bbox_dict, mlb, gcs_blob_map_names, unique_labels_list):
    # Trainer handles device assignment automatically
    # device = xm.xla_device() # No longer needed explicitly here for device assignment
    print(f"Process {rank}: Starting on XLA device.")

    # Initialize image processor locally
    try:
        local_image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        print(f"Process {rank}: ViTImageProcessor initialized.")
    except Exception as e:
        print(f"Process {rank}: Error initializing image processor: {e}")
        return

    # Initialize model
    model = None # Initialize model to None for safety

    try:
        print(f"Process {rank}: Attempting to load model from {MODEL_NAME}...")

        # Strategy: Load on CPU first, then let Trainer handle moving to XLA device.
        # This is the most robust approach for older transformers/torch_xla versions
        # that struggle with direct-to-device loading or meta tensors.
        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True, # Keeps classifier.weight/bias warnings, which is fine
            id2label={i: c for i, c in enumerate(unique_labels_list)},
            label2id={c: i for i, c in enumerate(unique_labels_list)}

           # Do NOT pass 'device' or 'low_cpu_mem_usage' here for this specific strategy
        )
        print(f"Process {rank}: Model loaded onto CPU. Trainer will move it to XLA device.")

    except Exception as e:
        print(f"Process {rank}: FATAL ERROR during model loading: {e}")
        if model is None:
            print(f"Process {rank}: Model variable is still None after from_pretrained attempt. This indicates from_pretrained itself failed.")
        return # Ensure the process exits if model loading fails

    if model is None or not isinstance(model, torch.nn.Module):
        print(f"Process {rank}: Critical: Model is None or not a valid PyTorch module after loading block. Exiting process.")
        return # Explicitly exit if model is still None or invalid

    # No need for manual optimizer/criterion here, Trainer handles it
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Load train/val/test splits
    worker_storage_client = storage.Client(project=GCP_PROJECT_ID)
    worker_bucket = worker_storage_client.bucket(GCS_BUCKET_NAME)

    train_val_list_blob = worker_bucket.blob(GCS_TRAIN_VAL_LIST_PATH)
    train_val_files = train_val_list_blob.download_as_bytes().decode('utf-8').splitlines()
    test_list_blob = worker_bucket.blob(GCS_TEST_LIST_PATH)
    test_files = test_list_blob.download_as_bytes().decode('utf-8').splitlines()

    train_files_final, val_files_final = train_test_split(train_val_files, test_size=0.15, random_state=42)

    # Create datasets
    train_dataset = NIHChestDataset(
        data_entry_df, train_files_final, bbox_dict, mlb,
        transform=roi_preprocess_transforms,
        image_processor=local_image_processor,
        gcs_blob_map_names=gcs_blob_map_names,
        use_subset=USE_SUBSET_DATA
    )

    val_dataset = NIHChestDataset(
        data_entry_df, val_files_final, bbox_dict, mlb,
        transform=roi_preprocess_transforms,
        image_processor=local_image_processor,
        gcs_blob_map_names=gcs_blob_map_names,
        use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None
    )

    print(f"Process {rank}: Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")

    # Define TrainingArguments - mostly from blog post, adjusted for your needs
    # Trainer handles distributed batching and sampling
    training_args = TrainingArguments(
      output_dir=os.path.join(OUTPUT_DIR, f"vit-finetune-chest-xray-rank{rank}"), # Unique output dir per rank for safety
      per_device_train_batch_size=BATCH_SIZE_PER_CORE,
      per_device_eval_batch_size=BATCH_SIZE_PER_CORE, # Add eval batch size
      eval_strategy="steps",
      num_train_epochs=NUM_EPOCHS,
      bf16=True, # Enable mixed precision
      save_steps=500, # Increased save frequency for larger datasets
      eval_steps=500, # Increased eval frequency
      logging_steps=50, # Log more frequently
      learning_rate=LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
      save_total_limit=2,
      remove_unused_columns=False, # Crucial: tells Trainer not to drop image column
      push_to_hub=False, # Set to True if you want to push to HF Hub
      report_to='tensorboard', # Recommended for logging
      load_best_model_at_end=True,
      metric_for_best_model="avg_auroc", # Specify metric to track for best model
      greater_is_better=True, # For AUROC, higher is better
      # You might also want to set `dataloader_num_workers` here if not 0
      # dataloader_num_workers=NUM_WORKERS, # Set to 0 for TPU usually
    )

    # Create a closure for compute_metrics to pass unique_labels_list
    _compute_metrics = lambda eval_pred: compute_metrics_fn(eval_pred, unique_labels_list)

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn, # Use our custom collate_fn
        compute_metrics=_compute_metrics, # Use our adapted compute_metrics
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=local_image_processor, # Pass processor here for saving config etc.
    )

    print(f"Process {rank}: Starting training with Trainer...")

    # Train
    train_results = trainer.train()
    # Trainer automatically saves the best model if load_best_model_at_end is True

    if rank == 0: # Only save logs/metrics from rank 0
        trainer.save_model() # Saves the model (including best if configured)
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        print(f"Process {rank}: Training completed and model/logs saved.")

        # Evaluate final model on validation set
        metrics = trainer.evaluate(val_dataset) # Evaluate on the actual val_dataset
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"Process {rank}: Final evaluation metrics: {metrics}")

    # No need for xm.mark_step() or manual model saving inside epoch loop with Trainer

# --- Main Execution (Keep your existing one) ---
if __name__ == '__main__':
    if data_entry_df is None or mlb is None or not gcs_blob_map_names:
        print("ERROR: Metadata not loaded properly. Cannot proceed.")
        exit(1)

    try:
        print("Main process: Pre-downloading/caching Hugging Face model to ensure all processes have it locally...")
        # Call from_pretrained once in the main process to cache the model.
        # num_labels doesn't matter for caching purposes.
        _ = ViTForImageClassification.from_pretrained(MODEL_NAME)
        _ = ViTImageProcessor.from_pretrained(MODEL_NAME) # Also pre-cache the processor
        print("Main process: Model and processor pre-download complete.")
    except Exception as e:
        print(f"Main process: WARNING: Failed to pre-download model or processor: {e}")
        print("Main process: Child processes might download concurrently, which could cause issues.")

    print(f"Starting training on TPU cores via xmp.spawn...")
    # xmp.spawn will run _mp_fn on each TPU core
    xmp.spawn(_mp_fn, args=(data_entry_df, bbox_dict, mlb, gcs_blob_map_names, unique_labels_list), nprocs=None)
