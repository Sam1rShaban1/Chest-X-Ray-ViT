# --- Library Imports ---
import os
import site

# CRITICAL TPU ENVIRONMENT SETUP - MUST BE FIRST
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_USE_BF16"] = "1"  # This is NOT used by Trainer for TPUs, but OK as an XLA env var
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000000"  # 100GB - Good setting
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
import matplotlib.pyplot as plt
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
from scipy.special import expit

# Hugging Face Transformers
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Trainer, TrainingArguments

# TPU-specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

print("All necessary libraries imported.")

# --- Configuration ---
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

BATCH_SIZE_PER_CORE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2

USE_SUBSET_DATA = 500 # Ensure this subset is small enough for testing

print("Configuration set.")

# --- Global Variables (will be populated in main, then passed) ---
# It's cleaner to remove these global = None lines, but leaving them commented for clarity.
# bbox_df = None
# data_entry_df = None
# mlb = None
# unique_labels_list = []
# NUM_CLASSES = 0
# gcs_blob_map_names = {}


# --- New Function to Load All Global Metadata in Main Process ---
def load_global_metadata(project_id, bucket_name, bbox_csv_path, data_entry_csv_path, image_base_prefix, gcs_image_subfolders):
    # Initialize GCS client and bucket for this one-time loading in the main process
    _storage_client = storage.Client(project=project_id)
    _bucket = _storage_client.bucket(bucket_name)

    # --- Load BBox Data ---
    print(f"Main process: Downloading {bbox_csv_path}...")
    bbox_blob = _bucket.blob(bbox_csv_path)
    csv_bytes = bbox_blob.download_as_bytes()
    bbox_df = pd.read_csv(io.BytesIO(csv_bytes))
    bbox_df.columns = bbox_df.columns.str.replace('[\[\]]', '', regex=True)
    bbox_df.columns = bbox_df.columns.str.replace(' ', '_', regex=False)
    bbox_df = bbox_df.loc[:, ~bbox_df.columns.str.contains('^Unnamed')]
    print("Main process: BBox_List_2017.csv loaded successfully.")

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
    print(f"Main process: Created bbox_dict with {len(bbox_dict)} unique image entries.")
    
    # --- Load Data Entry Data ---
    print(f"Main process: Downloading {data_entry_csv_path}...")
    data_entry_blob = _bucket.blob(data_entry_csv_path)
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
    num_classes = len(unique_labels_list)
    print("Main process: Data_Entry_2017.csv loaded successfully.")
    print(f"Main process: Total unique labels: {unique_labels_list}")
    print(f"Main process: Number of classes: {num_classes}")

    if num_classes == 0:
        print("Main process: FATAL: NUM_CLASSES is 0. Exiting.")
        exit(1)

    # --- Build GCS Image Path Map ---
    print("\n--- Main process: Building GCS Image Path Map ---")
    gcs_blob_map_names = {}
    base_img_prefix_local = image_base_prefix # Use local copy
    if base_img_prefix_local and not base_img_prefix_local.endswith('/'):
        base_img_prefix_local += '/'

    for subfolder in gcs_image_subfolders: # Use passed list of subfolders
        current_prefix = f"{base_img_prefix_local}{subfolder}/images/"
        try:
            blobs_in_folder = list(_bucket.list_blobs(prefix=current_prefix))
            for blob_obj in blobs_in_folder:
                if not blob_obj.name.endswith('/'):
                    gcs_blob_map_names[os.path.basename(blob_obj.name)] = blob_obj.name
        except Exception as e:
            print(f"Main process: Warning: Error listing blobs from {current_prefix}: {e}")

    print(f"Main process: Built GCS blob map with {len(gcs_blob_map_names)} unique image filenames.")

    # Return all the data needed by child processes
    return bbox_df, bbox_dict, data_entry_df, mlb, unique_labels_list, num_classes, gcs_blob_map_names


# --- Helper Functions (keep as is) ---
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
    transforms.Lambda(lambda img: img.convert('RGB')),
])


# --- Dataset Class (Updated to use passed dataframes) ---
class NIHChestDataset(Dataset):
    def __init__(self, df_for_dataset, image_filenames_list, bbox_dict_global, label_binarizer_global, transform=None, image_processor=None, gcs_blob_map_names_global=None, use_subset=None, gcs_client=None, gcs_bucket=None):
        self.transform = transform
        self.image_processor = image_processor
        self.bbox_dict = bbox_dict_global # Use the global dict passed
        self.label_binarizer = label_binarizer_global # Use the global mlb passed
        self.gcs_blob_names_for_dataset = gcs_blob_map_names_global # Use the global map passed
        self.gcs_client = gcs_client
        self.gcs_bucket = gcs_bucket

        # CRITICAL: df_for_dataset is assumed to be already filtered for this dataset's specific files
        self.df_filtered = df_for_dataset # Pass the already filtered DataFrame slice
        self.df_filtered.set_index('Image Index', inplace=True) # Ensure index is set
        self.image_filenames = image_filenames_list # This list should be the actual subset for this dataset

        if use_subset: # This subsetting should ideally happen before passing image_filenames_list
            self.image_filenames = self.image_filenames[:use_subset] # Keep for safety, but check caller

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

        worker_bucket = self.gcs_bucket # CRITICAL: uses stored bucket

        if blob_name_to_download:
            try:
                worker_blob = worker_bucket.blob(blob_name_to_download)
                image_bytes = worker_blob.download_as_bytes()
                original_pil = Image.open(io.BytesIO(image_bytes)).convert('L')
                del image_bytes
            except Exception as e:
                # Add process PID to warnings for clarity
                print(f"Process {os.getpid()}: Warning: Could not download image {img_name}: {e}. Returning dummy image.")
                original_pil = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
        else:
            print(f"Process {os.getpid()}: Warning: Image {img_name} not found in GCS map. Returning dummy image.")
            original_pil = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)

        if img_name in self.bbox_dict and self.bbox_dict[img_name]:
            bbox_coords = self.bbox_dict[img_name][0]
            cropped_padded_pil_image = crop_and_pad_from_bbox(original_pil, bbox_coords, padding_value=0)
        else:
            cropped_padded_pil_image = pad_to_square(original_pil, padding_value=0)

        if self.transform:
            cropped_padded_pil_image = self.transform(cropped_padded_pil_image)

        if self.image_processor:
            processed_output = self.image_processor(images=cropped_padded_pil_image, return_tensors="pt")
            image_tensor = processed_output.pixel_values.squeeze(0)
        else:
            image_tensor = cropped_padded_pil_image

        return {'pixel_values': image_tensor, 'labels': label_vector}


# --- Data Collator for Trainer ---
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

# --- Compute Metrics for Trainer (AUROC) ---
def compute_metrics_fn(eval_pred, unique_labels_list): # unique_labels_list is passed via closure
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # OPTIMIZED LINE: Use scipy's expit for element-wise sigmoid on NumPy array
    probs = expit(logits) 
    
    labels_np = labels # eval_pred.label_ids is already a NumPy array

    auroc_per_class = {}
    valid_classes = 0
    total_auroc = 0

    for i, label_name in enumerate(unique_labels_list):
        try:
            # Ensure there are at least two unique actual labels (binary classification requirement for roc_auc_score)
            # and at least two samples in the array to avoid ValueError.
            if len(np.unique(labels_np[:, i])) > 1 and labels_np.shape[0] >= 2:
                class_roc_auc = roc_auc_score(labels_np[:, i], probs[:, i])
                auroc_per_class[label_name] = class_roc_auc
                total_auroc += class_roc_auc
                valid_classes += 1
            else:
                # If only one class is present or not enough samples, AUC is undefined.
                auroc_per_class[label_name] = np.nan
        except ValueError as e:
            # Catch specific ValueError if roc_auc_score encounters an issue despite checks
            # (e.g., all predictions are identical for a class, which can happen with small batches)
            auroc_per_class[label_name] = np.nan

    avg_auroc = total_auroc / valid_classes if valid_classes > 0 else 0.0
    metrics = {"avg_auroc": avg_auroc}
    return metrics


# --- Main Training Function for TPU multiprocessing via xmp.spawn (MODIFIED TO ACCEPT SHARED METADATA) ---
def _mp_fn(index, shared_metadata): # 'index' is rank, 'shared_metadata' is the tuple
    """Main training function for TPU multiprocessing"""

    # Unpack shared metadata received from the main process
    (global_bbox_df, global_bbox_dict, global_data_entry_df, global_mlb,
     global_unique_labels_list, global_num_classes, global_gcs_blob_map_names) = shared_metadata

    # Initialize TPU device and get rank/world_size
    device = xm.xla_device()
    rank = index
    world_size = xr.addressable_device_count()
    
    print(f"Process {rank}: Starting training on device {device} (World Size: {world_size})")

    # Initialize image processor locally
    try:
        local_image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        print(f"Process {rank}: ViTImageProcessor initialized.")
    except Exception as e:
        print(f"Process {rank}: Error initializing image processor: {e}")
        return

    # Initialize model
    model = None

    try:
        print(f"Process {rank}: Attempting to load model from {MODEL_NAME}...")

        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=global_num_classes, # Use num_classes from shared_metadata
            ignore_mismatched_sizes=True,
            id2label={i: c for i, c in enumerate(global_unique_labels_list)}, # Use from shared_metadata
            label2id={c: i for i, c in enumerate(global_unique_labels_list)}, # Use from shared_metadata
            device_map=None # Load to CPU first, Trainer will move to TPU
        )
        
        # Remove explicit model.to(device) - Trainer handles this
        # model = model.to(device)
        print(f"Process {rank}: Model loaded (on CPU). Trainer will move it to TPU.")

    except Exception as e:
        print(f"Process {rank}: FATAL ERROR during model loading: {e}")
        return

    if model is None or not isinstance(model, torch.nn.Module):
        print(f"Process {rank}: Critical: Model is None or not a valid PyTorch module. Exiting process.")
        return

    # Initialize GCS Client and Bucket ONCE PER PROCESS HERE
    worker_storage_client = storage.Client(project=GCP_PROJECT_ID)
    worker_bucket = worker_storage_client.bucket(GCS_BUCKET_NAME)
    print(f"Process {rank}: GCS client and bucket initialized.")

    # Load train/val/test splits using the pre-initialized bucket
    try:
        train_val_list_blob = worker_bucket.blob(GCS_TRAIN_VAL_LIST_PATH)
        train_val_files = train_val_list_blob.download_as_bytes().decode('utf-8').splitlines()
        test_list_blob = worker_bucket.blob(GCS_TEST_LIST_PATH)
        test_files = test_list_blob.download_as_bytes().decode('utf-8').splitlines()
        print(f"Process {rank}: Train/Val/Test lists loaded.")
    except Exception as e:
        print(f"Process {rank}: FATAL ERROR loading train/val/test lists from GCS: {e}")
        return

    train_files_final, val_files_final = train_test_split(train_val_files, test_size=0.15, random_state=42)

    # Filter data_entry_df for the current process's dataset splits *before* passing to NIHChestDataset
    # This creates small, picklable DataFrame slices for each dataset.
    train_df_current_process = global_data_entry_df[global_data_entry_df['Image Index'].isin(train_files_final)].copy()
    val_df_current_process = global_data_entry_df[global_data_entry_df['Image Index'].isin(val_files_final)].copy()

    # Create datasets by passing the INITALIZED CLIENT AND BUCKET, and FILTERED DATAFRAMES
    train_dataset = NIHChestDataset(
        train_df_current_process, # Pass the filtered df
        train_files_final, # This list is already correct
        global_bbox_dict, # Pass the global bbox_dict
        global_mlb, # Pass the global mlb
        transform=roi_preprocess_transforms,
        image_processor=local_image_processor,
        gcs_blob_map_names=global_gcs_blob_map_names, # Pass the global map
        use_subset=USE_SUBSET_DATA,
        gcs_client=worker_storage_client,
        gcs_bucket=worker_bucket
    )

    val_dataset = NIHChestDataset(
        val_df_current_process, # Pass the filtered df
        val_files_final, # This list is already correct
        global_bbox_dict, # Pass the global bbox_dict
        global_mlb, # Pass the global mlb
        transform=roi_preprocess_transforms,
        image_processor=local_image_processor,
        gcs_blob_map_names=global_gcs_blob_map_names, # Pass the global map
        use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None,
        gcs_client=worker_storage_client,
        gcs_bucket=worker_bucket
    )

    print(f"Process {rank}: Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")

    # Define TrainingArguments
    training_args = TrainingArguments(
      output_dir=os.path.join(OUTPUT_DIR, f"vit-finetune-chest-xray-rank{rank}"),
      per_device_train_batch_size=BATCH_SIZE_PER_CORE,
      per_device_eval_batch_size=BATCH_SIZE_PER_CORE,
      eval_strategy="steps",
      num_train_epochs=NUM_EPOCHS,
      # bf16=True, # Leave commented out for TPUs
      save_steps=50,
      eval_steps=50,
      logging_steps=50,
      learning_rate=LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
      save_total_limit=2,
      remove_unused_columns=False,
      push_to_hub=False,
      report_to='tensorboard',
      load_best_model_at_end=True,
      metric_for_best_model="avg_auroc",
      greater_is_better=True,
    )

    # Create a closure for compute_metrics to pass unique_labels_list (from shared_metadata)
    _compute_metrics = lambda eval_pred: compute_metrics_fn(eval_pred, global_unique_labels_list)

    # Instantiate Trainer
    trainer = Trainer(
        model=model, # Model is loaded to CPU, Trainer will move to TPU
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=_compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=local_image_processor,
    )

    print(f"Process {rank}: Starting training with Trainer...")

    # Train
    train_results = trainer.train()

    if rank == 0: # Only save logs/metrics from rank 0
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        print(f"Process {rank}: Training completed and model/logs saved.")

        # Evaluate final model on validation set
        metrics = trainer.evaluate(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"Process {rank}: Final evaluation metrics: {metrics}")
    
    xm.mark_step() # Ensure all processes finish before exiting


# --- Main Execution (MODIFIED TO LOAD DATA AND PASS TO SPAWNED PROCESSES) ---
if __name__ == '__main__':
    # Define image subfolders for metadata loading (these are static)
    gcs_image_subfolders = [f"images_{i:03}" for i in range(1, 13)]

    # Load all metadata in the main process
    print("\n--- Main Process: Loading all metadata ---")
    global_bbox_df, global_bbox_dict, global_data_entry_df, global_mlb, global_unique_labels_list, global_num_classes, global_gcs_blob_map_names = \
        load_global_metadata(GCP_PROJECT_ID, GCS_BUCKET_NAME, GCS_BBOX_CSV_PATH, GCS_DATA_ENTRY_CSV_PATH, GCS_IMAGE_BASE_PREFIX, gcs_image_subfolders)
    print("Main Process: All metadata loaded successfully.")

    # Prepare shared metadata tuple to pass to spawned processes
    shared_metadata = (
        global_bbox_df, global_bbox_dict, global_data_entry_df, global_mlb,
        global_unique_labels_list, global_num_classes, global_gcs_blob_map_names
    )

    # Pre-download/cache Hugging Face model
    try:
        print("Main process: Pre-downloading/caching Hugging Face model to ensure all processes have it locally...")
        _ = ViTForImageClassification.from_pretrained(MODEL_NAME)
        _ = ViTImageProcessor.from_pretrained(MODEL_NAME)
        print("Main process: Model and processor pre-download complete.")
    except Exception as e:
        print(f"Main process: WARNING: Failed to pre-download model or processor: {e}")
        print("Main process: Child processes might download concurrently, which could cause issues.")

    # Get the total number of available TPU cores.
    world_size = xr.addressable_device_count()
    print(f"Starting TPU multiprocessing on {world_size} cores via xmp.spawn...")
    
    # Pass the shared_metadata tuple as an argument to _mp_fn
    xmp.spawn(_mp_fn, args=(shared_metadata,), nprocs=None, start_method='spawn')
