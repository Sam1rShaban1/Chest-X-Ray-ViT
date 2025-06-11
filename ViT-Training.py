# --- Library Imports ---
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

# Hugging Face Transformers
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Trainer, TrainingArguments

# TPU-specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

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
NUM_EPOCHS = 4

USE_SUBSET_DATA = None

print("Configuration set.")

# --- Global Variables ---
bbox_df = None
data_entry_df = None
mlb = None
unique_labels_list = []
NUM_CLASSES = 0
gcs_blob_map_names = {}

# --- Load Metadata ---
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

# --- Helper Functions ---
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

# --- Build GCS Image Path Map ---
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

# --- Dataset Class ---
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
def compute_metrics_fn(eval_pred):
    global unique_labels_list

    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels_np = labels

    auroc_per_class = {}
    valid_classes = 0
    total_auroc = 0

    for i, label_name in enumerate(unique_labels_list):
        try:
            if len(np.unique(labels_np[:, i])) > 1:
                class_roc_auc = roc_auc_score(labels_np[:, i], probs[:, i])
                auroc_per_class[label_name] = class_roc_auc
                total_auroc += class_roc_auc
                valid_classes += 1
            else:
                auroc_per_class[label_name] = np.nan
        except ValueError as e:
            auroc_per_class[label_name] = np.nan

    avg_auroc = total_auroc / valid_classes if valid_classes > 0 else 0.0
    metrics = {"avg_auroc": avg_auroc}
    return metrics


# --- Main Training Function for TPU ---
def _mp_fn(rank, world_size):
    """Main training function for TPU multiprocessing"""
    
    # Initialize TPU device
    device = xm.xla_device()
    print(f"Process {rank}: Starting training on device {device}")

    # Access global variables for metadata
    global data_entry_df, bbox_dict, mlb, gcs_blob_map_names, unique_labels_list, NUM_CLASSES

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
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
            id2label={i: c for i, c in enumerate(unique_labels_list)},
            label2id={c: i for i, c in enumerate(unique_labels_list)}
        )
        
        # Move model to TPU device
        model = model.to(device)
        print(f"Process {rank}: Model loaded and moved to TPU device: {device}")

    except Exception as e:
        print(f"Process {rank}: FATAL ERROR during model loading: {e}")
        return

    if model is None or not isinstance(model, torch.nn.Module):
        print(f"Process {rank}: Critical: Model is None or not a valid PyTorch module. Exiting process.")
        return

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

    # Create data samplers for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_CORE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0  # Important for TPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_PER_CORE,
        sampler=val_sampler,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0  # Important for TPU
    )

    # Wrap data loaders for TPU
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights with XLA optimization
            xm.optimizer_step(optimizer)
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Process {rank}, Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Process {rank}, Epoch {epoch+1}/{NUM_EPOCHS} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Validation at the end of each epoch
        if rank == 0:  # Only run validation on rank 0
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch['pixel_values']
                    labels = batch['labels']
                    
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits
                    
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Store predictions and labels for metrics
                    predictions = torch.sigmoid(logits).cpu().numpy()
                    all_predictions.append(predictions)
                    all_labels.append(labels.cpu().numpy())
            
            # Compute validation metrics
            if all_predictions:
                all_predictions = np.concatenate(all_predictions, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                
                # Compute AUROC
                valid_classes = 0
                total_auroc = 0
                
                for i in range(NUM_CLASSES):
                    try:
                        if len(np.unique(all_labels[:, i])) > 1:
                            class_roc_auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                            total_auroc += class_roc_auc
                            valid_classes += 1
                    except ValueError:
                        pass
                
                avg_auroc = total_auroc / valid_classes if valid_classes > 0 else 0.0
                avg_val_loss = val_loss / val_batches
                
                print(f"Process {rank}, Epoch {epoch+1}/{NUM_EPOCHS} - Validation Loss: {avg_val_loss:.4f}, Validation AUROC: {avg_auroc:.4f}")
            
            model.train()
    
    # Save model (only on rank 0)
    if rank == 0:
        output_path = os.path.join(OUTPUT_DIR, "final_model")
        os.makedirs(output_path, exist_ok=True)
        
        # Save the model state dict
        torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save the model config
        model.config.save_pretrained(output_path)
        
        # Save the image processor
        local_image_processor.save_pretrained(output_path)
        
        print(f"Process {rank}: Model saved to {output_path}")

    print(f"Process {rank}: Training completed successfully!")


# --- Main Execution ---
if __name__ == '__main__':
    # This block handles initial setup that should only happen once
    # and load global variables before spawning processes.

    if data_entry_df is None or mlb is None or not gcs_blob_map_names:
        print("ERROR: Metadata not loaded properly. Cannot proceed.")
        exit(1)

    try:
        print("Main process: Pre-downloading/caching Hugging Face model to ensure all processes have it locally...")
        _ = ViTForImageClassification.from_pretrained(MODEL_NAME)
        _ = ViTImageProcessor.from_pretrained(MODEL_NAME)
        print("Main process: Model and processor pre-download complete.")
    except Exception as e:
        print(f"Main process: WARNING: Failed to pre-download model or processor: {e}")
        print("Main process: Child processes might download concurrently, which could cause issues.")

    # Start TPU multiprocessing
    print("Starting TPU multiprocessing...")
    
    # Get the number of TPU cores
    world_size = xm.xrt_world_size()
    print(f"World size (number of TPU cores): {world_size}")
    
    # Launch training on all TPU cores
    xmp.spawn(_mp_fn, args=(world_size,), nprocs=world_size, start_method='fork')
