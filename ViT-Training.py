# --- Library Imports ---
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. Tell XLA to explicitly use the PJRT (TPU) backend
os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011" # Standard TPU VM config
os.environ["XLA_USE_PJRT"] = "True" # Explicitly tell XLA to use PJRT backend
os.environ["XLA_CLIENT_ALLOCATOR"] = "platform" # More explicit memory allocation for PJRT

# 3. Suppress some potential backend warnings (optional, but can clean logs)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json # For saving test results
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
from google.cloud import storage # Only import, do not initialize globally for pickling safety
from tqdm import tqdm # For terminal-based progress bars

# Hugging Face Transformers
from transformers import ViTForImageClassification, ViTImageProcessor

# TPU-specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp # Main entry point for TPU multiprocessing
import torch_xla.distributed.parallel_loader as pl # For efficient data loading to TPU

print("All necessary libraries imported.")


# --- Configuration (remains global) ---
GCP_PROJECT_ID = "affable-alpha-454813-t8"
GCS_BUCKET_NAME = "chest-xray-samir"

GCS_IMAGE_BASE_PREFIX = "" 

# GCS Paths to your metadata files (assuming they are at the root of GCS_BUCKET_NAME)
GCS_BBOX_CSV_PATH = "BBox_List_2017.csv"
GCS_DATA_ENTRY_CSV_PATH = "Data_Entry_2017.csv"
GCS_TRAIN_VAL_LIST_PATH = "train_val_list.txt"
GCS_TEST_LIST_PATH = "test_list.txt"

# --- Local Output Directory on the VM ---
OUTPUT_DIR = os.path.expanduser("~/vit_finetune_results/")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- ViT Model & Training Hyperparameters ---
MODEL_NAME = 'google/vit-base-patch16-384'
IMG_SIZE = 384

VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD = [0.229, 0.224, 0.225]

# TPU-specific batch size: This is PER CORE.
BATCH_SIZE_PER_CORE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 4
NUM_WORKERS = 4

# For faster development, use a subset of data
USE_SUBSET_DATA = None 

print("Configuration set.")


# --- Global Variables for Metadata (loaded by main process) ---
bbox_df = None
data_entry_df = None
mlb = None
unique_labels_list = []
NUM_CLASSES = 0
gcs_blob_map_names = {} 


# --- Load Metadata (from main process) ---
print("\n--- Loading Metadata ---")
try:
    _temp_storage_client = storage.Client(project=GCP_PROJECT_ID)
    _temp_bucket = _temp_storage_client.bucket(GCS_BUCKET_NAME)

    print(f"Attempting to download {GCS_BBOX_CSV_PATH} from gs://{GCS_BUCKET_NAME}/...")
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
    print(f"ERROR: Failed to load or process BBox_List_2017.csv: {e}")
    print("Please ensure GCS_BBOX_CSV_PATH is correct and the file exists at the root of your bucket.")
    bbox_df = None
    bbox_dict = {}

try:
    print(f"Attempting to download {GCS_DATA_ENTRY_CSV_PATH} from gs://{GCS_BUCKET_NAME}/...")
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
    print(f"Number of classes for model: {NUM_CLASSES}")

except Exception as e:
    print(f"ERROR: Failed to load or process Data_Entry_2017.csv: {e}")
    print("Please ensure GCS_DATA_ENTRY_CSV_PATH is correct and the file exists at the root.")
    data_entry_df = None
    mlb = None
    unique_labels_list = []
    NUM_CLASSES = 0

if NUM_CLASSES == 0:
    print("FATAL: NUM_CLASSES is 0 after metadata load. Exiting.")
    exit()

del _temp_storage_client
del _temp_bucket


print("\nHelper functions and main metadata loaded.")


# --- Helper Functions (remain global, can access global config vars) ---
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

# Image processor for ViT - THIS IS NOW INITIALIZED LOCALLY WITHIN _mp_fn
# image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME) # <--- THIS LINE IS REMOVED

roi_preprocess_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert('RGB')),
])


# --- Build GCS Image Path Map (Run Once at script start, in main process) ---
print("\n--- Building GCS Image Path Map (This may take a while for large datasets) ---")
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
print(f"Finished building GCS blob map with {len(gcs_blob_map_names)} unique image filenames.")

del _temp_storage_client_map
del _temp_bucket_map


# --- NIHChestDataset Class Definition (updated for GCS client per worker) ---
class NIHChestDataset(Dataset):
    def __init__(self, df, image_filenames_list, bbox_dict, label_binarizer, transform=None, image_processor=None, gcs_blob_map_names=None, use_subset=None):
        self.transform = transform
        self.image_processor = image_processor # This will now be the local_image_processor
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

        print(f"Dataset initialized to load {len(self.image_filenames)} images on demand.")


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        label_vector = torch.FloatTensor(self.encoded_labels[idx])

        original_pil = None
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
                original_pil = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0) 
        else:
            original_pil = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0) 

        cropped_padded_pil_image = None
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


# --- Training and Evaluation Functions (adapted for xmp.spawn) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, rank, total_processes):
    model.train()
    running_loss = 0.0
    para_loader = pl.ParallelLoader(dataloader, [device])
    pbar = tqdm(para_loader.per_device_loader(device), desc=f"Training (Rank {rank})", disable=(rank != 0))

    for i, batch in enumerate(pbar):
        images = batch['pixel_values'].to(torch.bfloat16) # Converted to bfloat16
        labels = batch['labels']

        optimizer.zero_grad()

        outputs = model(images).logits
        loss = criterion(outputs, labels)

        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        xm.mark_step()

        # All-reduce loss to get a sum across all cores for consistent logging
        reduced_loss = xm.all_reduce(xm.REDUCE_SUM, loss)
        running_loss += reduced_loss.item() * images.size(0)

        if rank == 0:
            pbar.set_postfix(loss=reduced_loss.item())

    # Aggregate total loss across all processes for epoch loss
    total_samples_this_rank = len(dataloader.dataset) 
    global_total_samples = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(total_samples_this_rank, dtype=torch.float32, device=device))
    
    global_total_loss_sum = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(running_loss, dtype=torch.float32, device=device))
    
    epoch_loss = global_total_loss_sum.item() / global_total_samples.item()

    return epoch_loss


def evaluate_model(model, dataloader, criterion, device, label_names, rank, total_processes):
    model.eval()
    running_loss = 0.0
    all_preds_list = []
    all_targets_list = []

    para_loader = pl.ParallelLoader(dataloader, [device])
    pbar = tqdm(para_loader.per_device_loader(device), desc=f"Evaluating (Rank {rank})", disable=(rank != 0))

    with torch.no_grad():
        for batch in pbar:
            images = batch['pixel_values'].to(torch.bfloat16) # Converted to bfloat16
            labels = batch['labels']

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            reduced_loss = xm.all_reduce(xm.REDUCE_SUM, loss)
            running_loss += reduced_loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            all_preds_list.append(xm.all_gather(probs, dim=0).cpu().numpy())
            all_targets_list.append(xm.all_gather(labels, dim=0).cpu().numpy())
            xm.mark_step()

    total_samples_this_rank = len(dataloader.dataset)
    global_total_samples = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(total_samples_this_rank, dtype=torch.float32, device=device))
    global_total_loss_sum = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(running_loss, dtype=torch.float32, device=device))
    epoch_loss = global_total_loss_sum.item() / global_total_samples.item()


    all_preds = np.concatenate(all_preds_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)

    auroc_per_class = {}
    valid_classes_for_auroc = 0
    total_auroc = 0

    for i, label_name in enumerate(label_names):
        try:
            if len(np.unique(all_targets[:, i])) > 1:
                class_roc_auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                auroc_per_class[label_name] = class_roc_auc
                total_auroc += class_roc_auc
                valid_classes_for_auroc += 1
            else:
                auroc_per_class[label_name] = np.nan
        except Exception:
            auroc_per_class[label_name] = np.nan

    avg_auroc = total_auroc / valid_classes_for_auroc if valid_classes_for_auroc > 0 else 0.0

    return epoch_loss, avg_auroc, auroc_per_class


# --- Main function for xmp.spawn ---
def _mp_fn(rank, data_entry_df, bbox_dict, mlb, gcs_blob_map_names, unique_labels_list):
    # This function is executed by each process on its assigned TPU core
    device = xm.xla_device()
    print(f"Process {rank}: Starting on device {device}")

    torch.set_default_device(device) 
    print(f"Process {rank}: PyTorch default device (after setting): {torch.get_default_device()}")

    # --- Initialize image_processor locally within each spawned process ---
    print(f"Process {rank}: Initializing ViTImageProcessor locally...")
    local_image_processor = None # Initialize defensively
    try:
        local_image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        print(f"Process {rank}: ViTImageProcessor initialized successfully.")
    except Exception as e:
        print(f"Process {rank}: ERROR: Failed to initialize ViTImageProcessor: {e}")
        import traceback; traceback.print_exc()
        return # Exit this process if critical initialization fails

    # --- Initialize model on this device ---
    print(f"Process {rank}: Attempting to load model from_pretrained...")
    model = None # Initialize model to None defensively
    try:
        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True, 
            id2label={i: c for i, c in enumerate(unique_labels_list)},
            label2id={c: i for c, i in enumerate(unique_labels_list)},
            torch_dtype=torch.float32, # Load to CPU as float32 first
            low_cpu_mem_usage=False, 
            device_map=None # Load on CPU first
        )
        if model is None: # Explicit check if from_pretrained somehow returned None
            raise ValueError("ViTForImageClassification.from_pretrained returned None.")
        print(f"Process {rank}: Model loaded to CPU (object type: {type(model)}). Now attempting to move to XLA device and convert to bfloat16...")
        model.to(device).to(torch.bfloat16) 
        print(f"Process {rank}: Model successfully moved to XLA device (bfloat16).")
    except Exception as e:
        print(f"Process {rank}: ERROR: Failed during model loading or transfer: {e}")
        import traceback; traceback.print_exc()
        return # Exit this process if critical initialization fails

    xm.mark_step() # Ensure model parameters are properly materialized on the device

    # --- Criterion and optimizer ---
    print(f"Process {rank}: Initializing criterion...")
    criterion = None # Initialize criterion to None defensively
    try:
        criterion = nn.BCEWithLogitsLoss().to(device)
        print(f"Process {rank}: Criterion initialized successfully.")
    except Exception as e:
        print(f"Process {rank}: ERROR: Failed to initialize criterion: {e}")
        import traceback; traceback.print_exc()
        return # Exit this process if critical initialization fails

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=False)

    # --- Re-initialize GCS client per process for robustness in multi-worker setup ---
    worker_storage_client = storage.Client(project=GCP_PROJECT_ID)
    worker_bucket = worker_storage_client.bucket(GCS_BUCKET_NAME)

    train_val_list_blob = worker_bucket.blob(GCS_TRAIN_VAL_LIST_PATH)
    train_val_files = train_val_list_blob.download_as_bytes().decode('utf-8').splitlines()
    test_list_blob = worker_bucket.blob(GCS_TEST_LIST_PATH)
    test_files = test_list_blob.download_as_bytes().decode('utf-8').splitlines()

    train_files_final, val_files_final = train_test_split(train_val_files, test_size=0.15, random_state=42)

    # --- Create Datasets ---
    print(f"Process {rank}: Initializing Train Dataset...")
    train_dataset = NIHChestDataset(data_entry_df, train_files_final, bbox_dict, mlb,
                                    transform=roi_preprocess_transforms, image_processor=local_image_processor, # Use local_image_processor
                                    gcs_blob_map_names=gcs_blob_map_names, use_subset=USE_SUBSET_DATA)
    print(f"Process {rank}: Initializing Validation Dataset...")
    val_dataset = NIHChestDataset(data_entry_df, val_files_final, bbox_dict, mlb,
                                  transform=roi_preprocess_transforms, image_processor=local_image_processor, # Use local_image_processor
                                  gcs_blob_map_names=gcs_blob_map_names, use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None)
    print(f"Process {rank}: Initializing Test Dataset...")
    test_dataset = NIHChestDataset(data_entry_df, test_files, bbox_dict, mlb,
                                   transform=roi_preprocess_transforms, image_processor=local_image_processor, # Use local_image_processor
                                   gcs_blob_map_names=gcs_blob_map_names, use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None)

    print(f"Process {rank}: Train dataset size: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # --- Create DataLoaders for each process ---
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xla_device_count(), 
        rank=xm.get_ordinal(),              
        shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xla_device_count(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xm.xla_device_count(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_CORE, sampler=train_sampler, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_CORE, sampler=val_sampler, num_workers=NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PER_CORE, sampler=test_sampler, num_workers=NUM_WORKERS, drop_last=False)

    print(f"Process {rank}: Train Dataloader: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")


    # --- Training Loop ---
    best_val_auroc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_avg_auroc': []}

    best_model_gcs_path = f"gs://{GCS_BUCKET_NAME}/vit_models/vit_nih_best_model.pth"

    print(f"\nProcess {rank}: Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nProcess {rank} --- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_sampler.set_epoch(epoch) 

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, rank, xm.xla_device_count())
        history['train_loss'].append(train_loss)
        if rank == 0: 
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        val_loss, val_avg_auroc, val_auroc_per_class = evaluate_model(model, val_loader, criterion, device, unique_labels_list, rank, xm.xla_device_count())
        history['val_loss'].append(val_loss)
        history['val_avg_auroc'].append(val_avg_auroc)
        if rank == 0: 
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Avg AUROC: {val_avg_auroc:.4f}")

        scheduler.step(val_loss)

        if rank == 0:
            if val_avg_auroc > best_val_auroc:
                best_val_auroc = val_avg_auroc # Corrected typo here
                xm.save(model.state_dict(), best_model_gcs_path) 
                print(f"Process {rank}: New best model saved with Avg AUROC: {best_val_auroc:.4f} to {best_model_gcs_path}")

            if epoch == NUM_EPOCHS - 1:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss Over Epochs')
                plt.legend()
                plt.grid(True)

                plt.subplot(1, 2, 2)
                plt.plot(history['val_avg_auroc'], label='Val Avg AUROC', color='orange')
                plt.xlabel('Epoch')
                plt.ylabel('Avg AUROC')
                plt.title('Validation Average AUROC Over Epochs')
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plot_path = os.path.join(OUTPUT_DIR, "training_metrics_plot.png")
                plt.savefig(plot_path)
                print(f"Process {rank}: Training metrics plot saved to {plot_path}")
                plt.close() 
        
        xm.mark_step() 

    print(f"\nProcess {rank}: Training finished.")


    # --- Final Evaluation on Test Set (only on master process) ---
    if rank == 0:
        print("\n--- Evaluating on Test Set ---")
        if os.path.exists(best_model_gcs_path) or best_model_gcs_path.startswith("gs://"): 
            try:
                model.load_state_dict(xm.load(best_model_gcs_path)) 
                model.to(device).to(torch.bfloat16) # Ensure model is on device after loading state dict

                test_loss, test_avg_auroc, test_auroc_per_class = evaluate_model(model, test_loader, criterion, device, unique_labels_list, rank, xm.xla_device_count())

                print(f"\nTest Loss: {test_loss:.4f}")
                print(f"Test Average AUROC: {test_avg_auroc:.4f}")
                print("\nTest AUROC per class:")
                sorted_auroc = sorted(test_auroc_per_class.items(), key=lambda item: item[1] if not np.isnan(item[1]) else -1, reverse=True)
                for disease, score in sorted_auroc:
                    print(f"  {disease}: {score:.4f}")

                test_results_path = os.path.join(OUTPUT_DIR, "test_results.json")
                with open(test_results_path, 'w') as f:
                    json.dump({"loss": test_loss, "avg_auroc": test_avg_auroc, "auroc_per_class": test_auroc_per_class}, f, indent=4)
                print(f"Test results saved to {test_results_path}")

                print("\n--- Evaluation Finished ---")
            except Exception as e:
                print(f"Process {rank}: ERROR during final test evaluation: {e}")
                print("Skipping final test evaluation due to error.")
        else:
            print(f"Process {rank}: Best model checkpoint not found at {best_model_gcs_path}. Skipping final test evaluation.")
    else:
        print(f"Process {rank}: Skipping test set evaluation (master-only).")


# --- Main execution block ---
if __name__ == '__main__':
    if data_entry_df is None or bbox_df is None or mlb is None or not gcs_blob_map_names:
        print("ERROR: Global metadata (DataFrames, MLB, or GCS map) not fully loaded. Cannot proceed with training.")
        exit(1)

    xmp.spawn(_mp_fn, args=(data_entry_df, bbox_dict, mlb, gcs_blob_map_names, unique_labels_list,), nprocs=None, start_method='spawn')
