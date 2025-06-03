# --- Library Imports ---
import os
import io
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
from torch.utils.data import Dataset, DataLoader # DataLoader will be used by Trainer internally
from torchvision import transforms
from google.cloud import storage
from tqdm import tqdm # For terminal-based progress bars (running as .py script)

# Hugging Face Transformers
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import TrainingArguments, Trainer

# TPU-specific imports
# Ensure these are at the very top of the script
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import tensorflow as tf # Used for TPUClusterResolver for external TPU connection

print("All necessary libraries imported.")


# --- Configuration ---
GCP_PROJECT_ID = "affable-alpha-454813-t8"
GCS_BUCKET_NAME = "chest-xray-samir"

GCS_IMAGE_BASE_PREFIX = "" # Confirm this is correct for your bucket structure

# GCS Paths to your metadata files
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
# For v3-8 (8 cores), total batch size will be BATCH_SIZE_PER_CORE * 8.
BATCH_SIZE_PER_CORE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 4
NUM_WORKERS = 8 # Number of CPU workers for data loading per DataLoader

# For faster development, use a subset of data
USE_SUBSET_DATA = None # Set to an integer (e.g., 1000) for fast testing, None for full dataset

# TPU Configuration (from Cell 2 in previous Colab structure)
TPU_NAME = "my-vit-tpu" # MUST match the name you used in `gcloud compute tpus tpu-vm create`
TPU_ZONE = "us-central1-b" # MUST match the zone you used (e.g., 'us-central1-b')

print("Configuration set.")


# --- Connect to TPU Device ---
# This block replaces the simpler `device = xm.xla_device()`
try:
    print(f"Connecting to TPU: {TPU_NAME} in zone {TPU_ZONE}...")
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_NAME, zone=TPU_ZONE, project=GCP_PROJECT_ID)
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    print("TPU system initialized successfully.")

    # Get XLA device for the current process
    device = xm.xla_device()
    # If running in multi-core context, each process gets its own device.
    # xm.set_default_device(device) # Often set by XLA's multiprocessing utilities
    print(f"Using XLA device: {device}")

except Exception as e:
    print(f"ERROR: Could not connect to external TPU: {e}")
    print("Please ensure your TPU VM is running, its name/zone/project are correct, and IAM permissions are set.")
    print("Falling back to CPU if no GPU available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using fallback device: {device}")


# --- Load Metadata (BBox and Main Data_Entry), Label Binarization, and Helper Functions ---

storage_client = storage.Client(project=GCP_PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

bbox_df = None
data_entry_df = None
mlb = None # MultiLabelBinarizer
unique_labels_list = []
NUM_CLASSES = 0

print("\n--- Loading BBox_List_2017.csv ---")
try:
    print(f"Attempting to download {GCS_BBOX_CSV_PATH} from gs://{GCS_BUCKET_NAME}/...")
    bbox_blob = bucket.blob(GCS_BBOX_CSV_PATH)
    csv_bytes = bbox_blob.download_as_bytes()
    bbox_df = pd.read_csv(io.BytesIO(csv_bytes))

    # Rename columns: 'Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]'
    bbox_df.columns = bbox_df.columns.str.replace('[\[\]]', '', regex=True) # Remove brackets
    bbox_df.columns = bbox_df.columns.str.replace(' ', '_', regex=False)   # Replace spaces
    bbox_df = bbox_df.loc[:, ~bbox_df.columns.str.contains('^Unnamed')] # Drop unnamed columns

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


print("\n--- Loading Data_Entry_2017.csv (Main Labels) ---")
try:
    print(f"Attempting to download {GCS_DATA_ENTRY_CSV_PATH} from gs://{GCS_BUCKET_NAME}/...")
    data_entry_blob = bucket.blob(GCS_DATA_ENTRY_CSV_PATH)
    csv_bytes_data_entry = data_entry_blob.download_as_bytes()
    data_entry_df = pd.read_csv(io.BytesIO(csv_bytes_data_entry))

    # Clean Finding Labels and get unique labels
    data_entry_df['Finding Labels'] = data_entry_df['Finding Labels'].apply(
        lambda x: x.replace('No Finding', '').strip() if '|' in x else x
    )
    data_entry_df['Finding Labels'] = data_entry_df['Finding Labels'].apply(
        lambda x: 'No Finding' if not x else x # If no other labels, put 'No Finding' back
    )

    all_labels_str = "|".join(data_entry_df['Finding Labels'].tolist())
    unique_labels_list = sorted(list(set([lbl for lbl in all_labels_str.split('|') if lbl])))

    if 'No Finding' not in unique_labels_list:
        unique_labels_list.append('No Finding')
    unique_labels_list = sorted(unique_labels_list) # Sort again after potentially adding 'No Finding'

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


# --- Helper Functions ---

def pad_to_square(pil_img, padding_value=0):
    w, h = pil_img.size
    if w == h:
        return pil_img
    mode = pil_img.mode # Keep original image mode (e.g., 'L' for grayscale)
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

    # Handle cases where bbox is invalid or outside image
    if right <= left or lower <= upper or w <= 0 or h <= 0:
        # Fallback to full image if bbox is invalid
        return pad_to_square(pil_img, padding_value)

    cropped_pil = pil_img.crop((left, upper, right, lower))
    return pad_to_square(cropped_pil, padding_value)

image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

roi_preprocess_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert('RGB')), # Convert grayscale to RGB
])

print("\nHelper functions and main metadata loaded.")


# --- NIHChestDataset Class Definition and DataLoader Creation for Trainer ---

# --- Build GCS Image Path Map (Run Once at script start) ---
# This map is crucial for quickly finding the GCS blob for a given image filename.
print("\n--- Building GCS Image Path Map (This may take a while for large datasets) ---")
gcs_blob_map = {}
image_subfolders = [f"images_{i:03}" for i in range(1, 13)] # images_001 to images_012

base_img_prefix = GCS_IMAGE_BASE_PREFIX
if base_img_prefix and not base_img_prefix.endswith('/'):
    base_img_prefix += '/'

for subfolder in image_subfolders:
    # This is a common NIH dataset structure: images_00X/images/FILENAME.png
    # Adjust 'images/' if your files are directly in 'images_00X/'
    current_prefix = f"{base_img_prefix}{subfolder}/images/"
    try:
        blobs_in_folder = list(bucket.list_blobs(prefix=current_prefix))
        for blob_obj in blobs_in_folder:
            if not blob_obj.name.endswith('/'): # Exclude folder blobs
                gcs_blob_map[os.path.basename(blob_obj.name)] = blob_obj
    except Exception as e:
        print(f"Warning: Error listing blobs from {current_prefix}: {e}")
print(f"Finished building GCS blob map with {len(gcs_blob_map)} unique image filenames.")


class NIHChestDataset(Dataset):
    def __init__(self, df, image_filenames_list, bbox_dict, label_binarizer, transform=None, image_processor=None, gcs_blob_map=None, use_subset=None):
        self.transform = transform
        self.image_processor = image_processor
        self.bbox_dict = bbox_dict
        self.gcs_blob_map = gcs_blob_map

        # Filter the main DataFrame to only include images in the provided list for this split
        self.df_filtered = df[df['Image Index'].isin(image_filenames_list)].copy()
        self.df_filtered.set_index('Image Index', inplace=True)

        # Prepare multi-hot encoded labels
        # Handle 'No Finding' for accurate binarization
        self.labels_original_list = []
        for img_idx in self.df_filtered.index:
            labels_str = self.df_filtered.loc[img_idx, 'Finding Labels']
            # Split and clean, if it results in empty, use ['No Finding']
            current_labels = [lbl.strip() for lbl in labels_str.split('|') if lbl.strip()]
            if not current_labels:
                current_labels = ['No Finding']
            self.labels_original_list.append(current_labels)

        self.encoded_labels = label_binarizer.transform(self.labels_original_list)


        self.image_filenames = self.df_filtered.index.tolist()

        if use_subset: # For faster testing
            self.image_filenames = self.image_filenames[:use_subset]
            self.encoded_labels = self.encoded_labels[:use_subset] # Match labels to subset

        # Pre-loading images into RAM is highly recommended for TPU/GPU performance
        self.images_in_memory = []
        print(f"Pre-loading {len(self.image_filenames)} images for this dataset split into RAM...")
        for img_name in tqdm(self.image_filenames, desc="Loading images to RAM"):
            blob_to_download = self.gcs_blob_map.get(img_name)
            if blob_to_download is None:
                # Store a black dummy image if file not found
                self.images_in_memory.append(Image.new('L', (IMG_SIZE, IMG_SIZE), color=0))
                # print(f"Warning: Image {img_name} not found in GCS map during pre-load. Storing black dummy.")
            else:
                try:
                    image_bytes = blob_to_download.download_as_bytes()
                    self.images_in_memory.append(Image.open(io.BytesIO(image_bytes)).convert('L')) # Load as grayscale
                    del image_bytes # Free up memory
                except Exception as e:
                    # Store a black dummy image if error loading
                    self.images_in_memory.append(Image.new('L', (IMG_SIZE, IMG_SIZE), color=0))
                    # print(f"Warning: Error loading {img_name} from GCS during pre-load: {e}. Storing black dummy.")
        print(f"Finished pre-loading {len(self.images_in_memory)} images for this split.")


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        original_pil = self.images_in_memory[idx] # Get pre-loaded image
        img_name = self.image_filenames[idx]
        label_vector = torch.FloatTensor(self.encoded_labels[idx])

        cropped_padded_pil_image = None
        # Use BBox cropping if available for this image, otherwise use full image (padded to square)
        if img_name in self.bbox_dict and self.bbox_dict[img_name]:
            # For simplicity, use the first bounding box found for the image
            bbox_coords = self.bbox_dict[img_name][0]
            cropped_padded_pil_image = crop_and_pad_from_bbox(original_pil, bbox_coords, padding_value=0)
        else:
            cropped_padded_pil_image = pad_to_square(original_pil, padding_value=0)

        # Apply torchvision transforms (e.g., Resize, Grayscale to RGB)
        if self.transform:
            cropped_padded_pil_image = self.transform(cropped_padded_pil_image)

        # Use Hugging Face ViTImageProcessor for final normalization and ToTensor
        if self.image_processor:
            processed_output = self.image_processor(images=cropped_padded_pil_image, return_tensors="pt")
            image_tensor = processed_output.pixel_values.squeeze(0) # Remove batch dimension added by processor
        else:
            # If not using HF processor, self.transform must include ToTensor and Normalize
            image_tensor = cropped_padded_pil_image

        # Trainer expects a dictionary for inputs
        return {'pixel_values': image_tensor, 'labels': label_vector}


# --- Create Datasets for Trainer ---
if data_entry_df is not None and mlb is not None and gcs_blob_map:
    print("\n--- Creating Datasets for Trainer ---")

    try:
        # Load image filenames for splits from GCS
        train_val_list_blob = bucket.blob(GCS_TRAIN_VAL_LIST_PATH)
        train_val_files = train_val_list_blob.download_as_bytes().decode('utf-8').splitlines()

        test_list_blob = bucket.blob(GCS_TEST_LIST_PATH)
        test_files = test_list_blob.download_as_bytes().decode('utf-8').splitlines()

        # Split train_val_files into actual train and validation
        train_files_final, val_files_final = train_test_split(train_val_files, test_size=0.15, random_state=42)

        print("Initializing Train Dataset...")
        train_dataset = NIHChestDataset(data_entry_df, train_files_final, bbox_dict, mlb,
                                        transform=roi_preprocess_transforms, image_processor=image_processor,
                                        gcs_blob_map=gcs_blob_map, use_subset=USE_SUBSET_DATA)
        print("Initializing Validation Dataset...")
        val_dataset = NIHChestDataset(data_entry_df, val_files_final, bbox_dict, mlb,
                                      transform=roi_preprocess_transforms, image_processor=image_processor,
                                      gcs_blob_map=gcs_blob_map, use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None)
        print("Initializing Test Dataset...")
        test_dataset = NIHChestDataset(data_entry_df, test_files, bbox_dict, mlb,
                                       transform=roi_preprocess_transforms, image_processor=image_processor,
                                       gcs_blob_map=gcs_blob_map, use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None)

        print(f"Train dataset size: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    except Exception as e:
        print(f"ERROR: Failed to load train/val/test lists from GCS or create datasets: {e}")
        print("Please ensure GCS_TRAIN_VAL_LIST_PATH and GCS_TEST_LIST_PATH are correct and files exist.")
        train_dataset = None
        val_dataset = None
        test_dataset = None
else:
    print("WARNING: DataFrames, MultiLabelBinarizer, or GCS blob map not initialized. Skipping Dataset creation.")

print("Datasets configured.")


# --- Data Collator and Evaluation Metrics for Trainer ---

def collate_fn(batch):
    # Trainer expects a dictionary of lists/tensors
    pixel_values = torch.stack([x['pixel_values'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

def compute_metrics(eval_pred):
    predictions = torch.sigmoid(torch.tensor(eval_pred.predictions)).numpy()
    labels = eval_pred.label_ids

    auroc_per_class = {}
    valid_classes_for_auroc = 0
    total_auroc = 0

    for i, label_name in enumerate(mlb.classes_):
        try:
            # roc_auc_score requires at least two unique class labels in targets
            if len(np.unique(labels[:, i])) > 1:
                class_roc_auc = roc_auc_score(labels[:, i], predictions[:, i])
                auroc_per_class[label_name] = class_roc_auc
                total_auroc += class_roc_auc
                valid_classes_for_auroc += 1
            else:
                # If only one class present (e.g., all 0s or all 1s), AUROC is undefined.
                auroc_per_class[label_name] = np.nan
        except Exception as e:
            # Catch other potential errors, e.g., if there's an issue with prediction/label arrays
            auroc_per_class[label_name] = np.nan

    avg_auroc = total_auroc / valid_classes_for_auroc if valid_classes_for_auroc > 0 else 0.0

    return {"avg_auroc": avg_auroc, **auroc_per_class}


# --- Model Instantiation ---

if NUM_CLASSES == 0:
    print("ERROR: NUM_CLASSES is 0. Cannot initialize model. Check previous steps for metadata loading errors.")
else:
    print(f"\n--- Loading Model: {MODEL_NAME} ---")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True, # Important for replacing the classifier head
        id2label={i: c for i, c in enumerate(mlb.classes_)}, # For better logging/analysis
        label2id={c: i for i, c in enumerate(mlb.classes_)} # For better logging/analysis
    )

    model.to(device) # Move model to the detected XLA or CPU/GPU device
    print(f"Model loaded: {MODEL_NAME} with {NUM_CLASSES} output classes for fine-tuning.")
    print(f"Model will be trained on: {device}")


# --- Define Training Arguments and Instantiate Trainer ---

if train_dataset is None or val_dataset is None or model is None:
    print("ERROR: Datasets or model not initialized. Cannot set up Trainer.")
else:
    training_args = TrainingArguments(
      output_dir=OUTPUT_DIR,
      # These batch sizes are PER DEVICE. Trainer handles distribution.
      per_device_train_batch_size=BATCH_SIZE_PER_CORE,
      per_device_eval_batch_size=BATCH_SIZE_PER_CORE,
      
      # Evaluation strategy: 'steps' for regular evaluation (e.g., every eval_steps)
      # or 'epoch' for evaluation at the end of each epoch.
      evaluation_strategy="steps", # Evaluate at specific steps
      eval_steps=len(train_dataset) // (BATCH_SIZE_PER_CORE * 10), # Evaluate every 10% of an epoch for example
      # Adjust eval_steps to a reasonable number. If eval_steps < 1, it won't evaluate.
      # You might also set this to `len(train_loader)` to evaluate per epoch.

      num_train_epochs=NUM_EPOCHS,
      
      # For TPUs, `fp16=True` internally translates to bfloat16.
      fp16=True,
      
      save_steps=len(train_dataset) // (BATCH_SIZE_PER_CORE * 5), # Save checkpoint every 20% of an epoch
      # Adjust save_steps similarly. If save_steps < 1, it won't save.

      logging_steps=len(train_dataset) // (BATCH_SIZE_PER_CORE * 20), # Log every 5% of an epoch

      learning_rate=LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
      save_total_limit=2, # Keep only the last 2 checkpoints
      remove_unused_columns=False, # Important if your dataset returns extra keys
      push_to_hub=False,
      report_to='tensorboard',
      load_best_model_at_end=True, # Load the model with the best metric_for_best_model after training
      metric_for_best_model="avg_auroc", # Metric to monitor for best model
      greater_is_better=True, # For AUROC, higher is better
      gradient_accumulation_steps=1, # Accumulate gradients over N steps before update (useful for larger effective batch size)
      dataloader_num_workers=NUM_WORKERS # Number of subprocesses for data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn, # Custom collate function
        compute_metrics=compute_metrics, # Custom metrics function
        train_dataset=train_dataset, # Pass the PyTorch Dataset directly
        eval_dataset=val_dataset,    # Pass the PyTorch Dataset directly
    )

    print("Training arguments and Trainer instantiated.")


# --- Training Execution ---

if 'trainer' in globals():
    print("\n--- Starting Training ---")
    train_results = trainer.train() # This starts the training loop

    # Save final model and metrics
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state() # Saves Trainer's internal state (e.g., optimizer, scheduler)

    print("\n--- Training Finished ---")

    # --- Plotting Training History (Saved to File) ---
    # Metrics are logged in trainer.state.log_history
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    train_steps = [log['step'] for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
    eval_steps = [log['step'] for log in trainer.state.log_history if 'eval_loss' in log]

    if train_losses:
        plt.plot(train_steps, train_losses, label='Train Loss')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Training Steps')
    plt.legend()
    plt.grid(True)

    # AUROC Plot
    plt.subplot(1, 2, 2)
    eval_auroc_history = [log['eval_avg_auroc'] for log in trainer.state.log_history if 'eval_avg_auroc' in log]
    if eval_auroc_history:
        plt.plot(eval_steps, eval_auroc_history, label='Validation Avg AUROC', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Avg AUROC')
    plt.title('Validation Average AUROC Over Steps')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_metrics_plot.png")
    plt.savefig(plot_path)
    print(f"Training metrics plot saved to {plot_path}")
    plt.close() # Close the figure to free memory

else:
    print("Trainer not instantiated. Skipping training execution.")


# --- Final Evaluation on Test Set ---

if 'trainer' in globals() and 'test_dataset' in globals() and test_dataset is not None:
    print("\n--- Evaluating on Test Set ---")
    # The Trainer.evaluate() method expects a Dataset, not a DataLoader
    metrics = trainer.evaluate(test_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print(f"\nTest results saved to {os.path.join(OUTPUT_DIR, 'eval_results.json')}")
    print("\nTest AUROC per class:")
    # Filter for individual class AUROC scores
    test_auroc_per_class = {k.replace('eval_', ''): v for k, v in metrics.items() if k.startswith('eval_') and k not in ['eval_loss', 'eval_avg_auroc', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']}
    # Sort for better readability
    sorted_auroc = sorted(test_auroc_per_class.items(), key=lambda item: item[1] if not np.isnan(item[1]) else -1, reverse=True)

    for disease, score in sorted_auroc:
        print(f"  {disease}: {score:.4f}")

    print("\n--- Evaluation Finished ---")

else:
    print("Trainer or test dataset not available. Skipping test set evaluation.")