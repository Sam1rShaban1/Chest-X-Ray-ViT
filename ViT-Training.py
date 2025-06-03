# --- Library Imports ---
import os  # Operating system interface
import io  # Handling byte streams

# --- Image Processing ---
from PIL import Image, ImageDraw  # Image handling and annotation
import matplotlib.pyplot as plt  # Plotting for training curves and visualizations

# --- Scientific and Numeric Libraries ---
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and CSV handling

# --- Machine Learning: Preprocessing & Evaluation ---
from sklearn.model_selection import train_test_split  # Splitting datasets
from sklearn.preprocessing import MultiLabelBinarizer  # Multi-label binarization
from sklearn.metrics import roc_auc_score  # AUROC evaluation metric

# --- PyTorch Core ---
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network layers
import torch.optim as optim  # Optimizers
from torch.utils.data import Dataset, DataLoader  # Custom dataset and batch loading
from torchvision import transforms  # Image transformations

# --- TPU & GCS Integration ---
from google.cloud import storage  # Google Cloud Storage API for TPU data loading

# --- Progress Bar ---
from tqdm.notebook import tqdm  # For notebook-based progress bars
from tqdm import tqdm  # For terminal-based progress bars

# --- Hugging Face Transformers ---
from transformers import ViTForImageClassification, ViTImageProcessor  # Pretrained ViT model and processor
from transformers import TrainingArguments, Trainer  # Training utility classes


print("All necessary libraries imported.")

# --- Configuration ---
GCP_PROJECT_ID = "affable-alpha-454813-t8"
GCS_BUCKET_NAME = "chest-xray-samir"

GCS_IMAGE_BASE_PREFIX = "" # Adjust this based on your actual bucket structure for images
GCS_BBOX_CSV_PATH = "BBox_List_2017.csv"
GCS_DATA_ENTRY_CSV_PATH = "Data_Entry_2017.csv"
GCS_TRAIN_VAL_LIST_PATH = "train_val_list.txt"
GCS_TEST_LIST_PATH = "test_list.txt"

# --- Local Output Directory on the VM ---
OUTPUT_DIR = os.path.expanduser("~/vit_finetune_results/")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- ViT Model & Training Hyperparameters ---
MODEL_NAME = 'google/vit-base-patch16-384' # This model is pre-trained on ImageNet-1k (not 21k)
IMG_SIZE = 384 # ViT model expects 384x384 input resolution

# ImageNet-1k mean/std for normalization as ViT is pre-trained on it
# Note: These values are for ImageNet-1k, different from ImageNet-21k
VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD = [0.229, 0.224, 0.225]

BATCH_SIZE_PER_CORE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 4
NUM_WORKERS = 8 # INCREASED: Leverage 96 CPU cores for faster data loading

USE_SUBSET_DATA = None # Set to an integer (e.g., 1000) for fast testing, None for full dataset


print("Configuration set.")

# --- Connect to TPU Device ---
try:
    device = xm.xla_device()
    print(f"Using XLA device: {device}")
except Exception as e:
    print(f"ERROR: Could not acquire XLA device: {e}")
    print("Ensure you are running on a TPU VM with PyTorch/XLA installed correctly.")
    device = torch.device("cpu")
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


print("\n--- Loading Data_Entry_2017.csv (Main Labels) ---")
try:
    print(f"Attempting to download {GCS_DATA_ENTRY_CSV_PATH} from gs://{GCS_BUCKET_NAME}/...")
    data_entry_blob = bucket.blob(GCS_DATA_ENTRY_CSV_PATH)
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

image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

roi_preprocess_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert('RGB')),
])

print("\nHelper functions and main metadata loaded.")


# --- NIHChestDataset Class Definition and DataLoader Creation ---

# --- Build GCS Image Path Map (Run Once at script start) ---
print("\n--- Building GCS Image Path Map (This may take a while for large datasets) ---")
gcs_blob_map = {}
image_subfolders = [f"images_{i:03}" for i in range(1, 13)]

base_img_prefix = GCS_IMAGE_BASE_PREFIX
if base_img_prefix and not base_img_prefix.endswith('/'):
    base_img_prefix += '/'

for subfolder in image_subfolders:
    current_prefix = f"{base_img_prefix}{subfolder}/images/" # Common NIH dataset structure
    try:
        blobs_in_folder = list(bucket.list_blobs(prefix=current_prefix))
        for blob_obj in blobs_in_folder:
            if not blob_obj.name.endswith('/'):
                gcs_blob_map[os.path.basename(blob_obj.name)] = blob_obj
    except Exception as e:
        print(f"Warning: Error listing blobs from {current_prefix}: {e}")
print(f"Finished building GCS blob map with {len(gcs_blob_map)} unique image filenames.")


class NIHChestDataset(Dataset):
    def __init__(self, df, image_filenames_list, bbox_dict, label_binarizer, transform=None, image_processor=None, gcs_blob_map=None, use_subset=None):
        self.transform = transform
        self.image_processor = image_processor
        self.bbox_dict = bbox_dict

        self.df_filtered = df[df['Image Index'].isin(image_filenames_list)].copy()
        self.df_filtered.set_index('Image Index', inplace=True)

        self.labels_original_list = self.df_filtered['Finding Labels'].apply(lambda x: x.split('|')).tolist()
        self.encoded_labels = label_binarizer.transform(self.labels_original_list)

        self.image_filenames = self.df_filtered.index.tolist()

        if use_subset:
            self.image_filenames = self.image_filenames[:use_subset]
            self.encoded_labels = self.encoded_labels[:use_subset]

        self.images_in_memory = []
        print(f"Pre-loading {len(self.image_filenames)} images for this dataset split into RAM...")
        for img_name in tqdm(self.image_filenames, desc="Loading images to RAM"):
            blob_to_download = gcs_blob_map.get(img_name)
            if blob_to_download is None:
                self.images_in_memory.append(Image.new('L', (IMG_SIZE, IMG_SIZE), color=0))
                print(f"Warning: Image {img_name} not found in GCS map during pre-load. Storing black dummy.")
            else:
                try:
                    image_bytes = blob_to_download.download_as_bytes()
                    self.images_in_memory.append(Image.open(io.BytesIO(image_bytes)).convert('L'))
                    del image_bytes
                except Exception as e:
                    self.images_in_memory.append(Image.new('L', (IMG_SIZE, IMG_SIZE), color=0))
                    print(f"Warning: Error loading {img_name} from GCS during pre-load: {e}. Storing black dummy.")
        print(f"Finished pre-loading {len(self.images_in_memory)} images for this split.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        original_pil = self.images_in_memory[idx]
        img_name = self.image_filenames[idx]
        label_vector = torch.FloatTensor(self.encoded_labels[idx])

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

        return image_tensor, label_vector


# --- Create DataLoaders ---
if data_entry_df is not None and mlb is not None and gcs_blob_map:
    print("\n--- Creating DataLoaders ---")

    try:
        train_val_list_blob = bucket.blob(GCS_TRAIN_VAL_LIST_PATH)
        train_val_files = train_val_list_blob.download_as_bytes().decode('utf-8').splitlines()

        test_list_blob = bucket.blob(GCS_TEST_LIST_PATH)
        test_files = test_list_blob.download_as_bytes().decode('utf-8').splitlines()

        train_files_final, val_files_final = train_test_split(train_val_files, test_size=0.15, random_state=42)

        print("Initializing Train Dataset (pre-loading images)...")
        train_dataset = NIHChestDataset(data_entry_df, train_files_final, bbox_dict, mlb,
                                        transform=roi_preprocess_transforms, image_processor=image_processor,
                                        gcs_blob_map=gcs_blob_map, use_subset=USE_SUBSET_DATA)
        print("Initializing Validation Dataset (pre-loading images)...")
        val_dataset = NIHChestDataset(data_entry_df, val_files_final, bbox_dict, mlb,
                                      transform=roi_preprocess_transforms, image_processor=image_processor,
                                      gcs_blob_map=gcs_blob_map, use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None)
        print("Initializing Test Dataset (pre-loading images)...")
        test_dataset = NIHChestDataset(data_entry_df, test_files, bbox_dict, mlb,
                                       transform=roi_preprocess_transforms, image_processor=image_processor,
                                       gcs_blob_map=gcs_blob_map, use_subset=USE_SUBSET_DATA // 5 if USE_SUBSET_DATA else None)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_CORE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_CORE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PER_CORE, shuffle=False, num_workers=NUM_WORKERS)

        print(f"Train dataset size: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"Train Dataloader: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")

        try:
            sample_images, sample_labels = next(iter(train_loader))
            print("Sample batch - Images shape:", sample_images.shape, "Labels shape:", sample_labels.shape)
            del sample_images, sample_labels
        except Exception as e:
            print(f"ERROR: Could not load a sample batch from DataLoader: {e}")
            print("This usually indicates an issue with image paths, loading, or preprocessing in NIHChestDataset.")

    except Exception as e:
        print(f"ERROR: Failed to load train/val/test lists from GCS: {e}")
        print("Please ensure GCS_TRAIN_VAL_LIST_PATH and GCS_TEST_LIST_PATH are correct and files exist.")
else:
    print("WARNING: DataFrames, MultiLabelBinarizer, or GCS blob map not initialized. Skipping DataLoader creation.")

print("Dataset and DataLoaders configured.")


# --- Data Collator, Evaluation Metrics, and Model Instantiation ---

def collate_fn(batch):
    pixel_values = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

def compute_metrics(p):
    predictions = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    labels = p.label_ids

    auroc_per_class = {}
    valid_classes_for_auroc = 0
    total_auroc = 0

    for i, label_name in enumerate(mlb.classes_):
        try:
            if len(np.unique(labels[:, i])) > 1:
                class_roc_auc = roc_auc_score(labels[:, i], predictions[:, i])
                auroc_per_class[label_name] = class_roc_auc
                total_auroc += class_roc_auc
                valid_classes_for_auroc += 1
            else:
                auroc_per_class[label_name] = np.nan
        except Exception as e:
            auroc_per_class[label_name] = np.nan

    avg_auroc = total_auroc / valid_classes_for_auroc if valid_classes_for_auroc > 0 else 0.0

    return {"avg_auroc": avg_auroc, **auroc_per_class}


if NUM_CLASSES == 0:
    print("ERROR: NUM_CLASSES is 0. Cannot initialize model. Check previous steps for metadata loading errors.")
else:
    print(f"\n--- Loading Model: {MODEL_NAME} ---")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        id2label={i: c for i, c in enumerate(mlb.classes_)},
        label2id={c: i for c, i in enumerate(mlb.classes_)} # Corrected label2id mapping
    )

    model.to(device)
    print(f"Model loaded: {MODEL_NAME} with {NUM_CLASSES} output classes for fine-tuning.")
    print(f"Model will be trained on: {device}")


# --- Define Training Arguments and Instantiate Trainer ---

if 'train_loader' not in globals() or 'val_loader' not in globals() or 'model' not in globals():
    print("ERROR: DataLoaders or model not initialized. Cannot set up Trainer.")
else:
    training_args = TrainingArguments(
      output_dir=OUTPUT_DIR,
      per_device_train_batch_size=BATCH_SIZE_PER_CORE,
      per_device_eval_batch_size=BATCH_SIZE_PER_CORE,
      evaluation_strategy="steps",
      num_train_epochs=NUM_EPOCHS,
      fp16=True, # TPU uses bfloat16 implicitly, this enables mixed precision
      save_steps=len(train_loader), # Save checkpoint every epoch (adjust if too frequent)
      eval_steps=len(train_loader), # Evaluate every epoch
      logging_steps=len(train_loader) // 10,
      learning_rate=LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
      save_total_limit=2,
      remove_unused_columns=False,
      push_to_hub=False,
      report_to='tensorboard',
      load_best_model_at_end=True,
      metric_for_best_model="avg_auroc",
      greater_is_better=True,
      gradient_accumulation_steps=1,
      dataloader_num_workers=NUM_WORKERS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_loader.dataset, # Pass the PyTorch Dataset directly
        eval_dataset=val_loader.dataset,    # Pass the PyTorch Dataset directly
    )

    print("Training arguments and Trainer instantiated.")


# --- Training Execution ---

if 'trainer' in globals():
    print("\n--- Starting Training ---")
    train_results = trainer.train()

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    print("\n--- Training Finished ---")

    # --- Plotting Training History (Saved to File) ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    
    train_steps = [log['step'] for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
    eval_steps = [log['step'] for log in trainer.state.log_history if 'eval_loss' in log]

    plt.plot(train_steps, train_losses, label='Train Loss')
    plt.plot(eval_steps, eval_losses, label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Training Steps')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    eval_auroc_history = [log['eval_avg_auroc'] for log in trainer.state.log_history if 'eval_avg_auroc' in log]
    plt.plot(eval_steps, eval_auroc_history, label='Validation Avg AUROC', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Avg AUROC')
    plt.title('Validation Average AUROC Over Steps')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics_plot.png"))
    print(f"Training metrics plot saved to {os.path.join(OUTPUT_DIR, 'training_metrics_plot.png')}")
    plt.close()

else:
    print("Trainer not instantiated. Skipping training execution.")


# --- Final Evaluation on Test Set ---

if 'trainer' in globals() and 'test_loader' in globals():
    print("\n--- Evaluating on Test Set ---")
    metrics = trainer.evaluate(test_loader.dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print(f"\nTest results saved to {os.path.join(OUTPUT_DIR, 'eval_results.json')}")
    print("\nTest AUROC per class:")
    test_auroc_per_class = {k.replace('eval_', ''): v for k, v in metrics.items() if k.startswith('eval_') and k not in ['eval_loss', 'eval_avg_auroc', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']}
    sorted_auroc = sorted(test_auroc_per_class.items(), key=lambda item: item[1] if not np.isnan(item[1]) else -1, reverse=True)

    for disease, score in sorted_auroc:
        print(f"  {disease}: {score:.4f}")

    print("\n--- Evaluation Finished ---")

else:
    print("Trainer or test dataset not available. Skipping test set evaluation.")