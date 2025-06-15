import os
import torch
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from datasets import load_dataset, DatasetDict
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    Trainer,
    TrainingArguments,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    CenterCrop,
    ToTensor,
)
from sklearn.metrics import f1_score, roc_auc_score, classification_report

# --- This function will contain all the logic that runs on each TPU core ---
def _mp_fn(index, flags):
    """
    The main training function that gets spawned on each TPU core.
    'index' is the process rank, from 0 to 7.
    'flags' is a dictionary containing our configuration.
    """
    print(f"--> Starting process with rank {index}...")
    
    # --- 1. Data Loading and Preparation ---
    # The master process (rank 0) downloads the data. Others wait.
    if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')
    
    dataset = load_dataset(flags['dataset_name'])
    train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    test_val_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": test_val_split["train"],
        "test": test_val_split["test"],
    })
    
    if xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    class_names = dataset["train"].features['labels'].feature.names
    num_classes = len(class_names)
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in id2label.items()}

    # --- 2. Image Processing ---
    processor = ViTImageProcessor.from_pretrained(flags['model_name'])
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)

    train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def apply_train_transforms(examples):
        examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["image"]]
        return examples

    def apply_val_transforms(examples):
        examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["image"]]
        return examples
    
    dataset['train'] = dataset['train'].with_transform(apply_train_transforms)
    dataset['validation'] = dataset['validation'].with_transform(apply_val_transforms)
    dataset['test'] = dataset['test'].with_transform(apply_val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples], dtype=torch.float)
        return {"pixel_values": pixel_values, "labels": labels}

    # --- 3. Model and Training Configuration ---
    model = ViTForImageClassification.from_pretrained(
        flags['model_name'],
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=flags['output_dir'],
        per_device_train_batch_size=flags['train_batch_size'],
        per_device_eval_batch_size=flags['eval_batch_size'],
        num_train_epochs=flags['num_train_epochs'],
        learning_rate=flags['learning_rate'],
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        bf16=True,
        tpu_num_cores=flags['num_workers'],
        seed=42,
        remove_unused_columns=True,
        report_to="none",
    )
    
    def compute_metrics(p):
        logits, labels = p.predictions, p.label_ids
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        predictions = (probs >= 0.5).int().cpu().numpy()
        f1_micro = f1_score(y_true=labels, y_pred=predictions, average="micro", zero_division=0)
        return {"f1_micro": f1_micro}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        processing_class=processor,
    )

    # --- 4. Start Training ---
    print(f"Rank {index}: Starting training...")
    trainer.train()
    
    # --- 5. Evaluate and Save (on main process only) ---
    if xm.is_master_ordinal():
        print("--- Master process: Evaluating on test set ---")
        test_results = trainer.predict(dataset["test"])
        
        logits = test_results.predictions
        y_true = test_results.label_ids
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        y_pred = (probs >= 0.5).int().cpu().numpy()

        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        
        with open(os.path.join(flags['output_dir'], "final_classification_report.txt"), "w") as f:
            f.write(report)
        print(f"Final report saved to {flags['output_dir']}/final_classification_report.txt")
        
    print(f"--- Rank {index}: Finished ---")


# --- This is the main entry point of the script ---
if __name__ == '__main__':
    # Configuration flags
    config = {
        "model_name": "google/vit-base-patch16-384",
        "dataset_name": "kerem/nih-chest-xray-14",
        "output_dir": "./nih-xray-vit-programmatic-finetuned",
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "num_train_epochs": 5,
        "learning_rate": 2e-5,
        "num_workers": 8, # Number of TPU cores
    }
    
    # This is the programmatic spawn call you found in the documentation.
    # It will start the `_mp_fn` function on `num_workers` different processes.
    xmp.spawn(_mp_fn, args=(config,), nprocs=None, start_method='fork')
