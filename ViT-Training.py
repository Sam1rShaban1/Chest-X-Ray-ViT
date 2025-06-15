import os
import torch
import numpy as np
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

def main():
    # --- 1. Configuration: All settings in one place ---
    
    # Model and Dataset
    MODEL_NAME = "google/vit-base-patch16-384"
    DATASET_NAME = "kerem/nih-chest-xray-14"
    OUTPUT_DIR = "./nih-xray-vit-base-384-finetuned"
    
    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    NUM_TRAIN_EPOCHS = 5  # Increase for better performance, e.g., to 10
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Dataloader optimization for powerful host VM
    DATALOADER_NUM_WORKERS = os.cpu_count() // 2
    
    # --- 2. Data Loading and Preparation ---
    print("--- Loading and preparing dataset ---")
    
    dataset = load_dataset(DATASET_NAME)
    train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    test_val_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": test_val_split["train"],
        "test": test_val_split["test"],
    })
    
    class_names = dataset["train"].features['labels'].feature.names
    num_classes = len(class_names)
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in id2label.items()}

    # --- 3. Image Processing ---
    print("--- Setting up image processor and transforms ---")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)

    train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

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

    # --- 4. Model and Training Configuration ---
    print("--- Configuring model and training arguments ---")

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        bf16=True, # Natively supported and faster on TPU v3+
        tpu_num_cores=8,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
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
        roc_auc_micro = roc_auc_score(y_true=labels, y_score=probs.cpu().numpy(), average="micro")
        
        return {"f1_micro": f1_micro, "roc_auc_micro": roc_auc_micro}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        # Validated against docs: `processing_class` is the correct, modern argument
        processing_class=processor,
    )

    # --- 5. Start Training ---
    print("\n--- Starting model training on TPU v4 ---")
    
    # Using resume_from_checkpoint=True ensures fault tolerance.
    # If the script is interrupted, it will automatically resume from the last save.
    train_results = trainer.train(resume_from_checkpoint=True)
    
    # --- 6. Save, Evaluate, and Report (on main process only) ---
    if trainer.is_world_process_zero():
        print("\n--- Saving final model and metrics ---")
        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)

        print("\n--- Evaluating on test set ---")
        test_results = trainer.predict(dataset["test"])
        trainer.log_metrics("test", test_results.metrics)
        trainer.save_metrics("test", test_results.metrics)

        # Generate and save final classification report
        logits = test_results.predictions
        y_true = test_results.label_ids
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        y_pred = (probs >= 0.5).int().cpu().numpy()

        print("\n--- Final Classification Report ---")
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        print(report)
        
        with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
            f.write("Test Metrics:\n")
            for key, value in test_results.metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\nClassification Report:\n")
            f.write(report)
            
        print(f"\n--- Training complete. Results saved in: {OUTPUT_DIR} ---")


if __name__ == "__main__":
    main()
