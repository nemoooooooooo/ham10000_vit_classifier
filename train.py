import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from datasets import load_dataset

import wandb
from tqdm import tqdm
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

from utils import set_seed, stratified_split, calculate_class_balanced_weights
from data_loader import HAM10000Dataset, MedicalImageAugmentation
from model import ViTClassifier

warnings.filterwarnings('ignore')




class Trainer:
    def __init__(self, model, device, num_classes, class_names):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names  # e.g., ['Actinic ...', 'Basal ...', ..., 'Melanoma', ...]
        self.best_val_acc = 0
        self.patience_counter = 0

        # Try to find the melanoma class index if present
        self.melanoma_label = None
        for name in self.class_names:
            if name.lower() == "melanoma":
                self.melanoma_label = name
                break

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = balanced_accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc, all_preds, all_labels

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = running_loss / len(dataloader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Per-class AUROC
        all_labels_bin = label_binarize(all_labels, classes=list(range(self.num_classes)))
        auroc_scores = []
        for i in range(self.num_classes):
            try:
                auroc = roc_auc_score(all_labels_bin[:, i], np.array(all_probs)[:, i])
                auroc_scores.append(auroc)
            except:
                auroc_scores.append(0.0)
        avg_auroc = np.mean(auroc_scores)

        # Per-class precision/recall/F1 using class names
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            digits=4,
            output_dict=True,
            zero_division=0
        )
        # Build clean per-class metrics dict
        per_class = {}
        for cname in self.class_names:
            if cname in report:
                per_class[cname] = {
                    "precision": float(report[cname]["precision"]),
                    "recall": float(report[cname]["recall"]),
                    "f1": float(report[cname]["f1-score"]),
                    "support": int(report[cname]["support"])
                }

        return {
            'loss': val_loss,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'auroc': avg_auroc,
            'auroc_per_class': auroc_scores,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'per_class': per_class  # <— added
        }

    def save_checkpoint(self, epoch, optimizer, scheduler, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_acc': self.best_val_acc,
        }, path)

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        return checkpoint['epoch']



def train_model(config):
    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if config['use_wandb']:
        wandb.init(
            project="ham10000-vit",
            name=config['experiment_name'],
            config=config
        )

    print("\n" + "="*50)
    print("Loading HAM10000 dataset from HuggingFace...")
    print("="*50)

    dataset = load_dataset("BoooomNing/ham10000", split="train")

    unique_labels = sorted(list(set(dataset['label'])))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"\nConsistent label mapping across all splits:")
    for label, idx in label_mapping.items():
        print(f"  {label}: {idx}")

    # Manually create stratified split
    print("\nCreating stratified train/val split...")
    train_indices, val_indices = stratified_split(
        dataset,
        test_size=0.1,
        seed=config['seed']
    )

    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")

    train_dataset = HAM10000Dataset(
        train_indices,
        dataset,
        transform=MedicalImageAugmentation.get_train_transforms(config['img_size']),
        label_mapping=label_mapping,
        is_training=True
    )

    val_dataset = HAM10000Dataset(
        val_indices,
        dataset,
        transform=MedicalImageAugmentation.get_val_transforms(config['img_size']),
        label_mapping=label_mapping,
        is_training=False
    )

    # Class names in index order (for per-class logs)
    class_names = [train_dataset.idx_to_label[i] for i in range(train_dataset.num_classes)]

    print("\n" + "="*50)
    print("Calculating class-balanced weights...")
    print("="*50)

    class_weights = calculate_class_balanced_weights(train_dataset.labels, beta=config['cb_beta'])
    print(f"Class-balanced weights (beta={config['cb_beta']}):")
    for idx, weight in enumerate(class_weights):
        print(f"  {train_dataset.idx_to_label[idx]}: {weight:.3f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    print("\n" + "="*50)
    print(f"Initializing {config['model_name']} model...")
    print("="*50)

    model = ViTClassifier(
        num_classes=train_dataset.num_classes,
        model_name=config['model_name'],
        pretrained=True,
        dropout=config['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---- Trainer now knows class names for per-class reporting ----
    trainer = Trainer(model, device, train_dataset.num_classes, class_names)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print("\n" + "="*50)
    print("STAGE 1: Linear Probing (Frozen Backbone)")
    print("="*50)

    model.freeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Stage 1): {trainable_params:,}")

    optimizer_stage1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr_stage1'],
        weight_decay=config['weight_decay']
    )

    scheduler_stage1 = CosineAnnealingLR(
        optimizer_stage1,
        T_max=config['epochs_stage1'],
        eta_min=config['lr_stage1'] * 0.01
    )

    best_val_metrics = None
    stage1_history = []

    # Find melanoma name (exact match) if present
    melanoma_name = None
    for n in class_names:
        if n.lower() == "melanoma":
            melanoma_name = n
            break

    for epoch in range(config['epochs_stage1']):
        print(f"\nEpoch {epoch+1}/{config['epochs_stage1']} (Stage 1)")
        print("-" * 40)

        train_loss, train_acc, train_preds, train_labels = trainer.train_epoch(
            train_loader, optimizer_stage1, criterion
        )
        val_metrics = trainer.evaluate(val_loader)

        scheduler_stage1.step()

        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # ---- Log melanoma-specific metrics if available ----
        if melanoma_name and melanoma_name in val_metrics['per_class']:
            mel_rec = val_metrics['per_class'][melanoma_name]['recall']
            mel_f1  = val_metrics['per_class'][melanoma_name]['f1']
            print(f"Melanoma — Recall: {mel_rec:.4f}, F1: {mel_f1:.4f}")

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['balanced_accuracy']:.4f}, "
              f"F1: {val_metrics['f1_score']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        print(f"LR: {optimizer_stage1.param_groups[0]['lr']:.6f}")

        stage1_history.append({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['balanced_accuracy']
        })

        if config['use_wandb']:
            log_dict = {
                'stage': 1,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_metrics['loss'],
                'val_balanced_acc': val_metrics['balanced_accuracy'],
                'val_f1': val_metrics['f1_score'],
                'val_auroc': val_metrics['auroc'],
                'lr': optimizer_stage1.param_groups[0]['lr']
            }
            if melanoma_name and melanoma_name in val_metrics['per_class']:
                log_dict['melanoma_recall'] = val_metrics['per_class'][melanoma_name]['recall']
                log_dict['melanoma_f1'] = val_metrics['per_class'][melanoma_name]['f1']
            wandb.log(log_dict)

        if val_metrics['balanced_accuracy'] > trainer.best_val_acc:
            trainer.best_val_acc = val_metrics['balanced_accuracy']
            best_val_metrics = val_metrics
            trainer.save_checkpoint(epoch, optimizer_stage1, scheduler_stage1, 'best_model_stage1.pth')
            trainer.patience_counter = 0
            print("✓ Best model saved!")
        else:
            trainer.patience_counter += 1

    print("\n" + "="*50)
    print("STAGE 2: Fine-tuning (Full Model)")
    print("="*50)

    trainer.load_checkpoint('best_model_stage1.pth')

    model.unfreeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Stage 2): {trainable_params:,}")

    optimizer_stage2 = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['lr_stage2']},
        {'params': model.head.parameters(), 'lr': config['lr_stage2'] * 10}
    ], weight_decay=config['weight_decay'])

    scheduler_stage2 = CosineAnnealingLR(
        optimizer_stage2,
        T_max=config['epochs_stage2'],
        eta_min=config['lr_stage2'] * 0.01
    )

    trainer.patience_counter = 0
    stage2_history = []

    for epoch in range(config['epochs_stage2']):
        print(f"\nEpoch {epoch+1}/{config['epochs_stage2']} (Stage 2)")
        print("-" * 40)

        train_loss, train_acc, train_preds, train_labels = trainer.train_epoch(
            train_loader, optimizer_stage2, criterion
        )
        val_metrics = trainer.evaluate(val_loader)

        scheduler_stage2.step()
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # ---- Melanoma metrics again ----
        if melanoma_name and melanoma_name in val_metrics['per_class']:
            mel_rec = val_metrics['per_class'][melanoma_name]['recall']
            mel_f1  = val_metrics['per_class'][melanoma_name]['f1']
            print(f"Melanoma — Recall: {mel_rec:.4f}, F1: {mel_f1:.4f}")

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['balanced_accuracy']:.4f}, "
              f"F1: {val_metrics['f1_score']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        print(f"LR (backbone): {optimizer_stage2.param_groups[0]['lr']:.6f}")
        print(f"LR (head): {optimizer_stage2.param_groups[1]['lr']:.6f}")

        stage2_history.append({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['balanced_accuracy']
        })

        if config['use_wandb']:
            log_dict = {
                'stage': 2,
                'epoch': epoch + config['epochs_stage1'],
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_metrics['loss'],
                'val_balanced_acc': val_metrics['balanced_accuracy'],
                'val_f1': val_metrics['f1_score'],
                'val_auroc': val_metrics['auroc'],
                'lr_backbone': optimizer_stage2.param_groups[0]['lr'],
                'lr_head': optimizer_stage2.param_groups[1]['lr']
            }
            if melanoma_name and melanoma_name in val_metrics['per_class']:
                log_dict['melanoma_recall'] = val_metrics['per_class'][melanoma_name]['recall']
                log_dict['melanoma_f1'] = val_metrics['per_class'][melanoma_name]['f1']
            wandb.log(log_dict)

        if val_metrics['balanced_accuracy'] > trainer.best_val_acc:
            trainer.best_val_acc = val_metrics['balanced_accuracy']
            best_val_metrics = val_metrics
            trainer.save_checkpoint(epoch, optimizer_stage2, scheduler_stage2, 'best_model_stage2.pth')
            trainer.patience_counter = 0
            print("✓ Best model saved!")
        else:
            trainer.patience_counter += 1

        if trainer.patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs in Stage 2")
            break

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)

    trainer.load_checkpoint('best_model_stage2.pth')
    final_metrics = trainer.evaluate(val_loader)

    print(f"\nBest Validation Performance:")
    print(f"  Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Score (Macro): {final_metrics['f1_score']:.4f}")
    print(f"  Average AUROC: {final_metrics['auroc']:.4f}")

    # Per-class AUROC
    print("\nPer-class AUROC:")
    for i, auroc in enumerate(final_metrics['auroc_per_class']):
        print(f"  {train_dataset.idx_to_label[i]}: {auroc:.4f}")

    # Per-class precision/recall/F1
    print("\nPer-class precision/recall/F1:")
    for cname in class_names:
        if cname in final_metrics['per_class']:
            m = final_metrics['per_class'][cname]
            print(f"  {cname:30s}  P:{m['precision']:.4f}  R:{m['recall']:.4f}  F1:{m['f1']:.4f}  (n={m['support']})")

    print("\nClassification Report:")
    print(classification_report(
        final_metrics['labels'],
        final_metrics['predictions'],
        target_names=list(train_dataset.label_to_idx.keys()),
        digits=4
    ))

    plot_confusion_matrix(
        final_metrics['labels'],
        final_metrics['predictions'],
        list(train_dataset.label_to_idx.keys())
    )

    plot_training_history(stage1_history, stage2_history, config['epochs_stage1'])

    if config['use_wandb']:
        wandb.finish()

    return final_metrics

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="ham10000_vit_optimal_v1")
    parser.add_argument("--model_name", default="vit_base_patch16_384")
    parser.add_argument("--img_size", type=int, default=384)

    parser.add_argument("--epochs_stage1", type=int, default=10)  # linear probe
    parser.add_argument("--epochs_stage2", type=int, default=20)  # finetune

    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.30)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cb_beta", type=float, default=0.9999)
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    config = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "img_size": args.img_size,
        "epochs_stage1": args.epochs_stage1,
        "epochs_stage2": args.epochs_stage2,

        "lr_stage1": args.lr_stage1,
        "lr_stage2": args.lr_stage2,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,

        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,

        "cb_beta": args.cb_beta,
        "use_wandb": args.use_wandb,
        "patience": 15,
    }

    train_model(config)

if __name__ == "__main__":
    main()
