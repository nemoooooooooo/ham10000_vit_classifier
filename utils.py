import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from torchvision import transforms

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(stage1_history, stage2_history, stage1_epochs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = list(range(1, len(stage1_history) + len(stage2_history) + 1))
    train_loss = [h['train_loss'] for h in stage1_history + stage2_history]
    val_loss = [h['val_loss'] for h in stage1_history + stage2_history]
    train_acc = [h['train_acc'] for h in stage1_history + stage2_history]
    val_acc = [h['val_acc'] for h in stage1_history + stage2_history]

    axes[0].plot(epochs, train_loss, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_loss, label='Val Loss', marker='s')
    axes[0].axvline(x=stage1_epochs + 0.5, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label='Train Acc', marker='o')
    axes[1].plot(epochs, val_acc, label='Val Acc', marker='s')
    axes[1].axvline(x=stage1_epochs + 0.5, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


class LetterboxResize:
    """
    Resize image to target size while maintaining aspect ratio using letterboxing.
    """

    def __init__(self, size, fill_color=(255, 255, 255)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        target_h, target_w = self.size

        scale = min(target_w / w, target_h / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = transforms.functional.resize(
            img,
            (new_h, new_w),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top

        img = transforms.functional.pad(
            img,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill_color
        )

        return img



def calculate_class_balanced_weights(labels, beta=0.9999):
    label_counts = Counter(labels)
    num_classes = len(label_counts)

    effective_num = {}
    for class_idx, count in label_counts.items():
        effective_num[class_idx] = (1 - beta ** count) / (1 - beta)

    weights = []
    for class_idx in range(num_classes):
        weight = 1.0 / effective_num[class_idx] if class_idx in effective_num else 1.0
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    return weights


def stratified_split(dataset, test_size=0.1, seed=42):
    labels = [item['label'] for item in dataset]
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    return train_indices, val_indices