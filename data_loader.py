import torch
from torchvision import transforms
from utils import LetterboxResize
from collections import Counter
from PIL import Image

class HAM10000Dataset(torch.utils.data.Dataset):
    def __init__(self, data_indices, full_dataset, transform=None, label_mapping=None, is_training=True):
        """
        Args:
            data_indices: List of indices into the full dataset
            full_dataset: The complete HuggingFace dataset
            transform: Transform to apply to images
            label_mapping: Dictionary mapping label strings to indices
            is_training: Whether this is training or validation
        """
        self.indices = data_indices
        self.full_dataset = full_dataset
        self.transform = transform
        self.is_training = is_training

        if label_mapping is not None:
            self.label_to_idx = label_mapping
        else:
            unique_labels = sorted(list(set(full_dataset['label'])))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        self.labels = [self.label_to_idx[self.full_dataset[idx]['label']] for idx in self.indices]
        label_counts = Counter(self.labels)

        print(f"Dataset size: {len(self.indices)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Label mapping: {self.label_to_idx}")
        print(f"Class distribution:")
        for label, idx in self.label_to_idx.items():
            count = label_counts[idx]
            percentage = count/len(self.indices)*100 if len(self.indices) > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        item = self.full_dataset[actual_idx]
        image = item['image']
        label = self.label_to_idx[item['label']]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class MedicalImageAugmentation:
    """
    Conservative augmentation strategy for dermoscopic images.
    """
    @staticmethod
    def get_train_transforms(img_size=384):
        return transforms.Compose([
            LetterboxResize(img_size, fill_color=(255, 255, 255)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90, expand=False, fill=255),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.03
                )
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms(img_size=384):
        return transforms.Compose([
            LetterboxResize(img_size, fill_color=(255, 255, 255)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])