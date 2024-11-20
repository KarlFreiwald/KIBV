from PIL import Image
import os
import torch


class RobSceneSegDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sequences, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Root directory containing sequence folders.
            sequences (list): List of sequence folder names to include (e.g., ['seq_1', 'seq_2']).
            transform (callable, optional): Transform to apply to the input images.
            target_transform (callable, optional): Transform to apply to the labels.
        """
        self.root_dir = root_dir
        self.sequences = sequences
        self.transform = transform
        self.target_transform = target_transform
        self.image_label_pairs = self._load_image_label_pairs()

    def _load_image_label_pairs(self):
        image_label_pairs = []
        for sequence in self.sequences:
            images_dir = os.path.join(self.root_dir, sequence, 'images')
            processed_labels_dir = os.path.join(self.root_dir, sequence, 'processed_labels')

            if not os.path.exists(images_dir) or not os.path.exists(processed_labels_dir):
                print(f"Warning: {sequence} does not contain both 'images' and 'processed_labels' folders.")
                continue

            image_files = sorted([img for img in os.listdir(images_dir) if img.endswith('.png')])

            for img in image_files:
                img_path = os.path.join(images_dir, img)
                label_path = os.path.join(processed_labels_dir, img.replace('.png', '_class_id.png'))

                if os.path.exists(label_path):
                    image_label_pairs.append((img_path, label_path))
                else:
                    print(f"Warning: Label for {img} not found in {processed_labels_dir}.")

        return image_label_pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, index):
        img_path, label_path = self.image_label_pairs[index]

        # Load image and label
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # Load label as grayscale

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
