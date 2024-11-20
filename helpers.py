import os
from PIL import Image
import json
import numpy as np


def load_color_mapping(json_file):
    with open(json_file) as f:
        color_mapping = json.load(f)

    color_dict = {tuple(entry["color"]): entry["classid"] for entry in color_mapping}
    return color_dict


def color_to_label(image, color_mapping):
    label_image = np.zeros((image.height, image.width), dtype=np.uint8)

    for color, class_id in color_mapping.items():
        mask = np.all(np.array(image)[:, :, :3] == color, axis=-1)
        label_image[mask] = class_id

    return Image.fromarray(label_image)


def label_to_color(label_image, color_mapping):
    color_image = np.zeros((label_image.height, label_image.width, 3), dtype=np.uint8)

    for color, class_id in color_mapping.items():
        mask = np.array(label_image) == class_id
        color_image[mask] = np.array(color)

    return Image.fromarray(color_image)


def save_transformed_images(sequences, color_mapping, directory):
    """Processes images in each sequence, transforms, and saves them to the output directory."""
    for sequence in sequences:
        images_path = os.path.join(directory, sequence, 'images')
        output_sequence_path = os.path.join(directory, sequence, 'processed_labels')

        # Ensure processed_labels directory exists
        os.makedirs(output_sequence_path, exist_ok=True)

        if os.path.exists(images_path):
            image_files = [img for img in os.listdir(images_path) if img.endswith('.png')]
            for img in image_files:
                img_path = os.path.join(images_path, img)

                with Image.open(img_path) as color_image:
                    # Convert color image to classid map
                    label_image = color_to_label(color_image, color_mapping)

                    # Save the processed image in the 'processed_labels' folder
                    output_img_path = os.path.join(output_sequence_path, img.replace('.png', '_class_id.png'))
                    label_image.save(output_img_path)