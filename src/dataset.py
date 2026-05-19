import os
import cv2
import torch

from torch.utils.data import Dataset


class XrayDataset(Dataset):

    def __init__(self, root_dir):

        self.image_paths = []
        self.labels = []

        classes = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }

        # Read all folders
        for class_name, label in classes.items():

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            for image_name in os.listdir(class_dir):

                image_path = os.path.join(
                    class_dir,
                    image_name
                )

                self.image_paths.append(image_path)

                self.labels.append(label)

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        img = cv2.imread(img_path)

        # Handle grayscale safely
        if len(img.shape) == 2:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2RGB
            )

        else:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

        # Resize
        img = cv2.resize(img, (224, 224))

        # Normalize
        img = img.astype("float32") / 255.0

        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        label = self.labels[idx]

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )