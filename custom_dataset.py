import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AntsBeesDataset(Dataset):
    def __init__(self, data_dir: str, split: str, transform=None):

        assert split in ["train", "val"], "Split must be 'train' or 'val'"

        self.dataset_dir = os.path.join(data_dir, split)
        self.classes = sorted(os.listdir(self.dataset_dir))  
        self.transform = transform

        self.images = []
        self.labels = []
        self.class_mapping = {label: idx for idx, label in enumerate(self.classes)}

        for label in self.classes:
            class_dir = os.path.join(self.dataset_dir, label)

            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.labels.append(self.class_mapping[label])  

    def __len__(self):
        return len(self.images)

    def get_mapping(self):
        return self.class_mapping

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")  

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label  
