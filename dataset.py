from torch.utils.data import Dataset
import os
import random
from PIL import Image
from network import preprocess_image


class ImageDataset(Dataset):
    def __init__(self, image_folder, df, fit_method, num_cutouts=0, is_clip=True):
        self.image_paths = df["image_path"].tolist()
        self.fit_method = fit_method
        self.image_paths = [os.path.join(image_folder, p) for p in self.image_paths]

        # self.image_paths = [p for p in self.image_paths if p.endswith(".jpg") or p.endswith(".png") or p.endswith(".jpeg")]
        # self.image_paths = [p for p in self.image_paths if os.path.getsize(p) > 5000]

        self.length = len(self.image_paths)
        self.processor = preprocess_image
        self.num_cutouts = num_cutouts
        self.is_clip = is_clip

        random.shuffle(self.image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        img_tensor = self.processor(img, self.fit_method, num_cutouts=self.num_cutouts, is_clip=self.is_clip)

        example = {"img": img_tensor, "img_path": image_path}
        return example