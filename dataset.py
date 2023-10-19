from torch.utils.data import Dataset
import os
import random
from PIL import Image
from network import preprocess_image
import torch


class ImageDataset(Dataset):
    def __init__(self, image_folder, df, fit_method, num_cutouts=0, is_clip=True, image_col_name="IAPS", color_jitter=0.1):
        self.image_paths = df[f"{image_col_name}"].tolist()
        self.image_paths = [str(p).replace(".0","") + ".jpg" for p in self.image_paths]
        self.image_paths = [os.path.join(image_folder, p) for p in self.image_paths]
        self.fit_method = fit_method

        self.valence_scores = [float(val) for val in df["valmn"].tolist()]
        self.arousal_scores = [float(val) for val in df["aromn"].tolist()]
        # self.dominance_scores = [float(val) for val in df["dom1mn"].tolist()]
        # self.dominance_scores2 = [float(val) for val in df["dom2mn"].tolist()]


        # self.image_paths = [p for p in self.image_paths if p.endswith(".jpg") or p.endswith(".png") or p.endswith(".jpeg")]
        # self.image_paths = [p for p in self.image_paths if os.path.getsize(p) > 5000]

        self.length = len(self.image_paths)
        self.processor = preprocess_image
        self.num_cutouts = num_cutouts
        self.is_clip = is_clip
        self.color_jitter = color_jitter

        random.shuffle(self.image_paths)

    def __len__(self):
        return self.length

    def try_to_get_image(self, idx):
        while True:
            try:
                return Image.open(self.image_paths[idx]).convert("RGBA").convert("RGB"), idx
            except Exception as e:
                print(e)
                idx = torch.randint(0, len(self.image_paths), (1,)).item()

    def __getitem__(self, idx):
        img, idx = self.try_to_get_image(idx)
        img_tensor = self.processor(img, self.fit_method, num_cutouts=self.num_cutouts, is_clip=self.is_clip, color_jitter=self.color_jitter)

        valence_score = self.valence_scores[idx]
        arousal_score = self.arousal_scores[idx]
        # dominance_score = self.dominance_scores[idx]
        # dominance_score2 = self.dominance_scores2[idx]

        #scores = torch.tensor([valence_score, arousal_score, dominance_score, dominance_score2])
        scores = torch.tensor([valence_score, arousal_score])

        example = {"pixel_values": img_tensor, "score": scores}
        return example