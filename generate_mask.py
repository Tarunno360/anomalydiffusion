import argparse, os, sys, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class GeneratedDatasetMask(Dataset):
    def __init__(self, sample_name, anomaly_name, size=256):
        self.data_root = './generated_dataset'
        self.sample_name = sample_name
        self.anomaly_name = anomaly_name
        self.size = size
        self.data_path = os.path.join(self.data_root, sample_name, anomaly_name, 'mask')
        if not os.path.exists(self.data_path):
            raise ValueError(f"Path {self.data_path} does not exist")
        self.files = sorted(os.listdir(self.data_path))
        self._length = len(self.files)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        example = {}
        file_path = os.path.join(self.data_path, self.files[idx])
        image = Image.open(file_path).convert("L")  # Assuming masks are grayscale
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.float32) / 255.0
        example["image"] = image
        return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_name",
        default='capsule',
        help="Sample name",
    )
    parser.add_argument(
        "--anomaly_name",
        default='crack',
        help="Anomaly name",
    )


    opt = parser.parse_args()
    sample_name = opt.sample_name
    anomaly_name = opt.anomaly_name
    cnt = 0
    dataset = GeneratedDatasetMask(sample_name, anomaly_name)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
    save_dir = 'generated_mask/%s/%s' % (sample_name, anomaly_name)
    os.makedirs(save_dir, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        if cnt > 500:
            break
        images = batch['image']  # Assuming 'image' contains the masks
        for idx2, mask in enumerate(images):
            # Convert to tensor if not already
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            # Ensure mask is in correct shape (1, H, W)
            if mask.dim() == 3 and mask.shape[0] == 3:  # If RGB, take mean to grayscale
                mask = mask.mean(0).unsqueeze(0)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # Normalize to 0-1 if needed
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            # Threshold to binary
            mask = (mask > 0.5).float()
            save_image(mask, os.path.join(save_dir, '%d.jpg' % cnt))
            cnt += 1
