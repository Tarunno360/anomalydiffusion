import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model

def make_batch(image_path, mask_path, device):
    image = np.array(Image.open(image_path).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(device)*2-1
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    print(image.shape, mask.shape)
    batch = {"image": image, "mask": mask}
    return batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--mask_path",
        required=True,
        help="Path to the mask image"
    )
    parser.add_argument(
        "--output_dir",
        default="anomaly_output",
        help="Directory to save the output"
    )
    parser.add_argument(
        "--sample_name",
        default='sample',
    )
    parser.add_argument(
        "--anomaly_name",
        default='anomaly',
    )
    parser.add_argument(
        "--adaptive_mask",
        action="store_true", default=False,
        help='whether use adaptive attention reweighting',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    opt = parser.parse_args()
    setup_seed(opt.seed)
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    actual_resume = './models/ldm/text2img-large/model.ckpt'
    model = load_model_from_config(config, actual_resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # Create batch from single image and mask
    batch = make_batch(opt.image_path, opt.mask_path, device)

    save_dir = os.path.join(opt.output_dir, opt.sample_name, opt.anomaly_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ori'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'recon'), exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            ori_images = batch['image'].permute(0, 3, 1, 2).cpu()
            images = model.log_images(batch, sample=False, inpaint=True, unconditional_only=True, adaptive_mask=opt.adaptive_mask)
            imgs = images['samples_inpainting'].cpu()
            recon_image = images['reconstruction'].cpu()
            mask = batch['mask'].cpu()

            # Save for the single sample
            save_image((imgs[0] + 1) / 2, os.path.join(save_dir, 'image', '0.jpg'), normalize=False)
            save_image((ori_images[0] + 1) / 2, os.path.join(save_dir, 'ori', '0.jpg'), normalize=False)
            save_image((recon_image[0] + 1) / 2, os.path.join(save_dir, 'recon', '0.jpg'), normalize=False)
            save_image(mask[0], os.path.join(save_dir, 'mask', '0.jpg'))

    print(f"Anomaly image generated and saved in {save_dir}")
