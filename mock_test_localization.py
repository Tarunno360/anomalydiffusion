import torch
from unet_utils.data_loader import MVTecDRAEMTestDataset_partial
from torch.utils.data import DataLoader
import numpy as np
from unet_utils.model_unet import DiscriminativeSubNetwork
import os
import cv2
from unet_utils.au_pro_util import calculate_au_pro
from sklearn.metrics import roc_auc_score, average_precision_score

# Simple mock test for anomaly localization on leather
def mock_test():
    # Paths
    mvtec_path = "C:/Users/tarun/AnnomalyDiffusion/testingdata"
    checkpoint_path = "checkpoints/localization/leather.pckl"
    sample_name = "leather"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    img_dim = 256
    model_seg = DiscriminativeSubNetwork(in_channels=3, out_channels=2)
    model_seg.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()

    dataset = MVTecDRAEMTestDataset_partial(mvtec_path + '/' + sample_name + "/test/", resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    gt_masks = []
    predicted_masks = []

    print("Testing anomaly localization on all leather examples...")

    for i_batch, sample_batched in enumerate(dataloader):
        try:
            gray_batch = sample_batched["image"].cuda()
            gray_batch = gray_batch[:, [2, 1, 0], :, :]
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]

            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
            out_mask = model_seg(gray_batch)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)
            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()

            gt_masks.append(true_mask_cv.squeeze())
            predicted_masks.append(out_mask_cv.squeeze())

            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1
        except Exception as e:
            print(f"Error processing batch {i_batch}: {e}")
            continue

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc_image = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap_image = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    # Compute AUROC and AP for pixel-level
    if len(set(total_gt_pixel_scores)) > 1:
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        print(f"Pixel-level AUROC: {auroc_pixel:.4f}")
    else:
        print("Pixel-level AUROC: Not defined (only one class)")

    if len(set(total_gt_pixel_scores)) > 1:
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        print(f"Pixel-level AP: {ap_pixel:.4f}")
    else:
        print("Pixel-level AP: Not defined (only one class)")

    # Compute PRO curve and AU-PRO
    try:
        pro_pixel, _ = calculate_au_pro(gt_masks, predicted_masks)
        print(f"Pixel-level PRO: {pro_pixel:.4f}")
    except Exception as e:
        print(f"Pixel-level PRO: Failed to compute ({e})")

    print(f"Image-level AUROC: {auroc_image:.4f}")
    print(f"Image-level AP: {ap_image:.4f}")

    print("Mock localization test completed.")

if __name__ == "__main__":
    mock_test()
