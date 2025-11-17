import os
import glob
import torch
import cv2
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from pytorch_msssim import ms_ssim
import lpips
import piq
from tqdm import tqdm
# -------------------------
# 自定义 LOE (Lightness Order Error)
# -------------------------
def compute_loe(enhanced, gt):
    # 转灰度
    en_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # 光照顺序差
    diff = np.abs(np.sort(en_gray.flatten()) - np.sort(gt_gray.flatten()))
    return np.mean(diff)
# -------------------------
# 图像预处理
# -------------------------
def load_image(path, size = None):
    img = cv2.imread(path)[:, :, ::-1]
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]))
    # BGR->RG
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [C,H,W], [0,1]
    return img
# -------------------------
# 批量评

def evaluate_metrics(gt_dir, en_dir, save_path="results.txt", device="cuda:0"):
    # LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    exts = ("*.png","*.PNG","*.jpg","*.jpeg","*.bmp","*.tif", "*.tiff")
    gt_files = []
    en_files = []
    for ext in exts:
        gt_files += glob.glob(os.path.join(gt_dir, ext))
        en_files += glob.glob(os.path.join(en_dir, ext))
    # 收集文件
    gt_files = sorted(gt_files)
    en_files = sorted(en_files)
    assert len(gt_files) == len(en_files), "GT 和 Enhanced 数量不匹配！"
    metrics = {
        "PSNR": [],
        "SSIM": [],
        "MS-SSIM": [],
        "LPIPS": [],
        "brisque": [],
        "LOE": []
    }

    batch_size = 15
    batch_metrics = {k: [] for k in ["PSNR", "SSIM", "MS-SSIM", "LPIPS", "brisque", "LOE"]}
    with open(save_path, 'w') as f:
        for idx, (gt_path, en_path) in tqdm(enumerate(zip(gt_files, en_files)), total=len(gt_files)):
            gt_raw = cv2.imread(gt_path)[:, :, ::-1].copy() # RGB
            h, w = gt_raw.shape[:2]
            en_raw = cv2.imread(en_path)[:, :, ::-1].copy()
            en_raw = cv2.resize(en_raw, (w, h))
            # 转成 tensor
            gt_img = torch.from_numpy(gt_raw).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            en_img = torch.from_numpy(en_raw).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            # numpy 版本
            gt_np = gt_raw
            en_np = en_raw
            # ---- 全参考指标 ----
            psnr_val = psnr(en_img, gt_img).item()
            ssim_val = ssim(en_img, gt_img).item()
            ms_ssim_val = ms_ssim(en_img, gt_img, data_range=1.0, size_average=True).item()
            lpips_val = lpips_model(en_img * 2 - 1, gt_img * 2 - 1).item()
            # ---- 无参考指标 ----
            brisque_metric = piq.brisque
            brisque_val = brisque_metric(en_img, data_range=1.0).item()
            loe_val = compute_loe(en_np, gt_np)

            metrics["PSNR"].append(psnr_val)
            metrics["SSIM"].append(ssim_val)
            metrics["MS-SSIM"].append(ms_ssim_val)
            metrics["LPIPS"].append(lpips_val)
            metrics["brisque"].append(brisque_val)
            metrics["LOE"].append(loe_val)

            for k, v in zip(batch_metrics.keys(), [psnr_val, ssim_val, ms_ssim_val, lpips_val, brisque_val, loe_val]):
                batch_metrics[k].append(v)
                metrics[k].append(v)

            # 写入单张结果
            fname = os.path.basename(gt_path)
            f.write(f"{fname}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, "
                    f"MS-SSIM={ms_ssim_val:.4f}, LPIPS={lpips_val:.4f}, "
                    f"brisque={brisque_val:.4f}, LOE={loe_val:.4f}, ")
            # 平均结果
            if (idx + 1 ) % batch_size == 0:
                f.write(f"\n === Average of first {idx+2-batch_size} to {idx+1} ---\n")
                for k, v in batch_metrics.items():
                    f.write(f"{k}: {np.mean(v):.4f}\n")
                f.write("\n")

                batch_metrics = {k: [] for k in batch_metrics}

        f.write("\n === Average Results final ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {np.mean(v):.4f}\n")
            
    print("评测完成，结果保存在:", save_path)
    return {k: np.mean(v) for k, v in metrics.items()}



# -------------------------
# 使用示例
# -------------------------
if __name__ == "__main__":
    gt_dir = "/media/media/a93cbdc8-0ee1-4e69-bf9c-9ecbf5625ec9/liulu/LLIE/LOLdataset/eval15/high"  # GT 图像文件夹
    en_dir = "/media/media/a93cbdc8-0ee1-4e69-bf9c-9ecbf5625ec9/liulu/LLIE/results/test7"  # 增强后图像文件夹
    evaluate_metrics(gt_dir, en_dir, save_path="/media/media/a93cbdc8-0ee1-4e69-bf9c-9ecbf5625ec9/liulu/LLIE/results/test7/metrics_results.txt")

