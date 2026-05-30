import os
import argparse
import logging
from importlib import import_module
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

from dataloader import MyDataset
from model.model import Model
from segment_anything.build_sam import build_sam_vit_h

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(work_dir):
    log_file = os.path.join(work_dir, f'test_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Model Testing Script")

    parser.add_argument("--work_dir", type=str, required=True, 
                        help="Path to model checkpoints directory")
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Path to test data")
    parser.add_argument("--vmap_root", type=str, default=None, 
                        help="Path to vmap data")
    parser.add_argument("--sam_checkpoint", type=str, 
                        default="sam_vit_h_4b8939.pth",
                        help="Path to SAM checkpoint")
    
    parser.add_argument("--model_type", type=str, default="default", 
                        help="Model type for building aggregation model")
    parser.add_argument("--vmap_type", type=str, default="default", 
                        help="Type of visual map")
    parser.add_argument("--lora_rank", type=int, default=4, 
                        help="LoRA rank")
    
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Test batch size")
    parser.add_argument("--image_size", type=int, default=1024, 
                        help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for dataloader")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="Device to use")
    
    parser.add_argument("--nsd_tau", type=float, default=2.0, 
                        help="Tolerance distance for NSD calculation")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Threshold for binary prediction")
    
    parser.add_argument("--start_epoch", type=int, default=1, 
                        help="Start epoch for testing")
    parser.add_argument("--end_epoch", type=int, default=100, 
                        help="End epoch for testing")
    parser.add_argument("--step", type=int, default=1, 
                        help="Step size between epochs")
    
    parser.add_argument("--save_predictions", action="store_true", 
                        help="Save prediction masks")
    parser.add_argument("--save_dir", type=str, default="predictions", 
                        help="Directory to save predictions")
    
    args = parser.parse_args()
    
    if args.vmap_root is None:
        args.vmap_root = args.data_root
        
    return args


class MetricsCalculator:
    
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth
    
    @staticmethod
    def to_binary(pred, threshold=0.5):
        if pred.max() > 1.0:
            pred = torch.sigmoid(pred)
        return (pred > threshold).float()
    
    def calculate_dice_iou(self, pred, mask):
        pred_binary = self.to_binary(pred, self.threshold)
        
        intersection = torch.sum(pred_binary * mask)
        pred_sum = torch.sum(pred_binary)
        mask_sum = torch.sum(mask)
        union = pred_sum + mask_sum - intersection
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + mask_sum + self.smooth)
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return dice, iou
    
    @staticmethod
    def get_boundary(mask):
        import scipy.ndimage as ndi
        eroded = ndi.binary_erosion(mask)
        boundary = mask ^ eroded
        return boundary
    
    def calculate_nsd(self, pred, mask, tau=2.0):
        pred = (pred > self.threshold).astype(bool)
        mask = mask.astype(bool)
        
        boundary_pred = self.get_boundary(pred)
        boundary_mask = self.get_boundary(mask)
        
        if not boundary_pred.any() and not boundary_mask.any():
            return 1.0
        
        dist_pred_to_mask = distance_transform_edt(~boundary_mask)
        dist_mask_to_pred = distance_transform_edt(~boundary_pred)
        
        dist_values_pred = dist_pred_to_mask[boundary_pred]
        dist_values_mask = dist_mask_to_pred[boundary_mask]
        
        matches_pred = (dist_values_pred <= tau).sum()
        matches_mask = (dist_values_mask <= tau).sum()
        
        total_boundary = boundary_pred.sum() + boundary_mask.sum()
        
        nsd = (matches_pred + matches_mask) / (total_boundary + 1e-6)
        
        return nsd
    
    def calculate_all_metrics(self, pred, mask, nsd_tau=2.0):
        if isinstance(pred, torch.Tensor):
            pred_np = pred.squeeze().cpu().detach().numpy()
        else:
            pred_np = pred.squeeze()
            
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().detach().numpy()
        else:
            mask_np = mask.squeeze()
        
        if isinstance(pred, torch.Tensor) and isinstance(mask, torch.Tensor):
            dice, iou = self.calculate_dice_iou(pred, mask)
            dice_val = dice.item()
            iou_val = iou.item()
        else:
            pred_tensor = torch.from_numpy(pred_np).float()
            mask_tensor = torch.from_numpy(mask_np).float()
            dice, iou = self.calculate_dice_iou(pred_tensor, mask_tensor)
            dice_val = dice.item()
            iou_val = iou.item()
        
        nsd_val = self.calculate_nsd(pred_np, mask_np, nsd_tau)
        
        return {
            'dice': dice_val * 100,
            'iou': iou_val * 100,
            'nsd': nsd_val * 100
        }


def load_model_and_checkpoint(checkpoint_path, model_type, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model = Model(args=None)
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        return model, state_dict
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None, None


def test_epoch(model, net, SAM, test_loader, metrics_calc, device, epoch, args):
    model.eval()
    net.eval()
    
    metrics_list = defaultdict(list)
    
    pbar = tqdm(test_loader, desc=f"Testing epoch {epoch}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            image = batch["image"].to(device)
            gt2D = batch["gt2D"].to(device)
            attribution_map = batch["attribution_map"].to(args.device)
            
            input_images = SAM.preprocess(image)
            image_embeddings = net.sam.image_encoder(input_images)
            B, C, H, W = image_embeddings.shape
            attribution_map = nn.functional.interpolate(attribution_map.float(), size=(H, W), mode='bilinear', align_corners=False)
            
            logits_pred = model(image_embeddings, attribution_map)
            logits_pred = F.interpolate(
                logits_pred, 
                size=(args.image_size, args.image_size), 
                mode="bilinear", 
                align_corners=False
            )
            
            for i in range(image.size(0)):
                metrics = metrics_calc.calculate_all_metrics(
                    logits_pred[i:i+1], 
                    gt2D[i:i+1], 
                    args.nsd_tau
                )
                
                for key, value in metrics.items():
                    metrics_list[key].append(value)
            
            if args.save_predictions:
                save_predictions(logits_pred, batch, epoch, args)
            
            current_metrics = {k: np.mean(v) for k, v in metrics_list.items()}
            pbar.set_postfix(current_metrics)
    
    stats = {}
    for metric_name, values in metrics_list.items():
        values_array = np.array(values)
        stats[metric_name] = {
            'mean': np.mean(values_array),
            'std': np.std(values_array),
            'median': np.median(values_array),
            'min': np.min(values_array),
            'max': np.max(values_array)
        }
    
    return stats


def save_predictions(logits, batch, epoch, args):
    save_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    
    preds = torch.sigmoid(logits)
    preds_binary = (preds > args.threshold).float()
    
    for i in range(preds_binary.size(0)):
        image_name = batch.get("image_name", [f"pred_{i}"])[i]
        pred_mask = preds_binary[i, 0].cpu().numpy()
        np.save(os.path.join(save_dir, f"{image_name}.png"), pred_mask)


def format_metrics_table(epoch_stats):
    header = f"{'Epoch':<10} {'Metric':<10} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}"
    separator = "=" * 70
    
    lines = [separator, header, separator]
    
    for epoch, stats in epoch_stats.items():
        for metric_name in ['dice', 'iou', 'nsd']:
            if metric_name in stats:
                s = stats[metric_name]
                lines.append(
                    f"{epoch:<10} {metric_name.upper():<10} "
                    f"{s['mean']:<10.2f} {s['std']:<10.2f} "
                    f"{s['median']:<10.2f} {s['min']:<10.2f} {s['max']:<10.2f}"
                )
        lines.append(separator)
    
    return "\n".join(lines)


def main(args):
    logger = setup_logging(args.work_dir)
    logger.info("=" * 50)
    logger.info("Starting testing with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 50)
    
    test_dataset = MyDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        data_aug=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.info(f"Test samples: {len(test_dataset)}")
    
    SAM = build_sam_vit_h(args.sam_checkpoint).to(args.device)
    pkg = import_module('sam_lora_image_encoder')
    net = pkg.LoRA_Sam(SAM, args.lora_rank).to(args.device)
    logger.info("SAM model loaded successfully")
    
    metrics_calc = MetricsCalculator(threshold=args.threshold)
    
    all_epoch_stats = {}
    
    epochs_to_test = range(args.start_epoch, args.end_epoch + 1, args.step)
    
    for epoch in epochs_to_test:
        checkpoint_path = os.path.join(args.work_dir, f"{epoch}.pth")
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        logger.info(f"\nTesting epoch {epoch}...")
        
        model, state_dict = load_model_and_checkpoint(
            checkpoint_path, args.model_type, args.device
        )
        
        if model is None:
            continue

        if "lora" in state_dict:
            net.load_state_dict(state_dict["lora"])
        else:
            logger.warning(f"No LoRA weights found in checkpoint {epoch}")

        stats = test_epoch(model, net, SAM, test_loader, metrics_calc, args.device, epoch, args)
        all_epoch_stats[epoch] = stats
        
        logger.info(f"Epoch {epoch} Results:")
        for metric_name in ['dice', 'iou', 'nsd']:
            if metric_name in stats:
                s = stats[metric_name]
                logger.info(
                    f"  {metric_name.upper()}: "
                    f"Mean={s['mean']:.2f}%, Std={s['std']:.2f}%, "
                    f"Median={s['median']:.2f}%"
                )

    logger.info("\n" + "=" * 70)
    logger.info("TESTING RESULTS SUMMARY")
    logger.info(format_metrics_table(all_epoch_stats))
    
    # 找出最佳epoch
    if all_epoch_stats:
        best_dice_epoch = max(all_epoch_stats.keys(), 
                             key=lambda e: all_epoch_stats[e].get('dice', {}).get('mean', 0))
        best_iou_epoch = max(all_epoch_stats.keys(), 
                            key=lambda e: all_epoch_stats[e].get('iou', {}).get('mean', 0))
        best_nsd_epoch = max(all_epoch_stats.keys(), 
                            key=lambda e: all_epoch_stats[e].get('nsd', {}).get('mean', 0))
        
        logger.info(f"\nBest Epochs:")
        logger.info(f"  Best Dice: Epoch {best_dice_epoch} "
                   f"({all_epoch_stats[best_dice_epoch]['dice']['mean']:.2f}%)")
        logger.info(f"  Best IoU: Epoch {best_iou_epoch} "
                   f"({all_epoch_stats[best_iou_epoch]['iou']['mean']:.2f}%)")
        logger.info(f"  Best NSD: Epoch {best_nsd_epoch} "
                   f"({all_epoch_stats[best_nsd_epoch]['nsd']['mean']:.2f}%)")
    
    logger.info("Testing completed!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
