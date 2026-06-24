import os
import datetime
import argparse
import logging
from os.path import join
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

import monai

from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import get_cosine_schedule_with_warmup

from dataloader import MyDataset
from model.model import Model
from segment_anything.build_sam import build_sam_vit_h

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_logging(work_dir):
    log_file = os.path.join(work_dir, f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--accumulation_steps", type=int , default=8, help="train accumulation_steps")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--data_root", type=str, default="/train data path", help="train data path")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument("--lora_lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--model_lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="default", help="model_type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seg_loss_weight", type=float, default=1.0, help="Weight of segmentation loss.")
    parser.add_argument("--ce_loss_weight", type=float, default=1.0, help="Weight of cross entropy loss.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument('--val_interval', type=int, default=1, help="validate every N epochs")
    args = parser.parse_args()

    if args.resume is not None:
        args.checkpoint = None
    return args

def dice_coeff(pred, mask):
    smooth = 1e-6
    assert pred.shape == mask.shape, "pred and mask should have the same shape."
    pred = (pred > 0.5).float()
    intersection = torch.sum(pred * mask)
    union = torch.sum(pred) + torch.sum(mask)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def evaluation(model_to_eval, dataloader, SAM, lora_net, args, logger):
    model_to_eval.eval()
    dices = []
    logger.info("Starting validation...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
            image = batch["image"].to(args.device)
            gt2D = batch["gt2D"].to(args.device)
            attribution_map = batch["attribution_map"].to(args.device)

            input_image = SAM.preprocess(image)
            image_embeddings = lora_net.sam.image_encoder(input_image)
            B, C, H, W = image_embeddings.shape
            attribution_map = nn.functional.interpolate(attribution_map.float(), size=(H, W), mode='bilinear', align_corners=False)
            logits = model_to_eval(image_embeddings, attribution_map)
            logits = F.interpolate(logits, (args.image_size, args.image_size), mode='bilinear', align_corners=False)
            pred = torch.sigmoid(logits)

            dice_i = dice_coeff(pred, gt2D)

            try:
                dices.extend(dice_i.cpu().numpy().tolist())
            except (AttributeError, TypeError):
                dices.append(dice_i.item() if hasattr(dice_i, 'item') else dice_i)
    
    mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
    logger.info(f"Validation completed. Mean Dice: {mean_dice:.4f}")
    return mean_dice

def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    logger = setup_logging(args.work_dir)
    
    logger.info("=" * 50)
    logger.info("Starting training with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 50)
    
    model = Model(args).to(args.device)
    logger.info(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")

    SAM = build_sam_vit_h("sam_vit_h_4b8939.pth").to(args.device)
    pkg = import_module('sam_lora_image_encoder')
    lora_net = pkg.LoRA_Sam(SAM, 4).cuda()
    model_grad_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f'Trainable LoRA parameters: {model_grad_params:,}')

    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    train_dataset = MyDataset(data_root=args.data_root, image_size=args.image_size, data_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    logger.info(f'Training samples: {len(train_dataset)}')

    val_dataset = MyDataset(data_root=args.data_root, image_size=args.image_size, data_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    logger.info(f'Validation samples: {len(val_dataset)}')

    effective_batch_size = args.batch_size * args.accumulation_steps
    num_training_steps = args.num_epochs * (len(train_dataset) // effective_batch_size)
    num_warmup_steps = int(0.05 * num_training_steps)
    
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")

    optimizer = optim.AdamW([
        {"params": net.parameters(), "lr": args.lora_lr}, 
        {"params": model.parameters(), "lr": args.model_lr}
    ], weight_decay=1e-4)

    if args.lr_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        logger.info('Using CosineAnnealingLR with Warmup')
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.model_lr * 0.01
        )
        logger.info('*******Use CosineAnnealingLR')

    torch.cuda.empty_cache()
    start_epoch = 0
    global_step = 0
    
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            last = "${args.resume##*/}"
            prefix = "${args.resume%/*}"
            lora_ckpt = "${prefix}/lora/${last}"
            lora_net.load_lora_parameters(lora_ckpt)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint.get('global_step', start_epoch * len(train_loader))
            logger.info(f"Resumed from {args.resume}, starting from epoch {start_epoch + 1}, global step {global_step}")
    torch.cuda.empty_cache()

    best_seg_dice = -1.0
    best_epoch = 0
    best_checkpoint = None

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        logger.info(f"\n{'='*50}\nEpoch {epoch}/{args.num_epochs}\n{'='*50}")
        model.train()
        net.train()
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            image = batch["image"].to(args.device)
            gt2D = batch["gt2D"].to(args.device)
            attribution_map = batch["attribution_map"].to(args.device)
            
            input_images = SAM.preprocess(image)
            image_embeddings = lora_net.sam.image_encoder(input_images)  # [b,256,64,64]
            B, C, H, W = image_embeddings.shape
            attribution_map = nn.functional.interpolate(attribution_map.float(), size=(H, W), mode='bilinear', align_corners=False)
            
            logits_pred = model(image_embeddings, attribution_map)
            logits_pred = F.interpolate(
                logits_pred, 
                (args.image_size, args.image_size), 
                mode="bilinear", 
                align_corners=False
            )
            
            l_seg = dice_loss(logits_pred, gt2D.float())
            l_ce = ce_loss(logits_pred, gt2D.float())
            loss = args.seg_loss_weight * l_seg + args.ce_loss_weight * l_ce
            
            epoch_losses.append(loss.item())
            
            loss = loss / args.accumulation_steps
            loss.backward()
            
            if (step + 1) % args.accumulation_steps == 0:
                optimizer.step()
                if args.lr_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                current_lr = scheduler.get_last_lr()
                logger.debug(f"Step {global_step}: LR = {[f'{lr:.2e}' for lr in current_lr]}")
                
            pbar.set_description(
                f"Epoch {epoch} | Loss: {loss.item()*args.accumulation_steps:.4f} | "
                f"Seg: {l_seg.item():.4f} | CE: {l_ce.item():.4f}"
            )
        
        if (step + 1) % args.accumulation_steps != 0:
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        if not args.lr_scheduler:
                scheduler.step()

        avg_epoch_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Average Loss: {avg_epoch_loss:.4f}")
        logger.info(f"  Current LR: {[f'{lr:.2e}' for lr in scheduler.get_last_lr()]}")
        
        logger.info("Running validation...")
        val_dice = evaluation(
            model, val_loader, SAM, lora_net, args, logger
        )
        logger.info(f"Validation Seg Dice: {val_dice:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "loss": avg_epoch_loss,
            "best_dice": best_seg_dice,
        }
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir, exist_ok=True)
        save_lora_dir = join(args.work_dir, "lora")
        if not os.path.exists(save_lora_dir):
            os.makedirs(save_lora_dir)
        
        if (epoch % args.val_interval) == 0:
            save_path = join(args.work_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, save_path)
            save_lora_path = os.path.join(save_lora_dir, 'checkpoint_epoch_{epoch}.pth')
            lora_net.save_lora_parameters(save_lora_path)
            logger.info(f"Saved checkpoint to {save_path}")
        
        if val_dice > best_seg_dice:
            best_seg_dice = val_dice
            best_epoch = epoch
            best_checkpoint = checkpoint.copy()
            best_path = join(args.work_dir, "best_model.pth")
            torch.save(best_checkpoint, best_path)
            save_lora_path = os.path.join(save_lora_dir, 'best_model.pth')
            lora_net.save_lora_parameters(save_lora_path)
            logger.info(f"New best model! Dice: {best_seg_dice:.4f} at epoch {best_epoch}")
        
        latest_path = join(args.work_dir, "latest_model.pth")
        torch.save(checkpoint, latest_path)
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Training Complete! Best Dice: {best_seg_dice:.4f} at epoch {best_epoch}")
    logger.info("=" * 50)
    print(f"\nTraining Complete! Best Dice: {best_seg_dice:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
