import os
import datetime
import argparse
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
from model.aggregation_model import aggregation_model_registry
from segment_anything.build_sam import build_sam_vit_h
from segment_anything.modeling.common import generate_map

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--accumulation_steps", type=int , default=8, help="train accumulation_steps")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--data_root", type=str, default="/train data path", help="train data path")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lora_lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--model_lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="default", help="model_type")
    parser.add_argument("-weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("-iou_loss_weight", type=float, default=1.0, help="Weight of IoU loss.")
    parser.add_argument("-seg_loss_weight", type=float, default=1.0, help="Weight of segmentation loss.")
    parser.add_argument("-ce_loss_weight", type=float, default=1.0, help="Weight of cross entropy loss.")
    parser.add_argument("-num_workers", type=int, default=4, help="Number of workers for dataloader.")
    args = parser.parse_args()

    if args.resume is not None:
        args.checkpoint = None
    return args

def dice_coeff(pred, mask):
    smooth = 1e-6
    assert pred.shape == mask.shape, "pred and mask should have the same shape."
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = torch.sum(pred * mask)
    union = torch.sum(pred) + torch.sum(mask)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def evaluation(model_to_eval, dataloader, SAM, net, CLIP, clip_processor, clip_tokenizer, args):
    model_to_eval.eval()
    dices = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            image = batch["image"].to(args.device)
            gt2D = batch["gt2D"].to(args.device)

            # get image embeddings from SAM if needed (keep same preprocessing as training)
            input_image = SAM.preprocess(image)
            image_embeddings = net.sam.image_encoder(input_image)
            attribution_map = generate_map(batch, args.data_root, CLIP, clip_processor, clip_tokenizer, args.device)
            logits = model_to_eval(image_embeddings, attribution_map)
            logits = F.interpolate(logits, (args.image_size, args.image_size), mode='bilinear', align_corners=False)
            pred = torch.sigmoid(logits)

            dice_i = dice_coeff(pred, gt2D)

            try:
                dices.extend(dice_i.cpu().numpy().tolist())
            except (AttributeError, TypeError):
                dices.append(dice_i.item() if hasattr(dice_i, 'item') else dice_i)
    mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
    return mean_dice

def main(args):
    model = aggregation_model_registry[args.model_type](args).to(args.device)
    print(f"model size: {sum(p.numel() for p in model.parameters())}")

    SAM = build_sam_vit_h("sam_vit_h_4b8939.pth").to(args.device)
    pkg = import_module('sam_lora_image_encoder')
    net = pkg.LoRA_Sam(SAM, 4).cuda()
    model_grad_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params))

    CLIP, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    CLIP.requires_grad_(False)

    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    train_dataset = MyDataset(data_root=args.data_root, image_size=args.image_size, data_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    print('*******Train data:', len(train_dataset))

    val_dataset = MyDataset(data_root=args.data_root, image_size=args.image_size, data_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    print('*******Train data:', len(train_dataset))

    effective_batch_size = args.batch_size * args.accumulation_steps
    num_training_steps = args.num_epochs * (len(train_dataset) // effective_batch_size)
    num_warmup_steps = int(0.05 * num_training_steps)

    optimizer = optim.Adam([{"params": net.parameters(), "lr": args.lora_lr}, {"params": model.parameters(), "lr": args.model_lr}], weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    print('*******Use warmup_cosineLR')

    torch.cuda.empty_cache()
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            net.load_state_dict(checkpoint["lora"])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            print(f"*******load {args.resume}")
    else:
        start_epoch = 0
    torch.cuda.empty_cache()

    best_seg_dice = -1.0
    best_epoch = 0
    best_checkpoint = None

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        model.train()
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"].to(args.device)
            gt2D = batch["gt2D"].to(args.device)
            input_images = SAM.preprocess(image)
            image_embeddings = net.sam.image_encoder(input_images)  # [b,256,64,64] vit_b
            attribution_map = generate_map(batch, args.data_root, CLIP, clip_processor, clip_tokenizer, args.device)  # [b,1,64,64] [b,1,256,768] ViT-B/16 [b,4,hw,c]
            logits_pred = model(image_embeddings, attribution_map)
            logits_pred = F.interpolate(logits_pred, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
            l_seg = dice_loss(logits_pred, gt2D.float())
            l_ce = ce_loss(logits_pred, gt2D.float())
            loss = args.seg_loss_weight * l_seg + args.ce_loss_weight * l_ce
            epoch_loss[step] = loss.item()
            loss = loss / args.accumulation_steps
            loss.backward()
            if (step + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description(
                f"Epoch {epoch} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch} summary | loss: {epoch_loss_reduced:.4f}")

        print("Running validation...")
        val_dice = evaluation(model, val_loader, SAM, net, CLIP, clip_processor, clip_tokenizer, args)
        print(f"Validation seg dice: {val_dice:.4f}")

        scheduler.step()
        model_weights = model.state_dict()
        checkpoint = {
            "lora": net.state_dict(),
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "loss": epoch_loss_reduced,
        }
        os.makedirs(args.work_dir, exist_ok=True)
        if (epoch % 10) == 0:
            torch.save(checkpoint, join(args.work_dir, f"{epoch}.pth"))

        if val_dice > best_seg_dice:
            best_seg_dice = val_dice
            best_epoch = epoch
            best_checkpoint = checkpoint

        torch.save(best_checkpoint, join(args.work_dir, "best_model.pth"))
        print(f"\nTraining Complete! Best Dice: {best_seg_dice:.4f} at epoch {best_epoch}")
        print("Training finished")


if __name__ == '__main__':
    args = parse_args()
    main(args)
