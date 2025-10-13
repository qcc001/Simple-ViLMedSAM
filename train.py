import torch.nn as nn
import math
from model.aggregation_model import aggregation_model_registry
from model.aggregation_model import build_aggregation_model
import torch
import argparse
from os.path import join
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from dataloader import TrainDataset
from torch.nn import functional as F
from segment_anything.build_sam import build_sam_vit_h
from segment_anything.modeling.common import generate_vmap
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
import timm
import os
from importlib import import_module
import random

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
    smooth = 1.0
    assert pred.shape == mask.shape, "pred and mask should have the same shape."
    pred = torch.sigmoid(pred)
    intersection = torch.sum(pred * mask)
    union = torch.sum(pred) + torch.sum(mask)
    dice_loss = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice_loss

def calculate_mean_iou(gt, pred):
    pred[pred > 0] = int(1)
    pred[pred <= 0] = int(0)
    intersection = torch.sum((gt == 1) & (pred == 1))
    union = torch.sum((gt == 1) | (pred == 1))
    iou = intersection / union
    return iou

def main(args):
    model = aggregation_model_registry[args.model_type](args).to(args.device)
    print(f"model size: {sum(p.numel() for p in model.parameters())}")

    SAM = build_sam_vit_h("sam_vit_h_4b8939.pth").to(args.device)
    pkg = import_module('sam_lora_image_encoder')
    net = pkg.LoRA_Sam(SAM, 4).cuda()
    model_grad_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params))

    CLIP = AutoModel.from_pretrained("clip-vit-large-patch14", trust_remote_code=True).to(args.device)
    clip_processor = AutoProcessor.from_pretrained("clip-vit-large-patch14", trust_remote_code=True)
    clip_tokenizer = AutoTokenizer.from_pretrained("clip-vit-large-patch14", trust_remote_code=True)
    CLIP.requires_grad_(False)

    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    train_dataset = TestDataset(data_root=args.data_root, image_size=args.image_size, data_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
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

    train_losses = []
    iou_train = []
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        model.train()
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        iou_train_list = []
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"].to('cuda')
            gt2D = batch["gt2D"].to('cuda')
            input_images = SAM.preprocess(image)
            image_embeddings = net.sam.image_encoder(input_images)  # [b,256,64,64] vit_b
            vmap, clip_v_features, clip_attn_weights = generate_vmap(batch, args.data_root, CLIP, clip_processor,clip_tokenizer,args.device,'test')  # [b,1,64,64] [b,1,256,768] ViT-B/16 [b,4,hw,c]
            logits_pred = model(image_embeddings, clip_v_features, clip_attn_weights, vmap)
            logits_pred = F.interpolate(logits_pred, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
            l_seg = dice_coeff(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            loss = args.seg_loss_weight * l_seg + args.ce_loss_weight * l_ce
            epoch_loss[step] = loss.item()
            loss = loss / args.accumulation_steps
            loss.backward()
            if (step + 1) % args.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_description(
                f"Epoch {epoch} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
            iou_score = calculate_mean_iou(gt2D, logits_pred)
            iou_train_list.append(iou_score.cpu().detach().numpy())

        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)

        print(f"train loss:{epoch_loss_reduced:.6f}")
        print(" train iou:", np.mean(iou_train_list))
        train_losses.append(epoch_loss_reduced)
        scheduler.step()
        iou_train.append(np.mean(iou_train_list))
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


if __name__ == '__main__':
    #random_seed = 42
    #np.random.seed(random_seed)
    #random.seed(random_seed)
    args = parse_args()
    main(args)


