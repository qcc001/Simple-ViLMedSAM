import warnings

warnings.filterwarnings('ignore')
import os
import cv2
import json
import pandas as pd
import itertools
import torch
import random
import numpy as np
from PIL import Image
from typing import Optional
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from methods import vision_heatmap_iba
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Disable parallel tokenization warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Function to calculate Dice coefficient for evaluating segmentation
def calculate_dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    denom = (mask1.sum() + mask2.sum())
    if denom == 0:
        return 1.0 if mask1.sum() == 0 and mask2.sum() == 0 else 0.0
    return (2.0 * intersection) / denom

# Function to evaluate a model on a sample image and calculate the Dice score
def evaluate_on_sample(model, processor, tokenizer, SAM, simple_text, image_paths, args):
    dice_scores = []  # Store Dice scores for each image
    for image_id in tqdm(image_paths):  # Iterate through images
        try:
            # Open and preprocess the image
            original_image = Image.open(f"{args.val_path}/{image_id}").convert('RGB')
            image = pad_to_square(original_image, fill_color=(0, 0, 0))
        except:
            continue

        # Convert the image to tensor
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        # image_feat = processor(image).to(args.device)

        # Tokenize the input text
        text_ids = torch.tensor([tokenizer.encode(args.text, add_special_tokens=True)]).to(args.device)
        # text_ids = torch.tensor(tokenizer(text, context_length=256)).to(args.device)

        # Generate a visual attention map using a custom method
        map = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar,
                                         device=args.device, ensemble=args.ensemble, progbar=False)

        # Load the ground truth mask for comparison
        gt_path = args.val_path.replace("images", "labels")
        gt = Image.open(f"{gt_path}/{image_id}").convert("L")
        gt_pad = pad_to_square_gt(gt, fill_color=(0, 0, 0))
        gt_mask = np.array(gt_pad)

        # Resize the generated map to match the ground truth mask size
        vmap_resized = cv2.resize(np.array(vmap), (gt_mask.shape[1], gt_mask.shape[0]))
        # visualize_masks_overlay(gt_mask, vmap_resized)
        # Threshold the map to create a binary mask
        cam_img = vmap_resized > 0.3
        # visualize_masks_overlay(gt_mask, cam_img)

        # Calculate the Dice score
        dice_score = calculate_dice_coefficient(gt_mask.astype(bool), cam_img.astype(bool))
        dice_scores.append(dice_score)

    # Return the average Dice score
    average_dice = np.mean(dice_scores)
    return average_dice


# Function to perform hyperparameter optimization
def hyper_opt(model, processor, tokenizer, SAM, simple_text, args):
    print("Running Hyperparameter Optimization ...")

    # Define lists of possible hyperparameter values
    vbeta_list = [0.01, 0.1, 1.0, 2.0]
    vvar_list = [0.01, 0.1, 1.0, 2.0]
    layers_list = [7, 8, 9, 10]

    # Create all combinations of the hyperparameters
    hyperparameter_combinations = list(itertools.product(vbeta_list, vvar_list, layers_list))

    # Get all image IDs from the validation path
    all_image_ids = sorted(os.listdir(args.val_path))

    results = []  # Store results of each combination

    # Iterate through each hyperparameter combination
    for combo in hyperparameter_combinations:
        vbeta, vvar, layer = combo
        args.vbeta = vbeta
        args.vvar = vvar
        args.vlayer = layer

        sample_dice_scores = []  # Store Dice scores for each sample

        print(f"Evaluating combination: vbeta={vbeta}, vvar={vvar}, layer={layer}")

        # Run 3 random samples to get an average performance
        for i in range(3):
            random.seed(i)
            sampled_images = random.sample(all_image_ids, 1)
            avg_dice = evaluate_on_sample(model, processor, tokenizer, SAM, simple_text, sampled_images, args)
            sample_dice_scores.append(avg_dice)
            print(f"  Sample {i + 1}: Average Dice Score = {avg_dice}")

        # Calculate mean Dice score for this hyperparameter combination
        mean_dice = np.mean(sample_dice_scores)
        results.append({
            'vbeta': vbeta,
            'vvar': vvar,
            'vlayer': layer,
            'average_dice': mean_dice
        })
        print(f"Mean Dice Score for this combination: {mean_dice}\n")

    # Convert results to a DataFrame for easy analysis
    results_df = pd.DataFrame(results)

    # Find the combination with the best Dice score
    best_combo = results_df.loc[results_df['average_dice'].idxmax()]

    print("Best Hyperparameter Combination:")
    print(best_combo)
    print("\n")

    return best_combo


def create_cam_overlay(image, heatmap, alpha=0.5):
    if hasattr(heatmap, 'cpu'):
        heatmap_np = heatmap.cpu().numpy()
    else:
        heatmap_np = np.array(heatmap)

    heatmap_normalized = cv2.normalize(src=heatmap_np, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # type: ignore
    heatmap_uint8 = np.uint8(heatmap_normalized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))

    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed

def unpad_square_from_resized_array(resized_padded_arr: np.ndarray, orig_w: int, orig_h: int, padded_original_size: Optional[int] = None) -> np.ndarray:
    h_new, w_new = resized_padded_arr.shape[:2]
    if padded_original_size is None:
        padded_original_size = max(orig_w, orig_h)
    scale_x = w_new / padded_original_size
    scale_y = h_new / padded_original_size
    paste_x = int(round((padded_original_size - orig_w) / 2 * scale_x))
    paste_y = int(round((padded_original_size - orig_h) / 2 * scale_y))
    # 右下角坐标（exclusive）
    x2 = paste_x + int(round(orig_w * scale_x))
    y2 = paste_y + int(round(orig_h * scale_y))

    # ensure bounds
    h_new, w_new = resized_padded_arr.shape[:2]
    paste_x = max(0, min(paste_x, w_new - 1))
    paste_y = max(0, min(paste_y, h_new - 1))
    x2 = max(paste_x + 1, min(x2, w_new))
    y2 = max(paste_y + 1, min(y2, h_new))

    cropped = resized_padded_arr[paste_y:y2, paste_x:x2].copy()
    if (cropped.shape[1], cropped.shape[0]) != (orig_w, orig_h):
        cropped = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def unpad_square(padded_img: Image.Image, orig_w: int, orig_h: int):
    max_side = max(orig_w, orig_h)
    paste_x = (max_side - orig_w) // 2
    paste_y = (max_side - orig_h) // 2

    left = paste_x
    upper = paste_y
    right = paste_x + orig_w
    lower = paste_y + orig_h
    cropped_img = padded_img.crop((left, upper, right, lower))
    return cropped_img


def pad_to_square(img: Image.Image, fill_color=(0, 0, 0)):
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new("RGB", (max_side, max_side), fill_color)
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def pad_to_square_gt(gt, fill_color=(0, 0, 0)):
    width, height = gt.size
    mode = gt.mode

    max_side = max(width, height)

    if mode == "L":
        if isinstance(fill_color, (tuple, list)) and len(fill_color) >= 1:
            fill_color_single = fill_color[0]
        else:
            fill_color_single = fill_color
    elif mode == "RGB":
        if isinstance(fill_color, (int, float)):
            fill_color_single = (fill_color, fill_color, fill_color)
        elif isinstance(fill_color, (tuple, list)) and len(fill_color) == 3:
            fill_color_single = fill_color
        else:
            fill_color_single = (0, 0, 0)
    elif mode == "RGBA":
        if isinstance(fill_color, (int, float)):
            fill_color_single = (fill_color, fill_color, fill_color, 255)
        elif isinstance(fill_color, (tuple, list)) and len(fill_color) == 3:
            fill_color_single = (*fill_color, 255)
        elif isinstance(fill_color, (tuple, list)) and len(fill_color) == 4:
            fill_color_single = fill_color
        else:
            fill_color_single = (0, 0, 0, 255)
    else:
        fill_color_single = 0

    new_img = Image.new(mode, (max_side, max_side), fill_color_single)

    paste_x = (max_side - width) // 2
    paste_y = (max_side - height) // 2

    new_img.paste(gt, (paste_x, paste_y))

    return new_img


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("Loading models ...")

    # Load the appropriate model based on the arguments
    if (args.model_name == "BiomedCLIP"):
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    else:
        model = CLIPModel.from_pretrained(args.model).to(args.device)
        processor = CLIPProcessor.from_pretrained(args.model)
        tokenizer = CLIPTokenizerFast.from_pretrained(args.model)

    # Perform hyperparameter optimization if required
    if args.hyper_opt:
        best_combo = hyper_opt(model, processor, tokenizer, SAM, simple_text, args)
        args.vbeta = best_combo['vbeta']
        args.vvar = best_combo['vvar']
        args.vlayer = int(best_combo['vlayer'])

    print("Generating Saliency Maps ...")

    # Iterate through the input images and generate saliency maps
    for image_id in tqdm(sorted(os.listdir(args.input_path))):
        try:
            original_image = Image.open(f"{args.input_path}/{image_id}").convert('RGB')
            orig_w, orig_h = original_image.size
            image = pad_to_square(original_image, fill_color=(0, 0, 0))
        except:
            print(f"Unable to load image at {image_id}", flush=True)
            continue

        # Preprocess the image and tokenize the text
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = torch.tensor([tokenizer.encode(text_, add_special_tokens=True)]).to(args.device)

        # Generate visual saliency map
        vmap = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar, device=args.device, ensemble=args.ensemble, progbar=False)

        # Resize and save the saliency map
        img = np.array(unpad_square(image, orig_w, orig_h))
        vmap_resized = cv2.resize(np.array(vmap), (image.size[1], image.size[0]), interpolation=cv2.INTER_NEAREST)
        vmap_unpad = unpad_square_from_resized_array(vmap_resized, orig_w=orig_w, orig_h=orig_h, padded_original_size=image.size[1])

        overlay_image = create_cam_overlay(img, vmap_unpad, alpha=0.5)
        os.makedirs(f"{args.output_path1}", exist_ok=True)
        cv2.imwrite(f"{args.output_path1}/{image_id}", overlay_image)
        os.makedirs(f"{args.output_path2}", exist_ok=True)
        cv2.imwrite(f"{args.output_path2}/{image_id}", vmap_unpad * 255)


# Entry point for the script
if __name__ == '__main__':
    # Define argument parser for input/output paths and hyperparameters
    parser = argparse.ArgumentParser('M2IB argument parser')
    parser.add_argument('--input-path', required=False, default="your image path", type=str, help='path to the images')
    parser.add_argument('--output-path1', required=False, default="your output path", type=str,
                        help='path to the output')
    parser.add_argument('--output-path2', required=False, default="your output path",
                        type=str, help='path to the output')
    parser.add_argument('--val-path', type=str, default=""your val image path",
                        help='path to the validation set for hyperparameter optimization')
    parser.add_argument('--vbeta', type=float, default=2.0)
    parser.add_argument('--vvar', type=float, default=2.0)
    parser.add_argument('--vlayer', type=int, default=7)
    parser.add_argument('--model_name', type=str, default="BiomedCLIP", help="Which CLIP model to use")
    parser.add_argument('--model_path', type=str, default="your model path")
    parser.add_argument('--text', type=str, default="simple text to describe target image category")
    parser.add_argument('--finetuned', action='store_true', help="Whether to use finetuned weights or not")
    parser.add_argument('--hyper_opt', default=True, help="Whether to optimize hyperparameters or not")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on")
    parser.add_argument('--ensemble', action='store_true', help="Whether to use text ensemble or not")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)

    print("Saliency Map Generation Done!")
