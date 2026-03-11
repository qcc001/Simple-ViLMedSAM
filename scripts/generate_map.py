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
from methods import vision_heatmap_iba, vision_heatmap_iba_hard, vision_heatmap_iba_proxy
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from Segment_Anything.build_sam import build_sam_vit_h

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

        if args.use_MedGemma:
            with open(args.val_json_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
            text_dict = {}
            lines = content.strip().split('\n')
            for line in lines:
                if line:
                    try:
                        data = eval(line)
                        text_dict.update(data)
                    except Exception as e:
                        print(f"Error parsing line: {line} - {e}")
            if image_id in text_dict:
                text_ = text_dict[image_id]
            else:
                print(f"Warning: {image_id} not found in TXT file")
                text_ = simple_text
        else:
            text_ = simple_text
        # Convert the image to tensor
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        # image_feat = processor(image).to(args.device)

        # Tokenize the input text
        text_ids = torch.tensor([tokenizer.encode(text_, add_special_tokens=True)]).to(args.device)
        # text_ids = torch.tensor(tokenizer(text, context_length=256)).to(args.device)

        if args.use_SAM:
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                image_np = cv2.resize(image_np,(1024,1024))
                image_tensor = torch.from_numpy(image_np).float()
                # 如果图像是 RGB，需要调整维度顺序 (H, W, C) -> (C, H, W)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)
            else:
                image_tensor = image
            input_image = SAM.preprocess(image_tensor.to(args.device))
            with torch.no_grad():
                ex_feat = SAM.image_encoder(input_image.unsqueeze(0).to(args.device))
        else:
            ex_feat = None

        # Generate a visual attention map using a custom method
        if args.use_hard:
            vmap1 = cv2.imread(f"{args.vmap_root}/vmap/{args.vmap_type1}/{image_id}", 0).astype(np.uint8)
            vmap1 = pad_vmap_to_square(vmap1, fill_value=0)
            vmap2 = cv2.imread(f"{args.vmap_root}/vmap/{args.vmap_type2}/{image_id}", 0).astype(np.uint8)
            vmap2 = pad_vmap_to_square(vmap2, fill_value=0)
            H = np.stack([vmap1, vmap2], axis=0)
            difficulty = H.var(axis=0)
            difficulty_map = difficulty / difficulty.max()
            difficulty_map = torch.from_numpy(difficulty_map).to(args.device)
            gamma = 0.1 / (difficulty.mean() + 1e-6)
            difficulty_14 = F.interpolate(difficulty_map.unsqueeze(0).unsqueeze(0), size=(14, 14), mode="bilinear")
            difficulty_tokens = difficulty_14.reshape(-1)  # [196]
            difficulty_tokens = torch.cat([torch.zeros(1).to(args.device), difficulty_tokens])  # [197]
            difficulty_tokens = difficulty_tokens.unsqueeze(1)  # [197,1]
            text_ids2 = torch.tensor([tokenizer.encode(simple_text, add_special_tokens=True)]).to(args.device)
            vmap, _ = vision_heatmap_iba_hard(text_ids, text_ids2, image_feat, model, args.vlayer, args.vbeta,
                                              args.vvar, gamma, difficulty_tokens, device=args.device,
                                              ensemble=args.ensemble, progbar=False)
        elif args.use_SAM:
            vmap, _ = vision_heatmap_iba_proxy(text_ids, image_feat, ex_feat, model, args.vlayer, args.vbeta, args.vvar,
                                               device=args.device, ensemble=args.ensemble, progbar=False)
        else:
            vmap, _ = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar,
                                         device=args.device, ensemble=args.ensemble, progbar=False)

        if args.proxy:
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                image_np = cv2.resize(image_np, (1024, 1024))
                image_tensor = torch.from_numpy(image_np).float()
                # 如果图像是 RGB，需要调整维度顺序 (H, W, C) -> (C, H, W)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)
            else:
                image_tensor = image
            input_image = SAM.preprocess(image_tensor.to(args.device))
            with torch.no_grad():
                ex_feat = SAM.image_encoder(input_image.unsqueeze(0).to(args.device))
            B, C, H, W = ex_feat.shape
            q_k = F.normalize(ex_feat.flatten(2, 3), dim=1)  # [b,256,4096]
            similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)  # [b,4096,4096]
            similarity = (similarity - torch.mean(similarity) * 1.2) * 3.0
            similarity[similarity < 0.0] = float('-inf')
            attn_weights = F.softmax(similarity, dim=-1)
            vmap_tensor = torch.from_numpy(vmap).unsqueeze(0).unsqueeze(0)
            vmap_tensor = F.interpolate(vmap_tensor.float(), size=(H, W), mode='bilinear', align_corners=False)
            b, c, h, w = vmap_tensor.shape
            v = vmap_tensor.flatten(2).to(args.device)  # [b,1,4096]
            v = v.permute(0, 2, 1)  # [b,4096,1]
            attn_output = torch.bmm(attn_weights, v)  # [b,4096,1]
            attn_output = attn_output.permute(0, 2, 1)  # [b,1,4096]
            attn_output_vmap = attn_output.view(b, c, h, w).squeeze(0).squeeze(0)
            vmap = attn_output_vmap.cpu().numpy()

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
    """创建热力图与原图的叠加效果"""
    # 确保heatmap是numpy数组且在0-1范围内
    if hasattr(heatmap, 'cpu'):
        heatmap_np = heatmap.cpu().numpy()
    else:
        heatmap_np = np.array(heatmap)

    # 归一化heatmap
    heatmap_normalized = cv2.normalize(src=heatmap_np, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # type: ignore
    heatmap_uint8 = np.uint8(heatmap_normalized)

    # 应用JET颜色映射
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 调整热力图大小与原图匹配（如果需要）
    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))

    # 叠加热力图和原图
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed


def medgemma_generate(image_original, image_seg, text, target, MedGemma_model, MedGemma_processor):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful medical assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "image": image_original},
                {"type": "text", "text": f"Only give an detailed description (about 50 words) about the true {target} (avoid the false positive sample above) to improve {target} segmentation accuracy."},
                {"type": "image", "image": image_seg}
            ]
        }
    ]
    inputs = MedGemma_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(MedGemma_model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = MedGemma_model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]

    decoded = MedGemma_processor.decode(generation, skip_special_tokens=True)
    decoded = decoded.replace('\n<end_of_turn>', '')

    return decoded


def human_interaction(image_original, image_seg, image_gt):
    """
        显示原始图像与分割结果的叠加，以及真实标注图像，等待人工输入评价文本

        Args:
            image_original: 原始图像 (BGR格式)
            image_seg: 分割结果图像 (单通道或BGR格式)
            image_gt: 真实标注图像 (单通道或BGR格式)

        Returns:
            str: 人工输入的评价文本
        """
    # 确保图像尺寸一致
    h, w = image_original.shape[:2]
    image_seg = cv2.resize(image_seg, (w, h))
    image_gt = cv2.resize(image_gt, (w, h))

    # 处理分割图像：如果是单通道，转换为彩色
    if len(image_seg.shape) == 2 or image_seg.shape[2] == 1:
        image_seg = cv2.cvtColor(image_seg, cv2.COLOR_GRAY2BGR)

    # 处理真实标注图像：如果是单通道，转换为彩色
    if len(image_gt.shape) == 2 or image_gt.shape[2] == 1:
        image_gt = cv2.cvtColor(image_gt, cv2.COLOR_GRAY2BGR)

    # 创建叠加图像（原始图像 + 分割结果）
    # 使用透明度混合
    alpha = 0.6  # 分割结果的透明度
    overlay = image_original.copy()
    cv2.addWeighted(image_seg, alpha, overlay, 1 - alpha, 0, overlay)

    # 创建水平拼接的显示图像
    combined_width = w * 2 + 20  # 两张图像 + 间隔
    combined_height = h
    combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

    # 将叠加图像和真实标注图像拼接到一起
    combined_image[0:h, 0:w] = overlay
    combined_image[0:h, w + 20: w * 2 + 20] = image_gt

    # 添加标签文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 0, 0)  # 黑色

    # 在图像上方添加标题
    title_bg_height = 40
    title_image = np.ones((combined_height + title_bg_height, combined_width, 3), dtype=np.uint8) * 255

    # 将原图像复制到标题图像的下半部分
    title_image[title_bg_height:title_bg_height + combined_height, :] = combined_image

    # 添加标题
    cv2.putText(title_image, "Left: Original + Segmentation | Right: Ground Truth",
                (10, 25), font, font_scale, color, thickness)

    # 添加分隔线
    cv2.line(title_image, (w, title_bg_height), (w, title_bg_height + combined_height), (0, 0, 0), 1)
    cv2.line(title_image, (w + 10, title_bg_height), (w + 10, title_bg_height + combined_height), (0, 0, 0), 1)

    # 显示图像
    cv2.imshow('Human Evaluation - Compare Segmentation Results', title_image)

    # 在控制台提示用户输入
    print("\n" + "=" * 60)
    print("请查看可视化窗口中的分割结果对比")
    print("左侧: 原始图像 + 分割结果叠加")
    print("右侧: 真实标注 (Ground Truth)")
    print("=" * 60)

    # 获取用户输入
    evaluation_text = input("请输入您的评价 (完成后按回车关闭窗口): ")

    # 关闭显示窗口
    cv2.destroyAllWindows()

    return evaluation_text


# ----------------------- Cached hyper-opt (Vmap) -----------------------

class VmapCache:
    """Simple cache for hyper-opt best combos per 'category' (or other key)."""
    def __init__(self, cache_file):
        self.cache_file = cache_file
        ensure_dir(os.path.dirname(self.cache_file))
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def get(self, key):
        return self.cache.get(key, None)

    def set(self, key, value):
        self.cache[key] = value
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)


def hyper_opt_once(model, processor, tokenizer, text, val_path, device, sample_n=3):
    """
    Run a small hyperopt (cheap sampling) and return best combo dict:
    {'vbeta':..,'vvar':..,'vlayer':..,'average_dice':..}
    This function is called only when cache miss occurs.
    """
    print("Running Hyperparameter Optimization (sampled) ...")
    vbeta_list = [0.1, 1.0, 2.0]
    vvar_list = [0.1, 1.0, 2.0]
    layers_list = [7, 8, 9]
    hyperparameter_combinations = list(itertools.product(vbeta_list, vvar_list, layers_list))

    all_image_ids = sorted(os.listdir(val_path))
    if len(all_image_ids) == 0:
        raise ValueError(f"Validation path {val_path} empty or wrong.")

    results = []
    rng = random.Random(0)

    for combo in hyperparameter_combinations:
        vbeta, vvar, layer = combo
        sample_scores = []
        for i in range(sample_n):
            rng.seed(i)
            sampled = rng.choice(all_image_ids)
            avg_dice = evaluate_on_sample(model, processor, tokenizer, text, [sampled], val_path, device, layer, vbeta, vvar)
            sample_scores.append(avg_dice)
        mean_dice = float(np.mean(sample_scores))
        results.append({'vbeta': vbeta, 'vvar': vvar, 'vlayer': layer, 'average_dice': mean_dice})

    # pick best
    best = max(results, key=lambda x: x['average_dice'])
    print(f"hyperopt best: {best}")
    return best


# Cached Vmap function (replaces your Vmap)
def Vmap(seed, text_simple, if_MedGemma, MedGemma_model, MedGemma_processor, device, data_root, category_list, image_name_list,
         model_for_heatmap, processor_for_heatmap, tokenizer_for_heatmap, clip_device, epoch, seg_initial=None, cache_dir="/data2/member/Qian/vmap_cache"):
    """
    For a batch of samples (category_list, image_name_list) produce:
      vlayer_list, form_list, text_list
    Implementation notes:
      - For each unique category in the batch, consult cache for best hyperparams.
      - If cache miss, run hyper_opt_once once (cheap sampling) and cache result.
      - Avoid expensive per-sample hyperopt.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ensure_dir(cache_dir)
    cache_file = os.path.join(cache_dir, "hyperopt_cache.json")
    vcache = VmapCache(cache_file)

    vlayer_list = []
    form_list = []
    text_list = []

    # We'll iterate samples; for each category, get best combo (from cache or compute)
    for i in range(len(category_list)):
        category_i = category_list[i]
        image_id = image_name_list[i]

        # choose dataset-specific val_path mapping (same as original)
        if category_i == 'polyp':
            data_name = 'polyp'
            prompt = 'This is the original Polyp Endoscopy.'
            target = 'polyp'
        elif category_i == 'skin melanoma':
            data_name = 'ISIC'
            prompt = 'This is the original Skin Lesion Photography.'
            target = 'skin melanoma'
        elif category_i == 'brain tumor':
            data_name = 'brain_tumor'
            prompt = 'This is the original T1-weighted contrast-enhanced Brain MRI.'
            target = 'brain tumor'
        elif category_i == 'breast tumor':
            data_name = 'breast_tumor'
            prompt = 'This is the original Breast Ultrasound image.'
            target = 'breast tumor'
        elif category_i == 'lung' and int(len(image_id)) < 32:
            data_name = 'lung_Xray'
            prompt = 'This is the original Chest X-Ray.'
            target = 'lung'
        else:
            data_name = 'lung_CT'
            prompt = 'This is the original Lung CT.'
            target = 'lung'
        val_path = f'/data2/member/Qian/open_vocabulary_segmentation_1.0/dataset/fully/{data_name}/val/images'

        if epoch <=30:
            cache_key = f"{data_name}"  # you can expand key (e.g., include model name) if needed
            cached = vcache.get(cache_key)
            if cached is None:
                # compute and cache
                print(f"[Vmap] cache miss for {cache_key}, running hyper_opt_once (this is slow — only once per category)")
                try:
                    best_combo = hyper_opt_once(model_for_heatmap, processor_for_heatmap, tokenizer_for_heatmap, category_i, val_path, device, sample_n=3)
                except Exception as e:
                    print("[Vmap] hyper_opt_once failed:", e)
                    # fallback default    ISIC(skin melanoma) 0.1/0.1/9 polyp 1/2/8 lung_Xray 0.1/0.1/7 lung_CT 0.1/0.1/8
                    if category_i == 'polyp':
                        best_combo = {'vbeta': 1.0, 'vvar': 2.0, 'vlayer': 8, 'average_dice': 0.0}
                    elif category_i == 'skin melanoma':
                        best_combo = {'vbeta': 0.1, 'vvar': 0.1, 'vlayer': 9, 'average_dice': 0.0}
                    elif category_i == 'lung' and int(len(image_id)) < 32:
                        best_combo = {'vbeta': 0.1, 'vvar': 0.1, 'vlayer': 7, 'average_dice': 0.0}
                    else:
                        best_combo = {'vbeta': 0.1, 'vvar': 0.1, 'vlayer': 8, 'average_dice': 0.0}
                vcache.set(cache_key, best_combo)
            else:
                best_combo = cached
        else:
            cache_key = f"{data_name}_MedGemma"  # you can expand key (e.g., include model name) if needed
            cached = vcache.get(cache_key)
            if cached is None:
                # compute and cache
                print(f"[Vmap] cache miss for {cache_key}, running hyper_opt_once (this is slow — only once per category)")
                try:
                    best_combo = hyper_opt_once(model_for_heatmap, processor_for_heatmap, tokenizer_for_heatmap, category_i, val_path, device, sample_n=3)
                except Exception as e:
                    print("[Vmap] hyper_opt_once failed:", e)
                    # fallback default
                    cache_key_default = f"{data_name}"  # you can expand key (e.g., include model name) if needed
                    cached_default = vcache.get(cache_key_default)
                    best_combo = cached_default
                vcache.set(cache_key, best_combo)
            else:
                best_combo = cached

        vbeta = best_combo['vbeta']
        vvar = best_combo['vvar']
        vlayer = int(best_combo['vlayer'])

        # Decide text for this sample
        image_path = os.path.join(data_root, 'images', image_id)
        gt_path_local = os.path.join(data_root, 'labels', image_id)
        image = Image.open(image_path).convert('RGB')
        gt = Image.open(gt_path_local).convert('RGB')

        if text_simple:
            text = category_i
            form = 'simple'
        else:
            if if_MedGemma:
                # Generate medgemma text using global medgemma model expected to be loaded externally
                # Here we expect medgemma_generate wrapper or external medgemma_model/processors to be available.
                # For safety, fallback to category text if medgemma unavailable.
                seg_pil = Image.fromarray(
                    (seg_initial[i] * 255).astype('uint8')) if seg_initial is not None else Image.new('L', image.size)
                # try:
                #     seg_pil = Image.fromarray((seg_initial[i] * 255).astype('uint8')) if seg_initial is not None else Image.new('L', image.size)
                # except Exception:
                #     seg_pil = Image.new('L', image.size)
                text = medgemma_generate(image, seg_pil, prompt, target, MedGemma_model, MedGemma_processor)
                text = text.replace('\n<end_of_turn>', '')
                form = 'MedGemma'
                MedGemma_dir = os.path.join('workdir', 'medgemma_review')
                os.makedirs(MedGemma_dir, exist_ok=True)
                result_dict = {image_id: text}
                with open(os.path.join(MedGemma_dir, f'medgemma_review_epoch{epoch}.txt'), 'a') as f:
                    f.write(str(result_dict) + '\n')
                # medgemma_generate is expected globally; if not, fall back.
                # try:
                #     text = medgemma_generate(image, seg_pil, MedGemma_model, MedGemma_processor)
                #     form = 'MedGemma'
                # except Exception:
                #     print("[Vmap] medgemma_generate failed, fallback to human prompt")
                #     text = category_i
                #     form = 'fallback'
            else:
                try:
                    text = human_interaction(np.array(image), np.array(seg_initial[i]) if seg_initial is not None else np.zeros_like(np.array(image)[:,:,0]),
                                             np.array(gt))
                    form = 'human'
                except Exception:
                    text = category_i
                    form = 'fallback'

        # generate vmap (single-sample) using chosen hyperparams
        image_feat = processor_for_heatmap(images=image, return_tensors="pt")['pixel_values'].to(clip_device).float()
        text_ids = torch.tensor([tokenizer_for_heatmap.encode(text, add_special_tokens=True)]).to(clip_device)

        vmap, _, _ = vision_heatmap_iba(text_ids, image_feat, model_for_heatmap, vlayer, vbeta, vvar, device=clip_device,
                                        ensemble=False, progbar=False)

        # Save or keep meta
        vlayer_list.append(vlayer)
        form_list.append(form)
        text_list.append(text)

        # Optionally save vmap to disk if needed (omitted here for speed)
        output_path = f'/data2/member/Qian/open_vocabulary_segmentation_1.0/dataset/fully/{data_name}/Vmap/simple_text'
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        img = np.array(image)
        vmap = cv2.resize(np.array(vmap), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay_image = create_cam_overlay(img, vmap, alpha=0.5)
        os.makedirs(f"{output_path}", exist_ok=True)
        name = image_id.replace('.png', f'_{form}.png')
        cv2.imwrite(f"{output_path}/{name}", vmap * 255)
        overlay_path = output_path.replace("Vmap", "overlay_vmap")
        os.makedirs(f"{overlay_path}", exist_ok=True)
        cv2.imwrite(f"{overlay_path}/{name}", overlay_image)

    return vlayer_list, form_list, text_list


def unpad_square_from_resized_array(resized_padded_arr: np.ndarray,
                                    orig_w: int,
                                    orig_h: int,
                                    padded_original_size: Optional[int] = None) -> np.ndarray:
    """
    从已被 cv2.resize 的 padded 图（numpy array HxWxC）裁出原始区域并返回 numpy 数组（与 cv2 风格 BGR/uint8 一致）。
    参数:
      - resized_padded_arr: numpy array, e.g. result of cv2.resize(padded_pil_arr, (new_w,new_h))
      - orig_w, orig_h: 原始图的宽高（pad 之前的尺寸）
      - padded_original_size: pad_to_square 函数中原先的方形边长 (max(orig_w, orig_h))。
          如果 None，会用 max(orig_w, orig_h) 作为默认（通常正确）。
    说明:
      - 假设 resize 时方形的宽高按相同比例缩放（通常是方形->方形）。
      - 返回裁切后的 numpy array (H=orig_h, W=orig_w, C).
    """
    h_new, w_new = resized_padded_arr.shape[:2]
    if padded_original_size is None:
        padded_original_size = max(orig_w, orig_h)  # 通常就是这样
    scale_x = w_new / padded_original_size
    scale_y = h_new / padded_original_size
    # 若 resize 保持纵横比且方形，则 scale_x == scale_y。我们分别计算以更稳健。
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
    # 若需要返回为原始像素尺寸 orig_w x orig_h，则对 cropped 再做一次 resize 恢复尺寸
    if (cropped.shape[1], cropped.shape[0]) != (orig_w, orig_h):
        cropped = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def unpad_square(padded_img: Image.Image, orig_w: int, orig_h: int):
    """
    从 pad 后的正方形图像中裁出原始区域。
    参数:
        padded_img: pad_to_square() 生成的方形 PIL 图像
        orig_w, orig_h: 原图的宽和高（在 pad 之前）
    返回:
        去掉 pad 的 PIL 图像（原始比例）
    """
    max_side = max(orig_w, orig_h)
    paste_x = (max_side - orig_w) // 2
    paste_y = (max_side - orig_h) // 2

    # 按原图区域 crop
    left = paste_x
    upper = paste_y
    right = paste_x + orig_w
    lower = paste_y + orig_h
    cropped_img = padded_img.crop((left, upper, right, lower))
    return cropped_img


def pad_to_square(img: Image.Image, fill_color=(0, 0, 0)):
    """
    将输入PIL图像 pad 成正方形（用指定颜色填充）。
    返回新的PIL图像。
    """
    w, h = img.size
    max_side = max(w, h)
    # 创建一个新的正方形背景
    new_img = Image.new("RGB", (max_side, max_side), fill_color)
    # 把原图 paste 到正中间（也可以贴左上角，看需求）
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def pad_to_square_gt(gt, fill_color=(0, 0, 0)):
    """
    将GT图像填充为正方形

    Args:
        gt: PIL Image 对象
        fill_color: 填充颜色，根据图像模式自动调整
    """
    # 获取图像尺寸和模式
    width, height = gt.size
    mode = gt.mode

    # 计算最大边长
    max_side = max(width, height)

    # 根据图像模式调整填充颜色
    if mode == "L":
        # 灰度模式：只需要一个值
        if isinstance(fill_color, (tuple, list)) and len(fill_color) >= 1:
            fill_color_single = fill_color[0]  # 取第一个值
        else:
            fill_color_single = fill_color
    elif mode == "RGB":
        # RGB模式：需要三个值
        if isinstance(fill_color, (int, float)):
            fill_color_single = (fill_color, fill_color, fill_color)
        elif isinstance(fill_color, (tuple, list)) and len(fill_color) == 3:
            fill_color_single = fill_color
        else:
            fill_color_single = (0, 0, 0)  # 默认黑色
    elif mode == "RGBA":
        # RGBA模式：需要四个值
        if isinstance(fill_color, (int, float)):
            fill_color_single = (fill_color, fill_color, fill_color, 255)
        elif isinstance(fill_color, (tuple, list)) and len(fill_color) == 3:
            fill_color_single = (*fill_color, 255)  # 添加alpha通道
        elif isinstance(fill_color, (tuple, list)) and len(fill_color) == 4:
            fill_color_single = fill_color
        else:
            fill_color_single = (0, 0, 0, 255)  # 默认黑色不透明
    else:
        # 其他模式，使用默认值
        fill_color_single = 0

    # 创建新的正方形图像
    new_img = Image.new(mode, (max_side, max_side), fill_color_single)

    # 计算粘贴位置（居中）
    paste_x = (max_side - width) // 2
    paste_y = (max_side - height) // 2

    # 粘贴原图像
    new_img.paste(gt, (paste_x, paste_y))

    return new_img


def pad_vmap_to_square(img: np.ndarray, fill_value=0):
    """
    专门处理vmap灰度图的pad版本
    """
    h, w = img.shape
    max_side = max(h, w)

    # 创建正方形背景
    new_img = np.full((max_side, max_side), fill_value, dtype=img.dtype)

    # 计算粘贴位置（居中）
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2

    # 粘贴原图
    new_img[paste_y:paste_y + h, paste_x:paste_x + w] = img

    return new_img


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("Loading models ...")

    # Load the appropriate model based on the arguments
    if (args.model_name == "BiomedCLIP-finetuned"):
        model = AutoModel.from_pretrained("/data2/member/Qian/FrozenModel/BiomedCLIP-vit-bert-hf", trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained("/data2/member/Qian/FrozenModel/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("/data2/member/Qian/FrozenModel/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    elif (args.model_name == "BiomedCLIP"):
        model = AutoModel.from_pretrained("/data2/member/Qian/FrozenModel/BiomedCLIP-vit-bert-hf", trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained("/data2/member/Qian/FrozenModel/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("/data2/member/Qian/FrozenModel/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    else:
        model = CLIPModel.from_pretrained("/data/member/Qian/FrozenModel/clip-vit-large-patch14").to(args.device)
        processor = CLIPProcessor.from_pretrained("/data/member/Qian/FrozenModel/clip-vit-large-patch14")
        tokenizer = CLIPTokenizerFast.from_pretrained("/data/member/Qian/FrozenModel/clip-vit-large-patch14")

    if args.use_SAM or args.proxy:
        SAM = build_sam_vit_h("/data2/member/Qian/FrozenModel/sam_checkpoints/sam_vit_h_4b8939.pth").to(args.device)
        SAM.requires_grad_(False)
    else:
        SAM = None

    simple_text = args.text

    # Perform hyperparameter optimization if required
    if args.hyper_opt:
        best_combo = hyper_opt(model, processor, tokenizer, SAM, simple_text, args)
        args.vbeta = best_combo['vbeta']
        args.vvar = best_combo['vvar']
        args.vlayer = int(best_combo['vlayer'])

    print("Generating Saliency Maps ...") #ISIC(skin melanoma) 0.1/0.1/9 polyp 1/2/8 lung_Xray 0.1/0.1/7 lung_CT 0.1/0.1/8

    # Iterate through the input images and generate saliency maps
    for image_id in tqdm(sorted(os.listdir(args.input_path))):
        try:
            original_image = Image.open(f"{args.input_path}/{image_id}").convert('RGB')
            orig_w, orig_h = original_image.size
            image = pad_to_square(original_image, fill_color=(0, 0, 0))
        except:
            print(f"Unable to load image at {image_id}", flush=True)
            continue

        if args.use_MedGemma:
            with open(args.json_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
            text_dict = {}
            lines = content.strip().split('\n')
            for line in lines:
                if line:
                    try:
                        data = eval(line)
                        text_dict.update(data)
                    except Exception as e:
                        print(f"Error parsing line: {line} - {e}")
            if image_id in text_dict:
                text_ = text_dict[image_id]
            else:
                print(f"Warning: {image_id} not found in TXT file")
                text_ = simple_text
        else:
            print("\n!!!use simle")
            text_ = simple_text

        if args.use_SAM:
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                image_np = cv2.resize(image_np, (1024, 1024))
                image_tensor = torch.from_numpy(image_np).float()
                # 如果图像是 RGB，需要调整维度顺序 (H, W, C) -> (C, H, W)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)
            else:
                image_tensor = image
            input_image = SAM.preprocess(image_tensor.to(args.device))
            with torch.no_grad():
                ex_feat = SAM.image_encoder(input_image.unsqueeze(0).to(args.device))
        else:
            ex_feat = None

        # Preprocess the image and tokenize the text
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = torch.tensor([tokenizer.encode(text_, add_special_tokens=True)]).to(args.device)

        # Generate visual saliency map
        if args.use_hard:
            # print("use hard!")
            vmap1 = cv2.imread(f"{args.vmap_root}/vmap/{args.vmap_type1}/{image_id}", 0).astype(np.uint8)
            vmap1 = pad_vmap_to_square(vmap1, fill_value=0)
            vmap2 = cv2.imread(f"{args.vmap_root}/vmap/{args.vmap_type2}/{image_id}", 0).astype(np.uint8)
            vmap2 = pad_vmap_to_square(vmap2, fill_value=0)
            H = np.stack([vmap1, vmap2], axis=0)
            difficulty = H.var(axis=0)
            difficulty_map = difficulty / difficulty.max()
            difficulty_map = torch.from_numpy(difficulty_map).to(args.device)
            gamma = 0.1 / (difficulty.mean() + 1e-6)
            difficulty_14 = F.interpolate(difficulty_map.unsqueeze(0).unsqueeze(0), size=(14, 14), mode="bilinear")
            difficulty_tokens = difficulty_14.reshape(-1)  # [196]
            difficulty_tokens = torch.cat([torch.zeros(1).to(args.device), difficulty_tokens])  # [197]
            difficulty_tokens = difficulty_tokens.unsqueeze(1)  # [197,1]
            text_ids2 = torch.tensor([tokenizer.encode(simple_text, add_special_tokens=True)]).to(args.device)
            vmap, _ = vision_heatmap_iba_hard(text_ids, text_ids2, image_feat, model, args.vlayer, args.vbeta, args.vvar, gamma, difficulty_tokens, device=args.device, ensemble=args.ensemble, progbar=False)
        elif args.use_SAM:
            vmap, _ = vision_heatmap_iba_proxy(text_ids, image_feat, ex_feat, model, args.vlayer, args.vbeta, args.vvar, device=args.device, ensemble=args.ensemble, progbar=False)
        else:
            vmap, _ = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar, device=args.device, ensemble=args.ensemble, progbar=False)

        if args.proxy:
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                image_np = cv2.resize(image_np, (1024, 1024))
                image_tensor = torch.from_numpy(image_np).float()
                # 如果图像是 RGB，需要调整维度顺序 (H, W, C) -> (C, H, W)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)
            else:
                image_tensor = image
            input_image = SAM.preprocess(image_tensor.to(args.device))
            with torch.no_grad():
                ex_feat = SAM.image_encoder(input_image.unsqueeze(0).to(args.device))
            B, C, H, W = ex_feat.shape
            q_k = F.normalize(ex_feat.flatten(2, 3), dim=1)  # [b,256,4096]
            similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)  # [b,4096,4096]
            similarity = (similarity - torch.mean(similarity) * 1.2) * 3.0
            similarity[similarity < 0.0] = float('-inf')
            attn_weights = F.softmax(similarity, dim=-1)
            vmap_tensor = torch.from_numpy(vmap).unsqueeze(0).unsqueeze(0)
            vmap_tensor = F.interpolate(vmap_tensor.float(), size=(H, W), mode='bilinear', align_corners=False)
            b, c, h, w = vmap_tensor.shape
            v = vmap_tensor.flatten(2).to(args.device)  # [b,1,4096]
            v = v.permute(0, 2, 1)  # [b,4096,1]
            attn_output = torch.bmm(attn_weights, v)  # [b,4096,1]
            attn_output = attn_output.permute(0, 2, 1)  # [b,1,4096]
            attn_output_vmap = attn_output.view(b, c, h, w).squeeze(0).squeeze(0)
            vmap = attn_output_vmap.cpu().numpy()

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
    parser.add_argument('--input-path', required=False, default="/data3/member/Qian/open_vocabulary_segmentation_1.0/dataset/fully/polyp/val_image/images", type=str, help='path to the images')
    parser.add_argument('--output-path1', required=False, default="/data3/member/Qian/open_vocabulary_segmentation_1.0/dataset/fully/polyp/overlay_vmap/gpt_text", type=str,
                        help='path to the output')
    parser.add_argument('--output-path2', required=False, default="/data3/member/Qian/open_vocabulary_segmentation_1.0/dataset/fully/polyp/vmap/gpt_text",
                        type=str, help='path to the output')
    parser.add_argument('--val-path', type=str, default="/data3/member/Qian/open_vocabulary_segmentation_1.0/dataset/fully/polyp/test_image/images",
                        help='path to the validation set for hyperparameter optimization')
    parser.add_argument("--vmap_root", type=str, default="/data2/member/Qian/open_vocabulary_segmentation_1.0/dataset/fully/polyp")
    parser.add_argument('--vmap_type1', type=str, default="simple_text")
    parser.add_argument('--vmap_type2', type=str, default="medgemma_text")
    parser.add_argument('--vbeta', type=float, default=2.0) #brain 2  skin melanoma 0.1  polyp 0.1  lung(Xray) 0.1 lung(CT) 0.1  breast 2
    parser.add_argument('--vvar', type=float, default=2.0) #brain 2  skin melanoma 2  polyp 2  lung(Xray) 1 lung(CT) 1  breast 2
    parser.add_argument('--vlayer', type=int, default=7) #brain 9  skin melanoma 9  polyp 7  lung(Xray) 7 lung(CT) 7  breast 7
    parser.add_argument('--model_name', type=str, default="BiomedCLIP-finetuned", help="Which CLIP model to use")
    parser.add_argument('--text', type=str, default="polyp")
    parser.add_argument('--use_SAM', default=False, type=bool, help="Whether to use SAM")
    parser.add_argument('--use_MedGemma', default=True)
    parser.add_argument('--use_hard', default=False)
    parser.add_argument('--proxy', default=False)
    parser.add_argument('--finetuned', action='store_true', help="Whether to use finetuned weights or not")
    parser.add_argument('--hyper_opt', default=True, help="Whether to optimize hyperparameters or not")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on")
    parser.add_argument('--ensemble', action='store_true', help="Whether to use text ensemble or not")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--json_path', type=str, default="/data2/member/Qian/work2/gpt_text_all/polyp_test.txt", help="Path to the JSON file containing the text prompts")
    parser.add_argument('--val_json_path', type=str,
                        default="/data2/member/Qian/work2/gpt_text_all/polyp_val.txt",
                        help="Path to the JSON file containing the text prompts")
    args = parser.parse_args()
    main(args)

    print("Saliency Map Generation Done!")