# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import random
from typing import Type
import os
from PIL import Image
import cv2
from scripts.methods import *
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        #act: Type[nn.Module] = nn.GELU,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=512, embedding_dim=256, num_classes=20, index=11, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.indexes = index #6 #11
        linear_layers = [FeatureAdapter(input_dim=self.in_channels, embed_dim=embedding_dim) for i in range(self.indexes)]
        self.linears_modulelist = nn.ModuleList(linear_layers)
        self.linear_fuse = nn.Conv2d(embedding_dim*self.indexes, embedding_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x_all):
        x_list = []
        for ind in range(x_all.shape[0]):
            x = x_all[ind,:, :, :, :]
            n, _, h, w = x.shape
            x_r = x.flatten(2).transpose(1, 2)
            _x = self.linears_modulelist[ind](x_r.float()) #[1,hw,c]
            _x = _x.permute(0,2,1).reshape(n, -1, x.shape[2], x.shape[3]) #(b, c, h, w)
            x_list.append(_x)
        x_list = torch.cat(x_list, dim=1) #(b, c, h, w)
        x = self.linear_fuse(x_list)
        x = self.dropout(x)
        return x

class FeatureAdapter(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=512, embed_dim=256):
        super().__init__()
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.proj_1 = nn.Linear(input_dim, embed_dim)
        self.proj_2 = nn.Linear(embed_dim, embed_dim)
        # self.norm = norm_layer(embed_dim)
        # self.proj_3 = nn.Linear(embed_dim*2, embed_dim)

    def forward(self, x):
        x = x.flatten(2)
        x = self.proj_1(x)
        x = F.relu(x)
        x = self.proj_2(x)
        # x = self.norm(x)
        return x

def generate_vmap(batch, input_path, model, processor, tokenizer, device, mode):
    seed = 12
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Perform hyperparameter optimization if required
    vbeta = 1.0
    vvar = 1.0
    vlayer = 9

    # Iterate through the input images and generate saliency maps
    image_names = batch["image_name"]
    bs = len(image_names)

    vmap_all = []
    features_all = []
    feature_list_all = []
    for i in range(bs):
        categorys = batch["category"][i]
        if mode == "train":
            dataset = batch["dataset"][i]  # train
            input_path_i = os.path.join(input_path, dataset, "images")  # train
        else:
            input_path_i = os.path.join(input_path, "images")

        image_name_i = batch["image_name"][i]
        image_i = Image.open(f"{input_path_i}/{image_name_i}").convert('RGB')
        text_i = categorys[i]

        # Preprocess the image and tokenize the text
        inputs = processor(images=image_i, return_tensors="pt")['pixel_values'].to(device) #[1,3,224,224]
        text_ids = torch.tensor([tokenizer.encode(text_i, add_special_tokens=True)]).to(device)

        # Generate visual saliency map & image features
        vmap_i, feature, attn_weight_list = vision_heatmap_iba(text_ids, inputs, model, vlayer, vbeta, vvar, device, ensemble='store_true', progbar=False)
        vmap_resized_i = cv2.resize(np.array(vmap_i), (64, 64), interpolation=cv2.INTER_NEAREST)
        vmap_all.append(torch.tensor(vmap_resized_i))
        features_all.append(feature[:,1:,:]) # remove cls
        feature_list_all.append(attn_weight_list) #[4,h+1*w+1,c]

    maps = torch.stack(vmap_all).to(device)  #[batch_size,64,64]
    maps = maps.unsqueeze(1)  #[batch_size,1,64,64]
    visual_features = torch.stack(features_all).to(device)
    attn_weights = torch.stack(feature_list_all).to(device)

    return maps, visual_features, attn_weights

def refine_cams_with_aff_CRF(attr_map, attn_weights, n_iter):
    b, h, w = attr_map.shape
    hw = h * w
    # flatten
    A_flat = attr_map.view(b, hw, 1)  # [b,HW,1]

    # normalize affinity map (row-normalize, 保证和为1)
    W_norm = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)

    # propagation
    A_tmp = A_flat
    for _ in range(n_iter):
        A_tmp = torch.bmm(W_norm, A_flat)

    # reshape back
    A_refined = A_tmp.view(b, h, w)

    return A_refined

def get_clip_features(batch, input_path, CLIP, processor, tokenizer, device):
    CLIP.eval()
    image_names = batch["image_name"]
    categorys = batch["category"]
    batch_size = len(categorys)
    input_path = os.path.join(input_path, "images")
    image_feature_list = []
    text_feature_list = []
    with torch.no_grad():
        for i in range(batch_size):
            image_name = image_names[i]
            image = Image.open(f"{input_path}/{image_name}").convert('RGB')
            text = categorys[i]

            # Preprocess the image and tokenize the text
            image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
            text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
            states_image = CLIP.vision_model(image_feat, output_hidden_states=True)
            image_feature = states_image['hidden_states'][10]  # [1,257,1024]
            image_feature = image_feature[:, 1:, :]  # [1,256,1024]
            states_text = CLIP.text_model(text_ids, output_hidden_states=True)
            text_feature = states_text[0]  # [1,3,768]
            repeat_times = image_feature.size(1) // text_feature.size(1)  # 65
            remainder = image_feature.size(1) % text_feature.size(1)
            expanded_text = text_feature.repeat(1, repeat_times, 1)  # [B, 3*65=195, D]
            if remainder > 0:
                expanded_text = torch.cat([expanded_text, text_feature[:, :remainder, :]], dim=1)
            image_feature_list.append(image_feature.squeeze(0))
            text_feature_list.append(expanded_text.squeeze(0))

        image_features = torch.stack(image_feature_list).to(device)
        text_features = torch.stack(text_feature_list).to(device)
        return image_features, text_features # [1,256,1024] [1,256,768]