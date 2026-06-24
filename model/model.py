import math
from segment_anything.modeling.common import *
from segment_anything.modeling.transformer import Attention
import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 256,
            num_heads: int = 8,
            mlp_dim: int = 2048,
            act: Type[nn.Module] = nn.LeakyReLU,
            attention_downsample_rate: int = 2,
    ) -> None:

        super().__init__()
        self.norm_input = nn.LayerNorm([2, 64, 64])

        self.proj = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp_map = MLPBlock(embedding_dim, mlp_dim, act)
        self.norm_map = nn.LayerNorm(embedding_dim)

        self.cross_attn1 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, act)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn2 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.output_upscaling1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(embedding_dim // 2, embedding_dim // 4, kernel_size=3, stride=1, padding=1),
            act(),
            LayerNorm2d(embedding_dim // 4),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embedding_dim // 4, embedding_dim // 8, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(embedding_dim // 8, 1, kernel_size=1)
        )
        self.output_upscaling2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(embedding_dim // 2, embedding_dim // 4, kernel_size=3, stride=1, padding=1),
            act(),
            LayerNorm2d(embedding_dim // 4),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embedding_dim // 4, embedding_dim // 8, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(embedding_dim // 8, 1, kernel_size=1)
        )

    def forward(self, sam_image_feats: torch.Tensor, attribution_map: torch.Tensor):
        # [b,256,64,64] [b,1,64,64]

        B, C, H, W = sam_image_feats.shape
        q_k = F.normalize(sam_image_feats.flatten(2, 3), dim=1)  # [b,256,4096]
        similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)  # [b,4096,4096]
        similarity = (similarity - torch.mean(similarity) * 1.2) * 3.0
        similarity[similarity < 0.0] = float('-inf')
        attn_weights = F.softmax(similarity, dim=-1)  # [b,4096,4096]

        b, c, h, w = attribution_map.shape
        v = attribution_map.flatten(2)  # [b,1,4096]
        v = v.permute(0, 2, 1)  # [b,4096,1]
        attn_output = torch.bmm(attn_weights, v)  # [b,4096,1]
        attn_output = attn_output.permute(0, 2, 1)  # [b,1,4096]
        attn_output_map = attn_output.view(b, c, h, w)  # [b,1,64,64]

        attribution_map_final = torch.cat([attribution_map, attn_output_map], dim=1) #[b,2,h,w]
        attribution_map_normalized = self.norm_input(attribution_map_final)
        attribution_map_feature = self.proj(attribution_map_normalized)  # [b,256,64,64]
        attribution_map_flat = attribution_map_feature.flatten(2).permute(0, 2, 1)  # [b,h*w,256]
        attribution_map_feature = self.norm1(attribution_map_flat)  # [b,4096,256]
        attribution_map_feature_mlp = self.attribution_map_map(attribution_map_feature)
        attribution_map_feats = attribution_map_feature_mlp + attribution_map_feature
        attribution_map_feats = self.norm_map(attribution_map_feats)  # [b, 4096, 256]

        sam_image_feats = sam_image_feats.flatten(2).permute(0, 2, 1)  # [b,4096,256]

        attn_out = self.cross_attn1(q=sam_image_feats, k=attribution_map_feats, v=attribution_map_feats)
        queries1 = sam_image_feats + attn_out
        queries1 = self.norm2(queries1)  # [b,4096,256]

        mlp_out = self.mlp(queries1)
        mlp_queries = queries1 + mlp_out
        mlp_queries = self.norm3(mlp_queries)

        attn_out = self.cross_attn2(q=mlp_queries, k=sam_image_feats, v=sam_image_feats)
        keys = mlp_queries + attn_out
        keys = self.norm4(keys)

        queries1 = queries1.permute(0, 2, 1)  # [b,256,4096]
        keys = keys.permute(0, 2, 1)  # [b,256,4096]
        size = int(math.sqrt(queries1.shape[2]))
        queries = queries1.view(queries1.shape[0], queries1.shape[1], size, size)  # [b,256,64,64]
        queries = self.output_upscaling1(queries)  # [b,1,256,256]

        keys = keys.view(keys.shape[0], keys.shape[1], size, size)  # [b,256,64,64]
        keys = self.output_upscaling2(keys)  # [b,1,256,256]

        return queries + keys
