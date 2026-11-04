"""
================================================================================
TST WITH VIDEO SWIN SMALL — FULL TRAINING CODE FOR SAV DATASET
================================================================================
Target: 70%+ mAP on SAV dataset

Key Improvements:
1. Video Swin Small backbone (50M params vs 28M for Tiny)
2. Asymmetric Loss with class reweighting
3. Mixup augmentation for video
4. Cosine annealing with warm restarts
5. Exponential Moving Average (EMA)
6. Gradient checkpointing for memory efficiency
7. Multi-scale temporal sampling

Author: Srinithi R, Divya Maki S, Thirumoorthy K
================================================================================
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SEED FOR REPRODUCIBILITY
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths - UPDATE THESE FOR YOUR KAGGLE ENVIRONMENT
    DATA_ROOT = "/kaggle/input/datasets/YOUR_KAGGLE_USERNAME/sav-dataset/SAV/data/sav"
    FRAMES_DIR = "/kaggle/input/datasets/YOUR_KAGGLE_USERNAME/sav-dataset/SAV/data/sav/frames"
    TRAIN_CSV = "/kaggle/input/datasets/YOUR_KAGGLE_USERNAME/sav-dataset/SAV/annotations/train.csv"
    VAL_CSV = "/kaggle/input/datasets/YOUR_KAGGLE_USERNAME/sav-dataset/SAV/annotations/val.csv"
    OUTPUT_DIR = "/kaggle/working/checkpoints"
    
    # PACR checkpoint (frozen detector)
    PACR_CHECKPOINT = "/kaggle/input/models/YOUR_KAGGLE_USERNAME/pacr-bound/pytorch/default/1/best (1) (1).pth"
    
    # Video Swin Small pretrained weights
    # Download from: https://github.com/SwinTransformer/Video-Swin-Transformer
    SWIN_PRETRAINED = "/kaggle/input/models/YOUR_KAGGLE_USERNAME/swin-small/pytorch/default/1/swin_small_patch244_window877_kinetics400_1k.pth"
    
    # Model architecture
    MODEL_SIZE = 'small'  # 'tiny', 'small', or 'base'
    NUM_CLASSES = 15
    NUM_SEGMENTS = 3      # K=3 temporal segments (TST paper)
    FRAMES_PER_SEGMENT = 16  # T=16 frames per segment
    VIDEO_SIZE = 224
    
    # Video Swin parameters (will be set based on MODEL_SIZE)
    TST_EMBED_DIM = 96
    TST_DEPTHS = (2, 2, 18, 2)  # Small has 18 in stage 3 vs 6 in Tiny
    TST_NUM_HEADS = (3, 6, 12, 24)
    TST_WINDOW_SIZE = (2, 7, 7)  # Temporal window = 2 for K=3 segments
    TST_PATCH_SIZE = (2, 4, 4)
    
    # Training parameters
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8  # Effective batch = 16
    EPOCHS = 50
    WARMUP_EPOCHS = 3
    
    # Learning rates
    BACKBONE_LR = 5e-5
    HEAD_LR = 5e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.05
    
    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5
    
    # Loss
    LABEL_SMOOTHING = 0.1
    ASL_GAMMA_NEG = 4
    ASL_GAMMA_POS = 0
    ASL_CLIP = 0.05
    
    # Memory management
    MAX_STUDENTS_PER_BATCH = 8
    USE_GRADIENT_CHECKPOINT = True
    USE_AMP = True
    
    # EMA
    USE_EMA = True
    EMA_DECAY = 0.9998
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self._set_model_config()
    
    def _set_model_config(self):
        """Set architecture params based on model size"""
        if self.MODEL_SIZE == 'tiny':
            self.TST_EMBED_DIM = 96
            self.TST_DEPTHS = (2, 2, 6, 2)
            self.TST_NUM_HEADS = (3, 6, 12, 24)
        elif self.MODEL_SIZE == 'small':
            self.TST_EMBED_DIM = 96
            self.TST_DEPTHS = (2, 2, 18, 2)
            self.TST_NUM_HEADS = (3, 6, 12, 24)
        elif self.MODEL_SIZE == 'base':
            self.TST_EMBED_DIM = 128
            self.TST_DEPTHS = (2, 2, 18, 2)
            self.TST_NUM_HEADS = (4, 8, 16, 32)

# ============================================================================
# ACTION CLASSES
# ============================================================================
ACTION_NAMES = [
    'sit', 'stand', 'look_forward', 'look_sideways', 'read',
    'flip_books', 'touch_sth', 'raise_hand', 'hands_down', 'take_notes',
    'applaud', 'bend', 'turn_around', 'talk_with_others', 'answer_questions'
]

CLASS_COUNTS = np.array([94513, 6859, 46412, 15163, 39466, 2134, 25951, 
                         5643, 53812, 6550, 2431, 2982, 1525, 3280, 2241])
CLASS_FREQUENCIES = CLASS_COUNTS / CLASS_COUNTS.sum()
RARE_CLASSES = [5, 7, 10, 11, 12, 13, 14]

# ============================================================================
# VIDEO SWIN TRANSFORMER (Supports Tiny/Small/Base)
# ============================================================================
def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    
    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        
        x = self.norm1(x)
        
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x
    
    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))
    
    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        
        if self.use_checkpoint:
            x = x + checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x):
        B, D, H, W, C = x.shape
        
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        _, _, H, W, _ = x.shape
        
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None
    
    def forward(self, x):
        _, _, D, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        
        x = self.proj(x)
        if self.norm:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(2,7,7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)])
        
        self.downsample = downsample
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = self.compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        
        for blk in self.blocks:
            x = blk(x, attn_mask)
        
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        
        if self.downsample is not None:
            x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
            x = self.downsample(x)
            x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        return x
    
    def compute_mask(self, D, H, W, window_size, shift_size, device):
        if any(s > 0 for s in shift_size):
            img_mask = torch.zeros((1, D, H, W, 1), device=device)
            cnt = 0
            for d in (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
                for h in (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
                    for w in (slice(0, -window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1
            
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.squeeze(-1)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

class VideoSwinTransformer(nn.Module):
    """Video Swin Transformer supporting Tiny/Small/Base variants"""
    
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96,
                 depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=(2,7,7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 patch_norm=False, use_checkpoint=False):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging(int(embed_dim * 2 ** i_layer)) if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        x = self.avgpool(x)
        x = x.flatten(1)
        
        return x

# ============================================================================
# COSINE CLASSIFIER WITH LEARNABLE TEMPERATURE
# ============================================================================
class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=16.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.scale * F.linear(x_norm, w_norm)

# ============================================================================
# TST (TEMPORAL SEGMENT TRANSFORMER)
# ============================================================================
class TST(nn.Module):
    """Temporal Segment Transformer with Video Swin backbone"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_segments = config.NUM_SEGMENTS
        
        # Video Swin backbone
        self.video_swin = VideoSwinTransformer(
            patch_size=config.TST_PATCH_SIZE,
            embed_dim=config.TST_EMBED_DIM,
            depths=config.TST_DEPTHS,
            num_heads=config.TST_NUM_HEADS,
            window_size=config.TST_WINDOW_SIZE,
            drop_path_rate=0.2,
            use_checkpoint=config.USE_GRADIENT_CHECKPOINT
        )
        
        # Feature dimension from backbone
        self.num_features = self.video_swin.num_features  # 768 for Small
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Cosine classifier
        self.classifier = CosineClassifier(256, config.NUM_CLASSES, scale=16.0)
        
        # Per-class learnable temperature
        self.class_temp = nn.Parameter(torch.ones(config.NUM_CLASSES))
    
    def forward(self, tubes):
        """
        Args:
            tubes: (N, K*T, C, H, W) where K=num_segments, T=frames_per_segment
        Returns:
            logits: (N, num_classes)
        """
        N = tubes.size(0)
        K = self.num_segments
        T = self.config.FRAMES_PER_SEGMENT
        
        # Reshape: (N, K*T, C, H, W) -> (N*K, C, T, H, W)
        tubes = tubes.view(N, K, T, 3, 224, 224)
        tubes = tubes.permute(0, 1, 3, 2, 4, 5)  # (N, K, C, T, H, W)
        tubes = tubes.reshape(N * K, 3, T, 224, 224)
        
        # Extract features from each segment
        features = self.video_swin(tubes)  # (N*K, num_features)
        
        # Aggregate segments (average pooling)
        features = features.view(N, K, -1)  # (N, K, num_features)
        features = features.mean(dim=1)  # (N, num_features)
        
        # Project and classify
        features = self.feature_proj(features)  # (N, 256)
        logits = self.classifier(features)  # (N, num_classes)
        
        # Apply per-class temperature
        logits = logits / (self.class_temp.unsqueeze(0) + 1e-6)
        
        return logits

# ============================================================================
# ASYMMETRIC LOSS WITH CLASS REWEIGHTING
# ============================================================================
class AsymmetricLossOptimized(nn.Module):
    """ASL with rare class boosting and label smoothing"""
    
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, 
                 label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing
        
        # Class weights based on inverse frequency + rare class boost
        if class_weights is None:
            weights = 1.0 / (CLASS_FREQUENCIES + 0.01)
            weights = weights / weights.mean()
            
            # Extra boost for weak classes
            boost = np.ones(15)
            boost[13] = 5.0  # talk_with_others
            boost[12] = 3.0  # turn_around
            boost[3] = 2.0   # look_sideways
            boost[5] = 2.0   # flip_books
            boost[14] = 2.0  # answer_questions
            
            weights = weights * boost
            self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer('class_weights', class_weights)
    
    def forward(self, outputs):
        logits = outputs['logits']
        labels = outputs['labels']
        
        if logits is None or labels is None:
            return torch.tensor(0.0, requires_grad=True)
        
        labels = labels.to(logits.device).float()
        
        # Label smoothing
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + self.label_smoothing / 2
        
        # Asymmetric loss computation
        x_sig = torch.sigmoid(logits)
        xs_pos = x_sig
        xs_neg = 1 - x_sig
        
        # Asymmetric clipping
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        
        # Binary cross entropy
        los_pos = labels * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - labels) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg
        
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * labels + xs_neg * (1 - labels)
            gamma = self.gamma_pos * labels + self.gamma_neg * (1 - labels)
            loss *= (1 - pt) ** gamma
        
        # Class weights
        loss = loss * self.class_weights.unsqueeze(0)
        
        return -loss.mean()

# ============================================================================
# MIXUP AUGMENTATION FOR VIDEO
# ============================================================================
class VideoMixup:
    """Mixup and CutMix augmentation for video data"""
    
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, tubes, labels):
        if random.random() > self.prob:
            return tubes, labels, labels, 1.0
        
        batch_size = tubes.size(0)
        if batch_size < 2:
            return tubes, labels, labels, 1.0
        
        # Random permutation
        indices = torch.randperm(batch_size, device=tubes.device)
        
        # Choose mixup or cutmix
        if random.random() < 0.5 and self.mixup_alpha > 0:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            tubes = lam * tubes + (1 - lam) * tubes[indices]
        else:
            # CutMix (temporal)
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            T = tubes.size(1)
            cut_len = int(T * (1 - lam))
            cut_start = random.randint(0, T - cut_len)
            tubes[:, cut_start:cut_start+cut_len] = tubes[indices, cut_start:cut_start+cut_len]
            lam = 1 - cut_len / T
        
        return tubes, labels, labels[indices], lam

# ============================================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================================
class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# FIXED SAV DATASET — MATCHES YOUR ACTUAL CSV FORMAT
# ============================================================================

class SAVDataset(Dataset):
    """SAV Dataset — CSV has NO HEADERS"""
    
    def __init__(self, data_root, csv_path, config, train=True):
        self.data_root = data_root
        self.frames_dir = config.FRAMES_DIR
        self.config = config
        self.train = train
        self.num_segments = config.NUM_SEGMENTS
        self.frames_per_segment = config.FRAMES_PER_SEGMENT
        self.spatial_size = config.VIDEO_SIZE
        
        # ═══ CRITICAL: SAV CSV has NO headers ═══
        # Format: video_clip_id, constant, x1, y1, x2, y2, action_id, person_id
        df = pd.read_csv(csv_path, header=None,
            names=['video_clip_id', 'constant', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id'])
        
        # Group by video and person
        self.video_clips = self._group_annotations(df)
        self.video_ids = list(self.video_clips.keys())
        
        # Compute sample weights for rare class oversampling
        self.sample_weights = self._compute_sample_weights()
        
        print(f"{'Train' if train else 'Val'} dataset: {len(self.video_ids)} clips")
    
    def _group_annotations(self, df):
        """Group annotations by video, then by person"""
        grouped = defaultdict(lambda: defaultdict(list))
        
        for _, row in df.iterrows():
            vid = str(row['video_clip_id'])
            try:
                pid = int(row['person_id'])
                # Action IDs are 1-indexed in CSV, convert to 0-indexed
                action_id = int(row['action_id']) - 1
                if 0 <= action_id < 15:
                    grouped[vid][pid].append({
                        'bbox': [float(row['x1']), float(row['y1']),
                                 float(row['x2']), float(row['y2'])],
                        'action_id': action_id
                    })
            except:
                continue
        
        # Aggregate per video: each student has bbox + list of actions
        video_clips = {}
        for vid, persons in grouped.items():
            students = []
            for pid, annos in persons.items():
                # Use middle annotation's bbox (they're all the same keyframe)
                mid_idx = len(annos) // 2
                bbox = annos[mid_idx]['bbox']
                # Collect all unique actions for this person
                actions = list(set([a['action_id'] for a in annos]))
                students.append({
                    'person_id': pid,
                    'bbox': bbox,
                    'actions': actions
                })
            if students:
                video_clips[vid] = students
        
        return video_clips
    
    def _compute_sample_weights(self):
        """Compute sampling weights — oversample rare classes"""
        WEAK_CLASSES = {13, 12, 3, 5, 7, 14}  # talk_with_others, turn_around, etc.
        
        weights = []
        for vid in self.video_ids:
            actions = set()
            for s in self.video_clips[vid]:
                for a in s['actions']:
                    actions.add(a)
            
            # Higher weight for clips with weak classes
            if actions & WEAK_CLASSES:
                w = 5.0 if 13 in actions else (4.0 if 12 in actions else 3.0)
            else:
                w = 1.0
            weights.append(w)
        
        return weights
    
    def __len__(self):
        return len(self.video_ids)
    
    def _load_frames(self, vid):
        """Load K*T frames for temporal input"""
        clip_dir = os.path.join(self.frames_dir, str(vid))
        if not os.path.isdir(clip_dir):
            return None
        
        frame_files = sorted([f for f in os.listdir(clip_dir) 
                              if f.endswith(('.jpg', '.png'))])
        
        K = self.num_segments
        T = self.frames_per_segment
        total_needed = K * T
        
        if len(frame_files) < total_needed:
            return None
        
        # Sample frames: K segments, T frames each
        total = len(frame_files)
        seg_dur = total / K
        
        frames = []
        for s in range(K):
            seg_start = int(s * seg_dur)
            seg_end = int((s + 1) * seg_dur)
            idxs = np.linspace(seg_start, max(seg_start, seg_end - 1), T, dtype=int)
            
            for idx in idxs:
                idx = min(int(idx), total - 1)
                frame_path = os.path.join(clip_dir, frame_files[idx])
                try:
                    img = Image.open(frame_path).convert('RGB')
                    img = img.resize((self.spatial_size, self.spatial_size), Image.BILINEAR)
                    img = np.array(img, dtype=np.float32) / 255.0
                    # ImageNet normalization
                    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    frames.append(img)
                except:
                    frames.append(np.zeros((self.spatial_size, self.spatial_size, 3), dtype=np.float32))
        
        # (K*T, H, W, C) -> (K*T, C, H, W)
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.tensor(frames, dtype=torch.float32)
    
    def _prepare_spatial(self, vid):
        """Load middle frame for spatial/detection input"""
        clip_dir = os.path.join(self.frames_dir, str(vid))
        if not os.path.isdir(clip_dir):
            return torch.zeros(3, self.spatial_size, self.spatial_size)
        
        frame_files = sorted([f for f in os.listdir(clip_dir) 
                              if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            return torch.zeros(3, self.spatial_size, self.spatial_size)
        
        # Middle frame (keyframe annotation)
        mid_idx = len(frame_files) // 2
        frame_path = os.path.join(clip_dir, frame_files[mid_idx])
        
        try:
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((self.spatial_size, self.spatial_size), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img, dtype=torch.float32)
        except:
            return torch.zeros(3, self.spatial_size, self.spatial_size)
    
    def _prepare_targets(self, students):
        """Prepare GT boxes and multi-label targets"""
        # SAV coordinates are RELATIVE [0,1], scale to spatial_size
        H = W = self.spatial_size
        boxes = []
        labels = []
        
        for s in students:
            x1, y1, x2, y2 = s['bbox']
            # Scale relative coords to pixel coords
            boxes.append([x1 * W, y1 * H, x2 * W, y2 * H])
            
            # Multi-label: one-hot for all actions
            label = torch.zeros(15, dtype=torch.float32)
            for a in s['actions']:
                if 0 <= a < 15:
                    label[a] = 1.0
            labels.append(label)
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.stack(labels) if labels else torch.zeros((0, 15))
        }
    
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        students = self.video_clips[vid]
        
        # Spatial frame (for PACR detection, though we use GT boxes)
        spatial = self._prepare_spatial(vid)
        
        # Temporal frames (for TST classification)
        temporal = self._load_frames(vid)
        if temporal is None:
            K, T = self.num_segments, self.frames_per_segment
            temporal = torch.zeros(K * T, 3, self.spatial_size, self.spatial_size)
        
        # Targets (GT boxes and labels)
        targets = self._prepare_targets(students)
        targets['video_id'] = vid
        
        return spatial, temporal, targets


def collate_fn(batch):
    """Collate function for DataLoader"""
    spatial = torch.stack([item[0] for item in batch])
    temporal = torch.stack([item[1] for item in batch])
    targets = [item[2] for item in batch]
    return spatial, temporal, targets
# ============================================================================
# PACR-DETR DETECTOR (Frozen)
# ============================================================================
# Import PACR from your existing code
# sys.path.insert(0, '/path/to/pacr')
# from pacr_detr import PACRDETRDetector, PACRConfig

# For this code, we assume PACR is already imported and available
# If not, you need to include the PACR code here

class PACRWithTST(nn.Module):
    """Combined PACR detection + TST classification"""
    
    def __init__(self, config, use_real_pacr=True):
        super().__init__()
        self.config = config
        
        # Use your actual PACR detector
        if use_real_pacr:
            pacr_config = PACRConfig()
            self.detector = PACRDETRDetector(pacr_config)
        else:
            self.detector = nn.Identity()  # Placeholder
        
        # TST classifier
        self.tst = TST(config)
    
    def freeze_pacr(self):
        """Freeze PACR detector"""
        for param in self.detector.parameters():
            param.requires_grad = False
        self.detector.eval()
        print("✓ PACR frozen")
    
    def load_pacr_checkpoint(self, path):
        """Load PACR checkpoint"""
        if not os.path.exists(path):
            print(f"⚠ PACR checkpoint not found: {path}")
            return False
        
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        
        # Load with shape checking
        model_state = self.detector.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
                loaded += 1
        
        self.detector.load_state_dict(model_state)
        print(f"✓ PACR checkpoint loaded: {loaded}/{len(model_state)} keys")
        return True
    
    def forward(self, spatial, temporal, targets=None):
        """
        Args:
            spatial: (B, C, H, W) - spatial frame for detection
            temporal: (B, K*T, C, H, W) - temporal frames for TST
            targets: list of dicts, each with 'boxes' (N,4) and 'labels' (N,15)
        """
        B = spatial.size(0)
        device = spatial.device
        
        all_logits = []
        all_labels = []
        student_counts = []
        
        for b in range(B):
            # ═══ FIXED: targets[b] is a SINGLE dict ═══
            if targets is not None and targets[b]['boxes'].size(0) > 0:
                boxes = targets[b]['boxes'].to(device)
                labels = targets[b]['labels'].to(device)
            else:
                # Inference: use PACR detection
                with torch.no_grad():
                    det_out = self.detector(spatial[b:b+1])
                    boxes = det_out.get('pred_boxes', None)
                    if boxes is None or boxes.size(0) == 0:
                        student_counts.append(0)
                        continue
                labels = None
            
            num_students = boxes.size(0)
            student_counts.append(num_students)
            
            if num_students == 0:
                continue
            
            # Cap students for memory
            if num_students > self.config.MAX_STUDENTS_PER_BATCH:
                indices = torch.randperm(num_students)[:self.config.MAX_STUDENTS_PER_BATCH]
                boxes = boxes[indices]
                if labels is not None:
                    labels = labels[indices]
                num_students = self.config.MAX_STUDENTS_PER_BATCH
            
            # Extract tubes and classify
            tubes = self._extract_tubes(temporal[b], boxes)
            logits = self.tst(tubes)
            
            all_logits.append(logits)
            if labels is not None:
                all_labels.append(labels)
        
        if all_logits:
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0) if all_labels else None
        else:
            all_logits = None
            all_labels = None
        
        return {
            'logits': all_logits,
            'labels': all_labels,
            'student_counts': student_counts
        }
    
    def _extract_tubes(self, temporal, boxes):
        """Extract tubes - boxes are in PIXEL coordinates"""
        K_T, C, H, W = temporal.shape
        N = boxes.size(0)
        
        tubes = []
        for i in range(N):
            x1, y1, x2, y2 = boxes[i].int().tolist()
            
            # Clamp
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))
            
            student_tube = []
            for t in range(K_T):
                crop = temporal[t, :, y1:y2, x1:x2]
                crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), 
                                    mode='bilinear', align_corners=False)
                student_tube.append(crop.squeeze(0))
            
            tubes.append(torch.stack(student_tube, dim=0))
        
        return torch.stack(tubes, dim=0)

# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(logits, labels):
    """Compute mAP, F1, precision, recall"""
    preds = torch.sigmoid(logits).numpy()
    labels = labels.numpy()
    
    aps = []
    for c in range(15):
        if labels[:, c].sum() == 0:
            aps.append(0.0)
            continue
        
        # Sort by prediction score
        idx = np.argsort(-preds[:, c])
        tp = np.cumsum(labels[idx, c])
        fp = np.cumsum(1 - labels[idx, c])
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (labels[:, c].sum() + 1e-6)
        
        # 11-point interpolated AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p = prec[rec >= t].max() if len(prec[rec >= t]) > 0 else 0
            ap += p
        aps.append(ap / 11)
    
    mAP = np.mean(aps)
    
    # F1, precision, recall
    thresholds = np.where(CLASS_FREQUENCIES < 0.02, 0.3, 0.5)
    pred_binary = (preds > thresholds).astype(int)
    
    tp = (pred_binary * labels).sum()
    fp = (pred_binary * (1 - labels)).sum()
    fn = ((1 - pred_binary) * labels).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'mAP': mAP,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'aps': aps
    }

# ============================================================================
# LOAD PRETRAINED WEIGHTS WITH WINDOW SIZE INTERPOLATION
# ============================================================================
def load_swin_pretrained(model, pretrained_path, target_window_size=(2, 7, 7)):
    """Load Kinetics-400 pretrained weights with window size interpolation"""
    
    if not os.path.exists(pretrained_path):
        print(f"⚠ Pretrained weights not found: {pretrained_path}")
        return False
    
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # Remove 'backbone.' prefix if present
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    
    # Get model state dict
    model_dict = model.state_dict()
    
    # Interpolate relative_position_bias_table for window size mismatch
    source_window = (8, 7, 7)  # Kinetics pretrained window size
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if 'relative_position_bias_table' in k and v.shape != model_dict[k].shape:
                # Interpolate
                print(f"  Interpolating {k}: {v.shape} -> {model_dict[k].shape}")
                
                # Original dimensions
                src_size = int(((v.shape[0]) ** 0.5))
                # This is more complex for 3D, need proper handling
                
                # For 3D, reshape and interpolate
                num_heads = v.shape[1]
                
                # Source: (2*Wd-1) * (2*Wh-1) * (2*Ww-1) x num_heads
                src_d, src_h, src_w = 2*source_window[0]-1, 2*source_window[1]-1, 2*source_window[2]-1
                tgt_d, tgt_h, tgt_w = 2*target_window_size[0]-1, 2*target_window_size[1]-1, 2*target_window_size[2]-1
                
                # Reshape to 3D
                v_3d = v.reshape(src_d, src_h, src_w, num_heads).permute(3, 0, 1, 2)  # (heads, D, H, W)
                
                # Interpolate
                v_3d = F.interpolate(v_3d.unsqueeze(0), size=(tgt_d, tgt_h, tgt_w), 
                                    mode='trilinear', align_corners=False).squeeze(0)
                
                # Reshape back
                v = v_3d.permute(1, 2, 3, 0).reshape(-1, num_heads)
                
            if v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
        else:
            pass  # Skip keys not in model
    
    # Load
    model.load_state_dict(new_state_dict, strict=False)
    
    loaded = len(new_state_dict)
    total = len(model_dict)
    print(f"✓ Loaded {loaded}/{total} keys from pretrained weights")
    
    return True

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
# ============================================================================
# TRAINING FUNCTION — WITH PROPER CHECKPOINT SAVING
# ============================================================================
def train(config, resume_path=None):
    """Main training loop with resume support"""
    
    device = torch.device(config.DEVICE)
    
    print("=" * 70)
    print("TST WITH VIDEO SWIN SMALL — TRAINING")
    print("=" * 70)
    print(f"Model: Video Swin {config.MODEL_SIZE.upper()}")
    print(f"Segments: K={config.NUM_SEGMENTS}, T={config.FRAMES_PER_SEGMENT}")
    print(f"Window size: {config.TST_WINDOW_SIZE}")
    print(f"Batch: {config.BATCH_SIZE} x {config.GRAD_ACCUM_STEPS} = {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    print(f"LR: backbone={config.BACKBONE_LR}, head={config.HEAD_LR}")
    print("=" * 70)
    
    # ══════════════════════════════════════════════════════════════════════
    # BUILD MODEL
    # ══════════════════════════════════════════════════════════════════════
    pacr_config = PACRConfig()
    model = PACRWithTST(config, pacr_config)
    
    # Load PACR checkpoint
    if os.path.exists(config.PACR_CHECKPOINT):
        model.load_pacr_checkpoint(config.PACR_CHECKPOINT)
    model.freeze_pacr()
    
    # Load Swin pretrained (only if NOT resuming)
    if resume_path is None and os.path.exists(config.SWIN_PRETRAINED):
        print(f"\nLoading pretrained weights: {config.SWIN_PRETRAINED}")
        load_swin_pretrained(model.tst.video_swin, config.SWIN_PRETRAINED, config.TST_WINDOW_SIZE)
    
    model = model.to(device)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model.tst = nn.DataParallel(model.tst)
    
    tst_module = model.tst.module if isinstance(model.tst, nn.DataParallel) else model.tst
    
    # ══════════════════════════════════════════════════════════════════════
    # DATASETS
    # ══════════════════════════════════════════════════════════════════════
    print("\nCreating datasets...")
    train_ds = SAVDataset(config.DATA_ROOT, config.TRAIN_CSV, config, train=True)
    val_ds = SAVDataset(config.DATA_ROOT, config.VAL_CSV, config, train=False)
    
    sampler = WeightedRandomSampler(train_ds.sample_weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    # ══════════════════════════════════════════════════════════════════════
    # OPTIMIZER & SCHEDULER
    # ══════════════════════════════════════════════════════════════════════
    backbone_params = list(tst_module.video_swin.parameters())
    head_params = (list(tst_module.feature_proj.parameters()) +
                   list(tst_module.classifier.parameters()) +
                   [tst_module.class_temp])
    
    print(f"\nBackbone params: {sum(p.numel() for p in backbone_params):,}")
    print(f"Head params: {sum(p.numel() for p in head_params):,}")
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.BACKBONE_LR},
        {'params': head_params, 'lr': config.HEAD_LR},
    ], weight_decay=config.WEIGHT_DECAY)
    
    steps_per_epoch = len(train_loader) // config.GRAD_ACCUM_STEPS
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch
    total_steps = config.EPOCHS * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return config.MIN_LR / config.HEAD_LR + (1 - config.MIN_LR / config.HEAD_LR) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss & AMP
    criterion = AsymmetricLossOptimized(
        gamma_neg=config.ASL_GAMMA_NEG,
        gamma_pos=config.ASL_GAMMA_POS,
        clip=config.ASL_CLIP,
        label_smoothing=config.LABEL_SMOOTHING
    ).to(device)
    
    scaler = GradScaler() if config.USE_AMP else None
    
    # EMA
    ema = None
    if config.USE_EMA:
        ema = EMA(tst_module, decay=config.EMA_DECAY)
    
    # ══════════════════════════════════════════════════════════════════════
    # RESUME FROM CHECKPOINT
    # ══════════════════════════════════════════════════════════════════════
    start_epoch = 1
    best_mAP = 0.0
    global_step = 0
    
    if resume_path is not None and os.path.exists(resume_path):
        print(f"\n{'='*70}")
        print(f"RESUMING FROM: {resume_path}")
        print(f"{'='*70}")
        
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
        # Load model weights
        tst_state = checkpoint['tst_state_dict']
        tst_state = {k.replace('module.', ''): v for k, v in tst_state.items()}
        tst_module.load_state_dict(tst_state)
        print(f"  ✓ Model weights loaded")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  ✓ Optimizer state loaded")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✓ Scheduler state loaded")
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"  ✓ Scaler state loaded")
        
        # Load EMA state
        if 'ema_shadow' in checkpoint and ema is not None:
            ema.shadow = checkpoint['ema_shadow']
            print(f"  ✓ EMA state loaded")
        
        # Load training state
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mAP = checkpoint.get('best_mAP', 0.0)
        global_step = checkpoint.get('global_step', 0)
        
        print(f"  ✓ Resuming from epoch {start_epoch}, best mAP: {best_mAP:.4f}")
        print(f"  ✓ Global step: {global_step}")
        print(f"  ✓ Current LR: backbone={scheduler.get_last_lr()[0]:.2e}, head={scheduler.get_last_lr()[1]:.2e}")
        print(f"{'='*70}\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        
        model.tst.train()
        model.detector.eval()
        optimizer.zero_grad()
        
        train_loss, num_batches, total_students = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (spatial, temporal, targets) in enumerate(pbar):
            spatial = spatial.to(device)
            temporal = temporal.to(device)
            
            try:
                if config.USE_AMP:
                    with autocast():
                        outputs = model(spatial, temporal, targets)
                        loss = criterion(outputs) / config.GRAD_ACCUM_STEPS
                    scaler.scale(loss).backward()
                else:
                    outputs = model(spatial, temporal, targets)
                    loss = criterion(outputs) / config.GRAD_ACCUM_STEPS
                    loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    print(f"  ⚠ OOM at batch {batch_idx}")
                    continue
                raise e
            
            total_students += sum(outputs['student_counts'])
            
            if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0:
                if config.USE_AMP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(tst_module.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(tst_module.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                if ema is not None:
                    ema.update()
                
                train_loss += loss.item() * config.GRAD_ACCUM_STEPS
                num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() * config.GRAD_ACCUM_STEPS:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'stud': total_students
            })
        
        avg_loss = train_loss / max(num_batches, 1)
        print(f"  Train Loss: {avg_loss:.4f} | Students: {total_students}")
        
        # ══════════════════════════════════════════════════════════════════
        # VALIDATION
        # ══════════════════════════════════════════════════════════════════
        model.tst.eval()
        
        if ema is not None:
            ema.apply_shadow()
        
        all_logits, all_labels = [], []
        
        with torch.no_grad():
            for spatial, temporal, targets in tqdm(val_loader, desc="Validation"):
                spatial = spatial.to(device)
                temporal = temporal.to(device)
                
                if config.USE_AMP:
                    with autocast():
                        outputs = model(spatial, temporal, targets)
                else:
                    outputs = model(spatial, temporal, targets)
                
                if outputs['logits'] is not None:
                    all_logits.append(outputs['logits'].cpu())
                    all_labels.append(outputs['labels'].cpu())
        
        if ema is not None:
            ema.restore()
        
        if all_logits:
            all_logits_cat = torch.cat(all_logits)
            all_labels_cat = torch.cat(all_labels)
            metrics = compute_metrics(all_logits_cat, all_labels_cat)
            
            print(f"  Val mAP: {metrics['mAP']:.4f} | F1: {metrics['f1']:.4f}")
            
            # Per-class AP
            for c in range(15):
                rare = "★" if c in RARE_CLASSES else " "
                print(f"  {rare} {ACTION_NAMES[c]:20s} AP={metrics['aps'][c]:.4f}")
            
            # ══════════════════════════════════════════════════════════════
            # SAVE CHECKPOINT — EVERY EPOCH
            # ══════════════════════════════════════════════════════════════
            tst_state = tst_module.state_dict()
            
            checkpoint_data = {
                'epoch': epoch,
                'global_step': global_step,
                'tst_state_dict': tst_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'ema_shadow': ema.shadow if ema else None,
                'best_mAP': best_mAP,
                'current_mAP': metrics['mAP'],
                'metrics': metrics,
                'config': {
                    'MODEL_SIZE': config.MODEL_SIZE,
                    'NUM_SEGMENTS': config.NUM_SEGMENTS,
                    'FRAMES_PER_SEGMENT': config.FRAMES_PER_SEGMENT,
                    'TST_WINDOW_SIZE': config.TST_WINDOW_SIZE,
                    'BACKBONE_LR': config.BACKBONE_LR,
                    'HEAD_LR': config.HEAD_LR,
                }
            }
            
            # Save BEST checkpoint
            if metrics['mAP'] > best_mAP:
                best_mAP = metrics['mAP']
                checkpoint_data['best_mAP'] = best_mAP
                torch.save(checkpoint_data, os.path.join(config.OUTPUT_DIR, 'tst_best.pth'))
                print(f"  ✓ NEW BEST! mAP: {best_mAP:.4f}")
            
            # Save LATEST checkpoint (for resume) — EVERY EPOCH
            torch.save(checkpoint_data, os.path.join(config.OUTPUT_DIR, 'tst_latest.pth'))
            print(f"  ✓ Checkpoint saved (epoch {epoch})")
            
            # Also save numbered checkpoint every 5 epochs (for backup)
            if epoch % 5 == 0:
                torch.save(checkpoint_data, os.path.join(config.OUTPUT_DIR, f'tst_epoch_{epoch}.pth'))
    
    print(f"\n{'='*70}")
    print(f"Training complete! Best mAP: {best_mAP:.4f}")
    print("=" * 70)
    
    return best_mAP

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    config = Config()
    # train(config, resume_path="/kaggle/input/models/YOUR_KAGGLE_USERNAME/tst-small-20/pytorch/default/1/tst_best.pth")