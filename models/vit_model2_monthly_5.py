import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from functools import partial
from torchgeo.models import ViTSmall16_Weights
import numpy as np
from typing import Union, cast

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
    
def get_month_encoding_table(embed_dim):
        """Sinusoid month encoding table, for 42 months(2015-2018) indexed from 0-41
        Args:
        embed_dim (int): Embedding dimension, must be even number
        Returns:
        torch.FloatTensor: Encoding table of shape [4, embed_dim]
        """
        assert embed_dim % 2 == 0
        angles = np.arange(0, 43) / (42 / (2 * np.pi)) # 43 for 42 months plus boundary
        sin_table = np.sin(np.stack([angles for _ in range(embed_dim // 2)], axis=-1))
        cos_table = np.cos(np.stack([angles for _ in range(embed_dim // 2)], axis=-1))
        month_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
        return torch.FloatTensor(month_table)

def month_to_tensor(month, batch_size, seq_len = 42):
    """Convert month indices to tensor format
    Args:
        month: Starting month index (0-41)
        batch_size: Number of samples in batch
        seq_len: Length of temporal sequence (default 42 for monthly data)
    Returns:
        torch.Tensor: Month indices tensor of shape [batch_size, seq_len]
    """
    if isinstance(month, int):
        assert month < 42, f"Month index must be less than 4, got {month}"
        month = (
            torch.fmod(torch.arange(month, month + seq_len, dtype=torch.long), 42)
            .expand(batch_size, seq_len)
        )
    elif len(month.shape) == 1:
        month = torch.stack(
            [torch.fmod(torch.arange(y, y + seq_len, dtype=torch.long), 42) for y in month]
        )
    return month

def get_year_encoding_table(embed_dim):
       
        assert embed_dim % 2 == 0
        angles = np.arange(0, 5) / (4 / (2 * np.pi))
        sin_table = np.sin(np.stack([angles for _ in range(embed_dim // 2)], axis=-1))
        cos_table = np.cos(np.stack([angles for _ in range(embed_dim // 2)], axis=-1))
        year_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
        return torch.FloatTensor(year_table)

def year_to_tensor(year, batch_size, seq_len = 4):
    
    if isinstance(year, int):
        assert year < 4, f"Year index must be less than 4, got {year}"
        year = (
            torch.fmod(torch.arange(year, year + seq_len, dtype=torch.long), 4)
            .expand(batch_size, seq_len)
        )
    elif len(year.shape) == 1:
        year = torch.stack(
            [torch.fmod(torch.arange(y, y + seq_len, dtype=torch.long), 4) for y in year]
        )
    return year

class DropPath(nn.Module):
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x = self.fc1(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x) # (n_samples, n_patches + 1, out_features)
        return x

class Attention(nn.Module):
    
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    
    def __init__(self, img_size=5, patch_size=1, in_chans=10, embed_dim=384):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        
        B, C, H, W = x.shape
        x = self.proj(x) #(n_samples, embed_dim, n_patches** 0.5, n_patches**0.5)
        x = x.flatten(2) #(n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)
        return x
    


class SentinelViT(nn.Module):
    def __init__(self, img_size=5, patch_size=1, in_chans=10, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 temporal_mode = "monthly"):
        
        super().__init__()
        self.temporal_mode = temporal_mode
        self.num_features = embed_dim
        self.embed_dim = embed_dim

        #set temporal sequence length based on mode
        self.seq_len = 42 if temporal_mode == "monthly" else 4

        # # Create year embedding first
        # year_embed_dim = int(embed_dim * 0.2)  # 20% for year embedding
        # self.main_embed_dim = embed_dim - year_embed_dim

        # # Create year embedding table
        # year_tab = get_year_encoding_table(year_embed_dim)
        # self.year_embed = nn.Embedding.from_pretrained(year_tab, freeze=True)

        # Patch embedding with normalization
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.patch_norm = norm_layer(embed_dim)
        num_patches = (img_size // patch_size) ** 2

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        self.temp_embed_dim = int(embed_dim * 0.2)
        temp_table = get_month_encoding_table(self.temp_embed_dim) if temporal_mode == "monthly" \
                    else get_year_encoding_table(self.temp_embed_dim)
        self.temp_embedding = nn.Embedding.from_pretrained(temp_table, freeze= True)
        self.temp_proj = nn.Linear(self.temp_embed_dim, embed_dim)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Regression head for 7 fractions
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embed_dim),#384
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7 * 25),  # 7 fractions * (5x5) spatial output
            #Lambda(lambda x: x.view(-1, 7, 5, 5)),  # [B, 7, 5, 5]
            #Lambda(lambda x: torch.softmax(x, dim=1))  # Ensures fractions sum to 1 at each pixel
        )

        # Initialize weights
        self._init_weights()

    
    
    def _init_weights(self):
        
        # Initialize patch_embed
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=.02)
        
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
        # Initialize temporal embedding
        # nn.init.trunc_normal_(self.temporal_embed, std=.02)

        #Initialize regression head 
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _resize_pos_embed(self, pos_embed):
        """Resize position embeddings to match current model size"""
        # Handle class token and reshape
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        
        # Calculate dimensions
        H = W = int(math.sqrt(patch_pos_embed.shape[1]))
        target_H = target_W = int(math.sqrt(self.patch_embed.num_patches))
        
        # Interpolate
        patch_pos_embed = patch_pos_embed.reshape(1, H, W, -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(target_H, target_W),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
        
        # Recombine with class token
        pos_embed = torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)
        return pos_embed
    
    def load_pretrained_weights(self, state_dict):
        """Custom weight loading with adaptations for different sizes"""
        # Create a new state dict for the model
        new_state_dict = {}

        # Handle patch embedding weights
        if 'patch_embed.proj.weight' in state_dict:
            pretrained_weight = state_dict['patch_embed.proj.weight']
            # Adapt number of input channels (from 13 to 10), exclude B1, B9, B10. 13:bands B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12 \
            #adapted_weight = pretrained_weight[:, :10]
            indices_to_keep = [i for i in range(pretrained_weight.size(1)) if i not in [0,9,10]]
            #print(indices_to_keep)
            adapted_weight = pretrained_weight[:, indices_to_keep, :, :]
            # print(f"Pretrained weight shape: {pretrained_weight.shape}")
            # print(f"Adapted weight shape after band selection: {adapted_weight.shape}")
            # Adapt kernel size (from 16x16 to 3x3 to 1*1)
            # adapted_weight = F.interpolate(
            #     adapted_weight,
            #     size=(1, 1),
            #     mode='bicubic',
            #     align_corners=False
            # Get original shapes
            out_dim, in_dim, kh, kw = adapted_weight.shape
            
            # Adapt kernel size (from 3x3 to 1x1)
            adapted_weight = F.adaptive_avg_pool2d(
                adapted_weight.view(out_dim * in_dim, 1, kh, kw),  # reshape for pooling
                output_size=(1, 1)
            ).view(out_dim, in_dim, 1, 1)  # reshape back

            new_state_dict['patch_embed.proj.weight'] = adapted_weight

        # Handle position embedding
        if 'pos_embed' in state_dict:
            pretrained_pos_embed = state_dict['pos_embed']
            # Keep class token and reshape patch tokens
            class_pos_embed = pretrained_pos_embed[:, 0:1, :]
            patch_pos_embed = pretrained_pos_embed[:, 1:, :]

            # Calculate dimensions
            num_patches = int((5 // 1) ** 2)  # For 5x5 input with 1x1 patches
            patch_pos_embed = F.interpolate(
                patch_pos_embed.permute(0, 2, 1).view(1, -1, int(math.sqrt(patch_pos_embed.size(1))), int(math.sqrt(patch_pos_embed.size(1)))),
                size=int(math.sqrt(num_patches)),
                mode='bicubic',
                align_corners=False
            )
            patch_pos_embed = patch_pos_embed.view(1, -1, num_patches).permute(0, 2, 1)
            new_state_dict['pos_embed'] = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        
        # Copy remaining compatible weights
        for k, v in state_dict.items():
            if k not in ['patch_embed.proj.weight', 'pos_embed']:
                if k in self.state_dict() and self.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v

        # Load remaining weights
        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    def prepare_tokens(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = self.patch_norm(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x, start_time=0):
        B, C, T, H, W = x.shape
        temporal_outputs = []
        
        # Get time indices for temporal encoding
        times = month_to_tensor(start_time, B, self.seq_len) if self.temporal_mode == "monthly" \
               else year_to_tensor(start_time, B, self.seq_len)
        times = times.to(x.device)

        for t in range(T):
            # Process current timestep
            tokens = self.prepare_tokens(x[:, :, t, :, :])
            # print(f"Tokens shape: {tokens.shape}")

            # Get month embedding for current timestep
            temp_embed = self.temp_embedding(times[:, t])  # [B, temp_embed_dim]
            # print(f"Temp embed shape: {temp_embed.shape}")
            
            # Expand month embedding to match sequence length
            temp_embed = self.temp_proj(temp_embed)
            # print(f"Projected temp embed shape: {temp_embed.shape}")
            
            # Concatenate features with month embedding
            tokens[:, 0] = tokens[:,0] + temp_embed

            # Pass through transformer blocks
            for blk in self.blocks:
                tokens = blk(tokens)
            
            # Get features and add temporal embedding
            # features = self.norm(tokens)[:, 0] # [B, embed_dim]
            #Add temporal information
            # year_features = self.year_encoding[years[:,t]]
            # features = features + self.temporal_embed[:, t, :] + year_features
            # features = self.temporal_norm(features)
            features = self.norm(tokens)[:,0] #[B, embed_dim]
            # print(f"Features shape: {features.shape}")

            # Generate predictions
            output = self.regression_head(features) # [B, 7*5*5]
            # print(f"Output shape: {output.shape}")
            output = output.reshape(B, 7, 5, 5) # reshape to [B,7,5,5]
            output = F.softmax(output, dim=1)

            # Add verification
            if self.training:  # only check during training to save computation
                # Check if fractions sum to 1 for each pixel
                sums = output.sum(dim=1)  # Sum over fraction dimension
                if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-5):
                    print(f"Warning: Fractions don't sum to 1. Range: [{sums.min():.3f}, {sums.max():.3f}]")
                    
            temporal_outputs.append(output)
            
        return torch.stack(temporal_outputs, dim=2)  # [B, 7, T, 5, 5]
    
def create_model():
    """Create and initialize the model."""
    model = SentinelViT(temporal_mode="monthly")
    
    # Load pretrained weights
    weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO
    state_dict = weights.get_state_dict(progress=True)
    
    # Load weights that match
    msg = model.load_pretrained_weights(state_dict)
    print(f"Loaded pretrained weights with message: {msg}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print("Model created successfully")
    
    # Test forward pass
    x = torch.randn(32, 10, 42, 5, 5)  # Batch of 32, 10 bands, 42 timesteps, 5x5 spatial
    # Test with a few sample months (testing all 42 might be too verbose)
    test_months = [0, 10, 20, 30, 41]  # Test start months spread across the range
    for start_time in test_months:
        output = model(x, start_time=start_time)
        print(f"\nTesting with start_time = {start_time}")
        print(f"Output shape: {output.shape}")  # Should be [32, 7, 4, 5, 5]
        # Verify the output is within valid range (since we use sigmoid)
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Additional information
        if start_time == 0:
            print("\nDetailed shape information:")
            print(f"Input shape: {x.shape}")
            print(f"Output shape explanation:")
            print("- Batch size: 32")
            print("- Number of fractions: 7")
            print("- Time steps: 42 months")
            print("- Spatial dimensions: 5x5")

