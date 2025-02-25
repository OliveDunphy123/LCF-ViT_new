import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from functools import partial
import torch.utils
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

def get_year_encoding_table(embed_dim):
        """Sinusoid year encoding table, for 4 years(2015-2018) indexed from 0-3
        Args:
        embed_dim (int): Embedding dimension, must be even number
        Returns:
        torch.FloatTensor: Encoding table of shape [4, embed_dim]
        """
        assert embed_dim % 2 == 0
        angles = np.arange(0, 5) / (4 / (2 * np.pi))
        sin_table = np.sin(np.stack([angles for _ in range(embed_dim // 2)], axis=-1))
        cos_table = np.cos(np.stack([angles for _ in range(embed_dim // 2)], axis=-1))
        year_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
        return torch.FloatTensor(year_table)

def year_to_tensor(year, batch_size, seq_len = 4):
    """Convert year indices to tensor format
    Args:
        year: Starting year index (0-3)
        batch_size: Number of samples in batch
        seq_len: Length of temporal sequence (default 4 for yearly data)
    Returns:
        torch.Tensor: Year indices tensor of shape [batch_size, seq_len]
    """
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
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    Multilayer perceptron.
    """
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
    """Attention mechanism.
    """
    def __init__(self, 
                 dim, # input token dim
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
        """Run forward pass.
        """
        # [batch_size, num_patches + 1, total_embed_dim], dim = total_embed_dim
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
    """Transformer block.
    """
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
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=15, patch_size=3, in_chans=10, embed_dim=384):
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
    def __init__(self, img_size=15, patch_size=3, in_chans=10, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
    
        super().__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim

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


        self.year_embed_dim = int(embed_dim * 0.2)
        year_table = get_year_encoding_table(self.year_embed_dim)
        self.year_embedding = nn.Embedding.from_pretrained(year_table, freeze= True)
        self.year_proj = nn.Linear(self.year_embed_dim, embed_dim)

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
        # self.regression_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),#384
        #     nn.Linear(embed_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.Dropout(0.15),
        #     nn.Linear(512, 256),
        #     nn.GELU(),
        #     nn.Dropout(0.15),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 7 * 25),  # 7 fractions * (5x5) spatial output
        #     nn.Softmax() #ensure output are between 0 and 1
        # )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embed_dim),#384
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7 * 25),  # 7 fractions * (5x5) spatial output
            Lambda(lambda x: x.reshape(-1, 7, 5, 5)),  # [B, 7, 5, 5]
            Lambda(lambda x: torch.softmax(x, dim=1))  # Ensures fractions sum to 1 at each pixel
            #nn.Softmax(dim=1)
        )

        # Initialize weights
        self._init_weights()

    
    def _init_weights(self):
        """
        ViT weight initialization
        :param m: module
        """
        # Initialize patch_embed
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=.02)
        
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
        # Initialize temporal embedding
        # nn.init.trunc_normal_(self.temporal_embed, std=.02)

        #Initialize regression head 
        # for m in self.regression_head.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.trunc_normal_(m.weight, std=.02)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                if m is self.regression_head[-3]:  # Last linear layer
                    # Initialize last layer with smaller weights
                    nn.init.trunc_normal_(m.weight, std=.01)
                else:
                    nn.init.trunc_normal_(m.weight, std=.02)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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

        # Validate temporal embedding dimensions if present
        if hasattr(self, 'temp_proj'):
            assert self.temp_proj.in_features == self.temp_embed_dim, \
                f"Temporal embedding dimensions mismatch: {self.temp_proj.in_features} != {self.temp_embed_dim}"
        
        try:
            # Handle patch embedding weights
            if 'patch_embed.proj.weight' in state_dict:
                pretrained_weight = state_dict['patch_embed.proj.weight']
                print(f"Original patch embedding shape: {pretrained_weight.shape}")
                
                # Adapt number of input channels (exclude B1, B9, B10)
                indices_to_keep = [i for i in range(pretrained_weight.size(1)) if i not in [0,9,10]]
                adapted_weight = pretrained_weight[:, indices_to_keep, :, :]
                print(f"Shape after band selection: {adapted_weight.shape}")
                
                try:
                    # Adapt kernel size (from 16x16 to 3x3)
                    adapted_weight = F.interpolate(
                        adapted_weight,
                        size=(3, 3),
                        mode='bicubic',
                        align_corners=False
                    )
                    print(f"Final patch embedding shape: {adapted_weight.shape}")
                    new_state_dict['patch_embed.proj.weight'] = adapted_weight
                except Exception as e:
                    raise ValueError(f"Failed to resize patch embedding: {e}")

            # Handle position embedding
            if 'pos_embed' in state_dict:
                pretrained_pos_embed = state_dict['pos_embed']
                print(f"Original position embedding shape: {pretrained_pos_embed.shape}")
                
                try:
                    # Keep class token and reshape patch tokens
                    class_pos_embed = pretrained_pos_embed[:, 0:1, :]
                    patch_pos_embed = pretrained_pos_embed[:, 1:, :]

                    # Calculate dimensions
                    num_patches = int((15 // 3) ** 2)  # For 15x15 input with 3x3 patches
                    patch_pos_embed = F.interpolate(
                        patch_pos_embed.permute(0, 2, 1).view(
                            1, -1, 
                            int(math.sqrt(patch_pos_embed.size(1))), 
                            int(math.sqrt(patch_pos_embed.size(1)))
                        ),
                        size=int(math.sqrt(num_patches)),
                        mode='bicubic',
                        align_corners=False
                    )
                    patch_pos_embed = patch_pos_embed.view(1, -1, num_patches).permute(0, 2, 1)
                    new_state_dict['pos_embed'] = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
                    print(f"Final position embedding shape: {new_state_dict['pos_embed'].shape}")
                except Exception as e:
                    raise ValueError(f"Failed to resize position embedding: {e}")
            
            # Copy remaining compatible weights
            skipped_keys = []
            copied_keys = []
            for k, v in state_dict.items():
                if k not in ['patch_embed.proj.weight', 'pos_embed']:
                    if k in self.state_dict():
                        if self.state_dict()[k].shape == v.shape:
                            new_state_dict[k] = v
                            copied_keys.append(k)
                        else:
                            skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {self.state_dict()[k].shape})")
                    else:
                        skipped_keys.append(f"{k} (not in model)")

            # Load weights
            msg = self.load_state_dict(new_state_dict, strict=False)
            
            # Print summary
            print("\nWeight Loading Summary:")
            print(f"Successfully loaded: {len(copied_keys)} weights")
            print(f"Adapted weights: patch_embed.proj.weight, pos_embed")
            if skipped_keys:
                print(f"Skipped weights: {len(skipped_keys)}")
                for key in skipped_keys[:5]:  # Show first 5 skipped keys
                    print(f"  - {key}")
                if len(skipped_keys) > 5:
                    print(f"  ... and {len(skipped_keys)-5} more")
            
            if msg.missing_keys:
                print("\nMissing keys:")
                for key in msg.missing_keys[:5]:  # Show first 5 missing keys
                    print(f"  - {key}")
                if len(msg.missing_keys) > 5:
                    print(f"  ... and {len(msg.missing_keys)-5} more")
            
            return msg
            
        except Exception as e:
            print(f"Error during weight loading: {e}")
            raise

    def prepare_tokens(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x) #patch linear embedding
        x = self.patch_norm(x)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x, start_year=0):
        B, C, T, H, W = x.shape
        #print(f"\nInput shape: {x.shape}")  # [B, 10, 4, 15, 15]
    
        temporal_outputs = []
        years = year_to_tensor(start_year, B).to(x.device)
    
        for t in range(T):
            # Current timestep input
            current_input = x[:, :, t, :, :]
            #print(f"\nTimestep {t} input shape: {current_input.shape}")  # [B, 10, 15, 15]
            
            # Patch embedding
            patches = self.patch_embed(current_input)
            #print(f"After patch embedding: {patches.shape}")  # [B, 25, 384]
            
            patches = self.patch_norm(patches)
            
            # Add class token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, patches), dim=1)
            #print(f"After adding class token: {tokens.shape}")  # [B, 26, 384]
            
            # Add position embedding
            tokens = tokens + self.pos_embed
            tokens = self.pos_drop(tokens)
            #print(f"After position embedding: {tokens.shape}")  # [B, 26, 384]
            
            # Get year embedding
            year_embed = self.year_embedding(years[:, t])
            #print(f"Year embedding shape: {year_embed.shape}")  # [B, year_embed_dim]
            
            year_embed = self.year_proj(year_embed)
            #print(f"Projected year embedding: {year_embed.shape}")  # [B, 384]
            
            # Add year embedding to class token
            tokens[:, 0] = tokens[:, 0] + year_embed
            
            # Process through transformer blocks
            # if self.training:
            #     for blk in self.blocks:
            #         tokens = torch.utils.checkpoint.checkpoint(blk, tokens)
            # else:
            #     for blk in self.blocks:
            #         tokens = blk(tokens)
            #         #if i == 0:  # Print shape after first block
            #             #print(f"After first transformer block: {tokens.shape}")  # [B, 26, 384]
            for blk in self.blocks:
                tokens = blk(tokens)
            # Extract features
            features = self.norm(tokens)[:, 0]
            #print(f"After extracting CLS token: {features.shape}")  # [B, 384]
            
            # Generate predictions
            output = self.regression_head(features)
            #print(f"After regression head: {output.shape}")  # [B, 7*25]
            
            output = output.view(B, 7, 5, 5)
            #print(f"Reshaped output: {output.shape}")  # [B, 7, 5, 5]
            output = F.softmax(output, dim=1)  # To ensure fractions sum to 1
            
            # Add this after generating the output in the forward method
            if self.training:
                # Check if fractions sum to 1 for each pixel
                sums = output.sum(dim=1)  # Sum across fraction dimension
                if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-5):
                    print(f"Warning: Fractions don't sum to 1. Range: [{sums.min():.3f}, {sums.max():.3f}]")
            temporal_outputs.append(output)
    
        final_output = torch.stack(temporal_outputs, dim=2)
        #print(f"\nFinal output shape: {final_output.shape}")  # [B, 7, 4, 5, 5]

        return final_output
    
def create_model():
    """Create and initialize the model."""
    model = SentinelViT()
    
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
    x = torch.randn(32, 10, 4, 15, 15)  # Batch of 32, 10 bands, 4 timesteps, 15x15 spatial
    # Test with all possible starting years
    for start_year in range(4):
        output = model(x, start_year=start_year)
        print(f"\nTesting with start_year = {start_year}")
        print(f"Output shape: {output.shape}")  # Should be [32, 7, 4, 5, 5]
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

