import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from functools import partial
from torchgeo.models import ViTSmall16_Weights
import numpy as np
from typing import Union, cast

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
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

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
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
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x) # (n_samples, n_patches + 1, out_features)
        return x

class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
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

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`. 

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
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

    Parameters
    ----------
    dim : int
        Embeddinig dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
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
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    Parameters
    ----------
    imag_size: int
        sizeof the image (it is a square)
    patch_size: int
        size of the patch (it is a square)
    in_chans: int
        number of input channels
    embed_dim: int
        the embedding dimension
    Attribute
    ---------
    n_patches: int
        number of patches inside of our image
    proj: nn.Conv2d
        convolutional layer that does both the splitting into patches and their embedding 

    """
    def __init__(self, img_size=15, patch_size=3, in_chans=10, embed_dim=384):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """ Run forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Shape '(n_samples, in_chans, img_size, img_size)'.
        Returns
        ------
        torch.Tensor
            Shape '(n_samples, n_pathces, embed_dim)'
        """
        B, C, H, W = x.shape
        x = self.proj(x) #(n_samples, embed_dim, n_patches** 0.5, n_patches**0.5)
        x = x.flatten(2) #(n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)
        return x
    


class SentinelViT(nn.Module):
    def __init__(self, img_size=15, patch_size=3, in_chans=10, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head ???
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate ? three of them usually =0.
            attn_drop_ratio (float): attention dropout rate ?
            drop_path_ratio (float): stochastic depth rate ?
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer ? norm_layer=nn.LayerNorm from dino, norm_layer = None from deep-learning for image processing
        """
        """Simplified implementation of the Vision transformer.

    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

      : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
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
        num_patches = (img_size // patch_size) ** 2 ##num_patches = self.patch_embed.num_patches

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
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embed_dim),#384
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.LayerNorm(256),
            nn.Linear(256, 7 * 25),  # 7 fractions * (5x5) spatial output
            nn.Sigmoid() #ensure output are between 0 and 1
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
            # Adapt kernel size (from 16x16 to 3x3)
            adapted_weight = F.interpolate(
                adapted_weight,
                size=(3, 3),
                mode='bicubic',
                align_corners=False
            )
            new_state_dict['patch_embed.proj.weight'] = adapted_weight

        # Handle position embedding
        if 'pos_embed' in state_dict:
            pretrained_pos_embed = state_dict['pos_embed']
            # Keep class token and reshape patch tokens
            class_pos_embed = pretrained_pos_embed[:, 0:1, :]
            patch_pos_embed = pretrained_pos_embed[:, 1:, :]

            # Calculate dimensions
            num_patches = int((15 // 3) ** 2)  # For 15x15 input with 3x3 patches
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
        temporal_outputs = []
        
        # Get year indices for temporal encoding
        years = year_to_tensor(start_year, B).to(x.device)

        for t in range(T):
            # Process current timestep
            tokens = self.prepare_tokens(x[:, :, t, :, :])

            # Get year embedding for current timestep
            year_embed = self.year_embedding(years[:, t])  # [B, year_embed_dim]
            
            # Expand year embedding to match sequence length
            year_embed = self.year_proj(year_embed)
            
            # Concatenate features with year embedding
            tokens[:, 0] = tokens[:,0] + year_embed

            # Pass through transformer blocks
            for blk in self.blocks:
                tokens = blk(tokens)
            
            # Get features and add temporal embedding
            # features = self.norm(tokens)[:, 0] # [B, embed_dim]
            #Add temporal information
            # year_features = self.year_encoding[years[:,t]]
            # features = features + self.temporal_embed[:, t, :] + year_features
            # features = self.temporal_norm(features)
            features = self.norm(tokens)[:,0]

            # Generate predictions
            output = self.regression_head(features) # [B, 7*5*5]
            output = output.view(B, 7, 5, 5) # reshape to [B,7,5,5]
            temporal_outputs.append(output)
        
        return torch.stack(temporal_outputs, dim=2)  # [B, 7, T, 5, 5]
    
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
        # Verify the output is within valid range (since we use sigmoid)
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

