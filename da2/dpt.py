# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.


"""
Standalone DPT (Dense Prediction Transformer) Depth Estimation model.

This module provides a complete, self-contained implementation of ViT-based
depth estimation. It combines:
- DINOv2 Vision Transformer encoder (ViT-S/B/L)
- DPT-style decoder head
- Full depth/normal estimation pipeline

Usage:
    >>> from da2.dpt import DepthAnything, DPTHead
    >>> model = DepthAnything(encoder="vitl", load_pretrain_path="/path/to/weights.pth")
    >>> depth, surface_normal = model(image)  # image: (B, 3, H, W)
"""

import logging
import math
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_


__all__ = [
    "build_dinov2",
    "DPTHead",
    "DepthAnything",
    "HyDenDepthAnything",
    "DPT_MODEL_CONFIGS",
    "HYDEN_DECODER_CONFIGS",
]

logger = logging.getLogger(__name__)


class DPTModelConfig(TypedDict):
    encoder: str
    features: int
    use_bn: bool
    out_channels: List[int]
    embed_dim: int
    depth: int
    num_heads: int
    intermediate_layers: List[int]


DPT_MODEL_CONFIGS: Dict[str, DPTModelConfig] = {
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "use_bn": False,
        "out_channels": [256, 512, 1024, 1024],
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "intermediate_layers": [4, 11, 17, 23],
    },
}

HYDEN_DECODER_CONFIGS: Dict[str, List[int]] = {
    "large": [256, 512, 1024, 1024],
}


class Mlp(nn.Module):
    """MLP block for transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (used in ViT-G)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        # Round to multiple of 256 for efficiency
        hidden_features = 256 * ((hidden_features + 255) // 256)

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.w3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        hidden = self.act(x1) * x2
        return self.w2(hidden)


class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention (uses PyTorch 2.0+ native SDPA if available)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_output = attn @ v

        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(nn.Module):
    """Layer scale module."""

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


_DEFAULT_NORM_LAYER: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = _DEFAULT_NORM_LAYER,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        init_values: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class DinoVisionTransformer(nn.Module):
    """
    DINOv2 Vision Transformer for OSS.

    Simplified version without:
    - Register tokens
    - Mask tokens
    - Block chunking
    - xFormers memory-efficient attention
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = None,
        ffn_layer: str = "mlp",
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Select FFN layer
        if ffn_layer == "mlp":
            ffn_class = Mlp
        elif ffn_layer in ["swiglu", "swiglufused"]:
            ffn_class = SwiGLUFFN
        else:
            ffn_class = Mlp

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    ffn_layer=ffn_class,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_weights)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate position encodings for different input sizes."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input tokens with position embeddings."""
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning feature dictionary."""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
        }

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], ...]:
        """Get intermediate layer outputs."""
        x = self.prepare_tokens(x)

        # Determine which blocks to output
        total_block_len = len(self.blocks)
        if isinstance(n, int):
            blocks_to_take = list(range(total_block_len - n, total_block_len))
        else:
            blocks_to_take = list(n)

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                outputs.append(x)

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        patch_tokens = [out[:, 1:] for out in outputs]

        if return_class_token:
            return tuple(zip(patch_tokens, class_tokens))
        return tuple(patch_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class token."""
        ret = self.forward_features(x)
        return ret["x_norm_clstoken"]


def build_dinov2(
    model_name: str = "vitl",
    pretrained_weights_path: Optional[str] = None,
) -> DinoVisionTransformer:
    """
    Build DINOv2 model.

    Args:
        model_name: Encoder key in ``DPT_MODEL_CONFIGS`` (currently only "vitl").
        pretrained_weights_path: Path to pretrained weights (.pth file)

    Returns:
        DinoVisionTransformer model
    """
    config = DPT_MODEL_CONFIGS[model_name]

    model = DinoVisionTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        init_values=1.0,
        ffn_layer="mlp",
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )

    if pretrained_weights_path:
        logger.info(f"Loading pretrained weights from: {pretrained_weights_path}")
        checkpoint = torch.load(
            pretrained_weights_path, map_location="cpu", weights_only=True
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = {
                    k.replace("pretrained.", ""): v
                    for k, v in checkpoint.items()
                    if "pretrained" in k
                }
                if not state_dict:
                    state_dict = checkpoint
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        if missing and unexpected and len(missing) > 10:
            raise RuntimeError(
                f"Checkpoint appears incompatible: {len(missing)} missing keys and "
                f"{len(unexpected)} unexpected keys. "
                f"Use an OSS-converted checkpoint."
            )

    return model


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        bn: bool = False,
    ) -> None:
        super().__init__()
        self.bn = bn

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)

        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        bn: bool = False,
        align_corners: bool = True,
        unit1: bool = True,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners

        self.out_conv = nn.Conv2d(features, features, kernel_size=1, bias=True)
        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn) if unit1 else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

    def forward(
        self, *xs: torch.Tensor, size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        output = xs[0]

        if len(xs) == 2 and self.resConfUnit1 is not None:
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        if size is None:
            output = F.interpolate(
                output.contiguous(),
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        else:
            output = F.interpolate(
                output.contiguous(),
                size=size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        output = self.out_conv(output)
        return output


class _Scratch(nn.Module):
    """Typed container for DPT scratch layers (replaces bare nn.Module())."""

    def __init__(
        self,
        layer1_rn: nn.Conv2d,
        layer2_rn: nn.Conv2d,
        layer3_rn: nn.Conv2d,
        layer4_rn: nn.Conv2d,
        refinenet1: FeatureFusionBlock,
        refinenet2: FeatureFusionBlock,
        refinenet3: FeatureFusionBlock,
        refinenet4: FeatureFusionBlock,
        output_conv1: nn.Conv2d,
        output_conv2: nn.Sequential,
    ) -> None:
        super().__init__()
        self.layer1_rn = layer1_rn
        self.layer2_rn = layer2_rn
        self.layer3_rn = layer3_rn
        self.layer4_rn = layer4_rn
        self.refinenet1 = refinenet1
        self.refinenet2 = refinenet2
        self.refinenet3 = refinenet3
        self.refinenet4 = refinenet4
        self.output_conv1 = output_conv1
        self.output_conv2 = output_conv2


class DPTHead(nn.Module):
    """
    DPT decoder head for depth/normal estimation.

    Converts ViT features to dense predictions via a multi-scale fusion decoder.
    """

    def __init__(
        self,
        in_channels: int,
        features: int = 256,
        use_bn: bool = False,
        out_channels: Optional[List[int]] = None,
        surface_normal: bool = False,
        use_clstoken: bool = False,
        use_leaky_relu: bool = False,
        patch_size: int = 14,
        cnn_channels: Optional[List[int]] = None,
        hyden_decoder_channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.use_clstoken = use_clstoken
        self.patch_size = patch_size
        self.add_cnn_encoder = cnn_channels is not None
        relu_fn = nn.LeakyReLU if use_leaky_relu else nn.ReLU

        if out_channels is None:
            raise ValueError("out_channels must be provided")

        # Project ViT features to different channel dimensions
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels]
        )

        # Resize layers to create multi-scale features
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        # Optional class token readout
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                    for _ in range(len(self.projects))
                ]
            )

        # CNN feature fusion layers for HyDen mode
        if cnn_channels is not None:
            assert hyden_decoder_channels is not None
            self.combine_feats = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            out_channels[i] + cnn_channels[i],
                            out_channels[i] + cnn_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            out_channels[i] + cnn_channels[i],
                            hyden_decoder_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    )
                    for i in range(len(out_channels))
                ]
            )

        # Scratch layers for feature refinement
        head_features_1 = features
        head_features_2 = 32
        out_dim = 4 if surface_normal else 1

        scratch_in_channels = (
            hyden_decoder_channels
            if hyden_decoder_channels is not None
            else out_channels
        )

        self.scratch = _Scratch(
            layer1_rn=nn.Conv2d(
                scratch_in_channels[0], features, kernel_size=3, padding=1, bias=False
            ),
            layer2_rn=nn.Conv2d(
                scratch_in_channels[1], features, kernel_size=3, padding=1, bias=False
            ),
            layer3_rn=nn.Conv2d(
                scratch_in_channels[2], features, kernel_size=3, padding=1, bias=False
            ),
            layer4_rn=nn.Conv2d(
                scratch_in_channels[3], features, kernel_size=3, padding=1, bias=False
            ),
            refinenet1=FeatureFusionBlock(features, relu_fn(False), use_bn),
            refinenet2=FeatureFusionBlock(features, relu_fn(False), use_bn),
            refinenet3=FeatureFusionBlock(features, relu_fn(False), use_bn),
            refinenet4=FeatureFusionBlock(
                features, relu_fn(False), use_bn, unit1=False
            ),
            output_conv1=nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, padding=1
            ),
            # NOTE: keep the final activation fixed to ReLU to match production
            # DepthAnything decoder behavior (always non-negative inverse depth).
            # use_leaky_relu only affects the intermediate convs, not the output.
            output_conv2=nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    padding=1,
                ),
                relu_fn(True),
                nn.Conv2d(head_features_2, out_dim, kernel_size=1),
                nn.ReLU(True) if not surface_normal else nn.Identity(),
                nn.Identity(),
            ),
        )

    def forward(
        self,
        out_features: List[Tuple[torch.Tensor, torch.Tensor]],
        patch_h: int,
        patch_w: int,
        output_size: Optional[Tuple[int, int]] = None,
        cnn_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            out_features: List of (patch_tokens, class_token) tuples from ViT
            patch_h: Number of patches in height
            patch_w: Number of patches in width
            output_size: Optional output size for interpolation
            cnn_features: Optional CNN encoder features for HyDen fusion

        Returns:
            Dense prediction tensor
        """
        out = []

        for i, x in enumerate(out_features):
            if self.use_clstoken:
                patch_tokens, cls_token = x
                readout = cls_token.unsqueeze(1).expand_as(patch_tokens)
                x = self.readout_projects[i](torch.cat((patch_tokens, readout), -1))
            else:
                x = x[0] if isinstance(x, tuple) else x

            # Reshape from (B, N, C) to (B, C, H, W)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        if self.add_cnn_encoder and cnn_features is not None:
            new_out = []
            for layer_idx in range(len(out)):
                layer = out[layer_idx]
                layer_cnn = cnn_features[layer_idx]
                layer = F.interpolate(
                    layer,
                    (layer_cnn.shape[2], layer_cnn.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                combined_feat = torch.cat([layer, layer_cnn], dim=1)
                layer = self.combine_feats[layer_idx](combined_feat)
                new_out.append(layer)
            layer_1, layer_2, layer_3, layer_4 = new_out

        # Feature refinement
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Output
        out_tensor = self.scratch.output_conv1(path_1)

        if output_size is None:
            output_size = (
                int(patch_h * self.patch_size),
                int(patch_w * self.patch_size),
            )

        out_tensor = F.interpolate(
            out_tensor,
            output_size,
            mode="bilinear",
            align_corners=True,
        )
        out_tensor = self.scratch.output_conv2(out_tensor)

        return out_tensor


class DepthAnything(nn.Module):
    """
    Full Depth Anything model for training and inference.

    OSS version of DepthAnything from dpt.py with:
    - DINOv2 encoder
    - DPT decoder
    - Depth and optional surface normal prediction
    """

    def __init__(
        self,
        encoder: str = "vitl",
        load_pretrain_path: Optional[str] = None,
        predict_depth: bool = True,
        predict_surface_normal: bool = False,
        use_leaky_relu: bool = False,
        vit_internal_resolution: Optional[Tuple[int, int]] = None,
        freeze_encoder: bool = False,
        add_cnn_encoder: str = "",
    ) -> None:
        """
        Args:
            encoder: Encoder key in ``DPT_MODEL_CONFIGS`` (currently only "vitl").
            load_pretrain_path: Path to pretrained weights
            predict_depth: Whether to predict depth
            predict_surface_normal: Whether to predict surface normals
            use_leaky_relu: Use LeakyReLU instead of ReLU
            vit_internal_resolution: Internal resolution for ViT
            freeze_encoder: Whether to freeze encoder weights
            add_cnn_encoder: CNN encoder architecture for HyDen mode (e.g., "large")
        """
        super().__init__()

        config = DPT_MODEL_CONFIGS[encoder]
        self.encoder_name = encoder
        self.patch_size = 14
        self.predict_depth = predict_depth
        self.predict_surface_normal = predict_surface_normal
        self.vit_internal_resolution = vit_internal_resolution
        self.intermediate_layers = config["intermediate_layers"]

        # Build encoder
        self.pretrained = build_dinov2(
            model_name=encoder,
            pretrained_weights_path=load_pretrain_path,
        )

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.pretrained.parameters():
                param.requires_grad = False

        # Build CNN encoder for HyDen mode
        self.add_cnn_encoder = add_cnn_encoder
        if add_cnn_encoder:
            from ..cnn import build_cnn_encoder

            self.cnn_encoder = build_cnn_encoder(arch=add_cnn_encoder)
            # Last 4 stages: [32, 64, 160, 288]
            cnn_channels = [32, 64, 160, 288]
            hyden_decoder_channels = HYDEN_DECODER_CONFIGS.get(
                add_cnn_encoder, config["out_channels"]
            )
        else:
            cnn_channels = None
            hyden_decoder_channels = None

        # Build depth head
        if predict_depth:
            self.depth_head = DPTHead(
                in_channels=config["embed_dim"],
                features=config["features"],
                use_bn=config["use_bn"],
                out_channels=config["out_channels"],
                surface_normal=False,
                use_leaky_relu=use_leaky_relu,
                patch_size=self.patch_size,
                cnn_channels=cnn_channels,
                hyden_decoder_channels=hyden_decoder_channels,
            )

        # Build surface normal head
        if predict_surface_normal:
            self.surface_normal_head = DPTHead(
                in_channels=config["embed_dim"],
                features=config["features"],
                use_bn=config["use_bn"],
                out_channels=config["out_channels"],
                surface_normal=True,
                use_leaky_relu=use_leaky_relu,
                patch_size=self.patch_size,
                cnn_channels=cnn_channels,
                hyden_decoder_channels=hyden_decoder_channels,
            )

        # Load decoder weights from checkpoint if available
        if load_pretrain_path:
            self._load_pretrained_heads(
                load_pretrain_path, predict_depth, predict_surface_normal
            )

        self._log_model_info(encoder)

    @staticmethod
    def _get_size(
        width: int,
        height: int,
        target_width: int,
        target_height: int,
        patch_size: int,
    ) -> Tuple[int, int]:
        """Minimal aspect-preserving resize helper (lower bound, patch-aligned)."""
        scale_h = target_height / height
        scale_w = target_width / width
        scale = max(scale_h, scale_w)
        new_h = int((scale * height) // patch_size * patch_size)
        new_w = int((scale * width) // patch_size * patch_size)
        return new_h, new_w

    @staticmethod
    def _load_head_state(
        state_dict: Dict[str, torch.Tensor],
        head: nn.Module,
        prefix: str,
    ) -> None:
        """Load state dict into a head module, filtering by prefix."""
        head_state = {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if head_state:
            missing, unexpected = head.load_state_dict(head_state, strict=False)
            if missing or unexpected:
                logger.warning(
                    f"Head load ({prefix}) had missing={len(missing)} unexpected={len(unexpected)}"
                )
            if missing and len(missing) == len(list(head.state_dict().keys())):
                raise RuntimeError(
                    f"Head '{prefix}' loaded 0 keys from checkpoint. "
                    f"Checkpoint may be incompatible."
                )

    def _load_pretrained_heads(
        self,
        load_pretrain_path: str,
        predict_depth: bool,
        predict_surface_normal: bool,
    ) -> None:
        """Load decoder head weights from a full checkpoint."""
        checkpoint = torch.load(
            load_pretrain_path, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict):
            return
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if predict_depth:
            self._load_head_state(state_dict, self.depth_head, "depth_head.")
        if predict_surface_normal:
            self._load_head_state(
                state_dict, self.surface_normal_head, "surface_normal_head."
            )

    def _log_model_info(self, encoder: str) -> None:
        """Log model parameter counts."""
        encoder_params = sum(p.numel() for p in self.pretrained.parameters()) / 1e6
        logger.info(f"[DepthAnything] Encoder: {encoder}, {encoder_params:.2f}M params")
        if self.predict_depth:
            depth_params = sum(p.numel() for p in self.depth_head.parameters()) / 1e6
            logger.info(f"[DepthAnything] Depth head: {depth_params:.2f}M params")
        if self.predict_surface_normal:
            normal_params = (
                sum(p.numel() for p in self.surface_normal_head.parameters()) / 1e6
            )
            logger.info(f"[DepthAnything] Normal head: {normal_params:.2f}M params")

    def get_params_groups(self) -> List:
        """Get parameter groups for optimizer."""
        decoder_params = []
        if self.predict_depth:
            decoder_params += list(self.depth_head.parameters())
        if self.predict_surface_normal:
            decoder_params += list(self.surface_normal_head.parameters())
        return [list(self.pretrained.parameters()), decoder_params]

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        training: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Tuple],
    ]:
        """
        Forward pass.

        Args:
            x: Input RGB image tensor of shape ``(B, 3, H, W)`` with values
                in ``[0, 1]``.
            return_features: Whether to return intermediate features.
            training: Whether in training mode.

        Returns:
            - Inference with ``predict_depth`` only: ``depth`` tensor ``(B, H, W)``.
            - Inference with ``predict_surface_normal`` only: ``surface_normal``
              tensor ``(B, 3, H, W)``.
            - Inference with both: tuple ``(depth, surface_normal)``.
            - Training: tuple ``(depth, surface_normal)`` or
              ``(depth, surface_normal, features)`` if ``return_features=True``.
        """
        output_size = x.shape[2:]

        # Resize if needed (keep aspect ratio + patch divisibility, match internal)
        if not training and self.vit_internal_resolution is not None:
            target_w, target_h = self.vit_internal_resolution
            new_h, new_w = self._get_size(
                width=x.shape[3],
                height=x.shape[2],
                target_width=target_w,
                target_height=target_h,
                patch_size=self.patch_size,
            )
            x = F.interpolate(
                x,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=True,
            )

        # Check patch size compatibility
        assert (
            x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0
        ), (
            f"Input size ({x.shape[-2]}, {x.shape[-1]}) must be divisible by patch_size ({self.patch_size})"
        )

        patch_h, patch_w = (
            x.shape[-2] // self.patch_size,
            x.shape[-1] // self.patch_size,
        )

        # Get features
        features = self.pretrained.get_intermediate_layers(
            x,
            n=self.intermediate_layers,
            return_class_token=True,
        )

        # Run CNN encoder for HyDen mode
        cnn_features = None
        if self.add_cnn_encoder:
            cnn_out = self.cnn_encoder(x)
            # Extract last 4 stage features (stage1-stage4)
            cnn_features = [cnn_out[f"stage{i}"] for i in range(1, 5)]

        # Predict depth
        if self.predict_depth:
            depth = self.depth_head(
                features,
                patch_h,
                patch_w,
                output_size=output_size,
                cnn_features=cnn_features,
            )
            depth = depth.squeeze(1)
        else:
            depth = torch.ones(
                x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )

        # Predict surface normal
        if self.predict_surface_normal:
            surface_normal = self.surface_normal_head(
                features, patch_h, patch_w, cnn_features=cnn_features
            )
        else:
            surface_normal = torch.ones(
                x.shape[0], 3, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )

        if training:
            if return_features:
                return depth, surface_normal, features
            return depth, surface_normal

        # Inference output
        out = []
        if self.predict_depth:
            out.append(depth)
        if self.predict_surface_normal:
            out.append(surface_normal)

        if len(out) == 1:
            return out[0]
        return tuple(out)


class HyDenDepthAnything(DepthAnything):
    """HyDen (Hybrid DINOv2 + CNN Encoder) model for relative depth estimation.

    This is a convenience wrapper around DepthAnything with the CNN encoder
    pre-configured for HyDen mode.
    """

    def __init__(
        self,
        encoder: str = "vitl",
        cnn_encoder: str = "large",
        **kwargs,
    ) -> None:
        super().__init__(encoder=encoder, add_cnn_encoder=cnn_encoder, **kwargs)
