"""Independent implementation of MetaFi++ channel attention.

Architecture reference: DT-Pose's MetaFi++ baseline, commit
2fbbef271454e6fbb95b1a0163b8cdbb96163838, model/metafi/ChannelTrans.py.
This is channel-channel attention, NOT ordinary token self-attention: each of
three heads has a full 512-wide Q/K/V projection, and heads are averaged.
See metafi_sources.json for provenance and the source-license audit.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class ChannelEmbeddings(nn.Module):
    def __init__(self, joints: int, width: int, channels: int):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, joints * width, channels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x.flatten(2).transpose(1, 2) + self.position_embeddings)


class ChannelAttention(nn.Module):
    """Three full-width heads; normalize each channel-affinity map spatially."""

    def __init__(self, channels: int = 512, heads: int = 3):
        super().__init__()
        self.channels = channels
        self.query1 = nn.ModuleList(nn.Linear(channels, channels, bias=False) for _ in range(heads))
        self.key = nn.ModuleList(nn.Linear(channels, channels, bias=False) for _ in range(heads))
        self.value = nn.ModuleList(nn.Linear(channels, channels, bias=False) for _ in range(heads))
        self.psi = nn.InstanceNorm2d(heads)
        self.softmax = nn.Softmax(dim=-1)
        self.out1 = nn.Linear(channels, channels, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = torch.stack([layer(tokens) for layer in self.query1], dim=1)
        k = torch.stack([layer(tokens) for layer in self.key], dim=1)
        v = torch.stack([layer(tokens) for layer in self.value], dim=1)
        # B,H,C,C, not B,H,N,N. The denominator is sqrt(C), as in source.
        affinities = torch.matmul(q.transpose(-1, -2), k) / math.sqrt(self.channels)
        probabilities = self.attn_dropout(self.softmax(self.psi(affinities)))
        context = torch.matmul(probabilities, v.transpose(-1, -2))
        context = context.permute(0, 3, 2, 1).contiguous().mean(dim=-1)
        return self.proj_dropout(self.out1(context))


class ChannelMLP(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, 4 * channels)
        self.fc2 = nn.Linear(4 * channels, channels)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))


class ChannelBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.attn_norm1 = nn.LayerNorm(channels, eps=1e-6)
        # Retained for state-dict compatibility. The published forward does
        # not use this second attention normalization.
        self.attn_norm = nn.LayerNorm(channels, eps=1e-6)
        self.channel_attn = ChannelAttention(channels, heads)
        self.ffn_norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.ffn1 = ChannelMLP(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.channel_attn(self.attn_norm1(x))
        return x + self.ffn1(self.ffn_norm1(x))


class ChannelEncoder(nn.Module):
    def __init__(self, channels: int, heads: int, layers: int):
        super().__init__()
        self.layer = nn.ModuleList(ChannelBlock(channels, heads) for _ in range(layers))
        self.encoder_norm1 = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layer:
            x = block(x)
        return self.encoder_norm1(x)


class ChannelReconstruction(nn.Module):
    def __init__(self, channels: int, joints: int, width: int):
        super().__init__()
        self.joints, self.width = joints, width
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.transpose(1, 2).contiguous()
        x = x.reshape(len(x), x.shape[1], self.joints, self.width)
        return self.activation(self.norm(self.conv(x)))


class MetaFiChannelTransformer(nn.Module):
    def __init__(self, joints: int = 17, channels: int = 512):
        super().__init__()
        self.embeddings_1 = ChannelEmbeddings(joints, 12, channels)
        self.encoder = ChannelEncoder(channels, heads=3, layers=1)
        self.reconstruct_1 = ChannelReconstruction(channels, joints, width=12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.embeddings_1(x))
        return x + self.reconstruct_1(encoded)
