"""Independently implemented MM-Fi GraphPose-Fi architecture.

The operator specification is the author's GraphPose-Fi commit recorded in
``graphpose_sources.json``.  This is not vendored author code or a claim of
full-dataset reproduction.  In particular, the sample-standard-deviation
normalization, row-normalized skeleton, learned graph feed-forward layers,
Chebyshev order, and sequential shared-antenna encoder are intentional.

The only interface changes are accepting the repository's four-dimensional
CSI tensor, adding the singleton person dimension to the pose, and registering
fixed graph/mask tensors as buffers so CPU/CUDA/checkpoint movement is safe.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


MMFI_GRAPH_EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)


def mmfi_adjacency() -> torch.Tensor:
    """Author graph: undirected bones + self loops, then row normalization."""
    graph = torch.eye(17, dtype=torch.float64)
    for start, end in MMFI_GRAPH_EDGES:
        graph[start, end] = graph[end, start] = 1
    return (graph / graph.sum(dim=1, keepdim=True)).float()


class ChebConv(nn.Module):
    """Order-K polynomial filtering, with the source's K+1 coefficients."""

    def __init__(self, in_c: int, out_c: int, K: int = 2):
        super().__init__()
        self.K = K + 1
        self.weight = nn.Parameter(torch.empty(self.K, 1, in_c, out_c))
        self.bias = nn.Parameter(torch.zeros(1, 1, out_c))
        nn.init.xavier_normal_(self.weight)

    @staticmethod
    def get_laplacian(graph: torch.Tensor) -> torch.Tensor:
        inverse_sqrt_degree = torch.diag(graph.sum(dim=-1).pow(-0.5))
        return torch.eye(graph.shape[0], device=graph.device, dtype=graph.dtype) - (
            inverse_sqrt_degree @ graph @ inverse_sqrt_degree
        )

    def cheb_polynomial(self, laplacian: torch.Tensor) -> torch.Tensor:
        terms = [torch.eye(laplacian.shape[0], device=laplacian.device,
                           dtype=laplacian.dtype)]
        if self.K > 1:
            terms.append(laplacian)
        for _ in range(2, self.K):
            terms.append(2 * laplacian @ terms[-1] - terms[-2])
        return torch.stack(terms)

    def forward(self, inputs: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        basis = self.cheb_polynomial(self.get_laplacian(graph))
        filtered = basis[:, None] @ inputs
        return (filtered @ self.weight).sum(dim=0) + self.bias


class _GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, p_dropout: float = 0.1):
        super().__init__()
        self.gconv = ChebConv(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Retain the source's post-dropout ReLU as well as the first ReLU.
        return self.relu(self.dropout(self.relu(self.gconv(x, adj))))


class _ResChebGC(nn.Module):
    def __init__(self, adj: torch.Tensor, dim: int):
        super().__init__()
        self.register_buffer('adj', adj.clone())
        self.gconv1 = _GraphConv(dim, dim)
        self.gconv2 = _GraphConv(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gconv2(self.gconv1(x, self.adj), self.adj)


class LayerNorm(nn.Module):
    """GraFormer's sample-std normalization, NOT torch.nn.LayerNorm."""

    def __init__(self, features: int):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a_2 * (x - x.mean(dim=-1, keepdim=True)) / (
            x.std(dim=-1, keepdim=True, unbiased=True) + self.eps
        ) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        if d_model % h:
            raise ValueError('attention width must be divisible by head count')
        self.h, self.d_k = h, d_model // h
        template = nn.Linear(d_model, d_model)
        # Author clones one initialization (not four independent draws).
        self.linears = nn.ModuleList(copy.deepcopy(template) for _ in range(4))
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]
        projected = [layer(x).reshape(batch, -1, self.h, self.d_k).transpose(1, 2)
                     for layer, x in zip(self.linears, (query, key, value))]
        q, k, v = projected
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1).bool(), -1e9)
        self.attn = self.dropout(logits.softmax(dim=-1))
        values = (self.attn @ v).transpose(1, 2).contiguous()
        return self.linears[-1](values.reshape(batch, -1, self.h * self.d_k))


class LAM_Gconv(nn.Module):
    def __init__(self, in_features: int, out_features: int, activate: bool):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU(inplace=True) if activate else None

    @staticmethod
    def laplacian_batch(graph: torch.Tensor) -> torch.Tensor:
        inverse_sqrt_degree = (graph.sum(dim=1) + 1e-5).pow(-0.5)
        return inverse_sqrt_degree[:, :, None] * graph * inverse_sqrt_degree[:, None, :]

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        repeated = graph.unsqueeze(0).repeat(x.shape[0], 1, 1)
        result = self.fc(torch.bmm(self.laplacian_batch(repeated), x))
        return self.activation(result) if self.activation is not None else result


class GraphNet(nn.Module):
    def __init__(self, dim: int, n_pts: int):
        super().__init__()
        self.A_hat = nn.Parameter(torch.eye(n_pts))
        self.gconv1 = LAM_Gconv(dim, 2 * dim, True)
        self.gconv2 = LAM_Gconv(2 * dim, dim, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gconv2(self.gconv1(x, self.A_hat), self.A_hat)


class GraAttenLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module,
                 dropout: float):
        super().__init__()
        self.self_attn, self.feed_forward = self_attn, feed_forward
        self.sublayer = nn.ModuleList(copy.deepcopy(SublayerConnection(size, dropout))
                                     for _ in range(2))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mixed = self.sublayer[0](x, lambda normalized: self.self_attn(
            normalized, normalized, normalized, mask))
        return self.sublayer[1](mixed, self.feed_forward)


class GraFormer(nn.Module):
    def __init__(self, adj: torch.Tensor, hid_dim: int = 128, in_feat: int = 256,
                 out_dim: int = 3, num_layers: int = 4, n_head: int = 4,
                 dropout: float = 0.1, n_pts: int = 17):
        super().__init__()
        self.n_layers = num_layers
        self.register_buffer('adj', adj.clone())
        self.register_buffer('src_mask', torch.ones(1, 1, n_pts, dtype=torch.bool))
        self.gconv_input = ChebConv(in_feat, hid_dim)
        template_attention = MultiHeadedAttention(n_head, hid_dim)
        template_graph = GraphNet(hid_dim, n_pts)
        graphs, attentions = [], []
        for _ in range(num_layers):
            graphs.append(_ResChebGC(adj, hid_dim))
            attentions.append(GraAttenLayer(hid_dim, copy.deepcopy(template_attention),
                                            copy.deepcopy(template_graph), dropout))
        self.gconv_layers = nn.ModuleList(graphs)
        self.atten_layers = nn.ModuleList(attentions)
        self.gconv_output = ChebConv(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.gconv_input(x, self.adj)
        for attention_layer, graph_layer in zip(self.atten_layers, self.gconv_layers):
            result = graph_layer(attention_layer(result, self.src_mask))
        return self.gconv_output(result, self.adj)


class AntTimeAggregator(nn.Module):
    """Source default: time attention followed by antenna attention."""

    def __init__(self, C: int = 512, A: int = 3, W: int = 4,
                 out_dim: int = 256, mode: str = 'attn2'):
        super().__init__()
        if mode != 'attn2':
            raise ValueError('the frozen GraphPose-Fi experiment uses agg_mode=attn2')
        self.A, self.W, self.mode = A, W, mode
        self.proj = nn.Conv2d(C, out_dim, 1, bias=False)
        self.ln = nn.LayerNorm(out_dim)
        self.time_score = nn.Conv3d(1, 1, 1, bias=True)
        self.ant_score = nn.Conv3d(1, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.A * self.W:
            raise ValueError('aggregator requires three antenna blocks of width four')
        projected = self.proj(x)
        batch, channels, joints, _ = projected.shape
        by_antenna = projected.reshape(batch, channels, joints, self.A, self.W)
        time_weights = self.time_score(by_antenna.mean(dim=1, keepdim=True)).softmax(-1)
        time_pooled = (by_antenna * time_weights).sum(dim=-1)
        antenna_logits = self.ant_score(time_pooled.mean(dim=1, keepdim=True).unsqueeze(-1))
        antenna_weights = antenna_logits.squeeze(-1).softmax(dim=-1)
        return self.ln((time_pooled * antenna_weights).sum(dim=-1).permute(0, 2, 1))


class GraphPoseFiNet(nn.Module):
    def __init__(self, num_keypoints: int = 17, num_coor: int = 3,
                 num_person: int = 1, dataset: str = 'mmfi',
                 pretrained: bool = False, num_layers: int = 4,
                 agg_mode: str = 'attn2', subcarrier_num: int = 114):
        super().__init__()
        if (dataset not in ('mmfi', 'mmfi-csi') or num_keypoints != 17 or
                num_coor != 3 or num_person != 1 or subcarrier_num != 114):
            raise ValueError('this GraphPose-Fi adapter is MM-Fi, 17 joints, 3D, one person only')
        if pretrained:
            raise ValueError('this frozen experiment is from scratch; pretrained weights are disabled')
        if num_layers != 4:
            raise ValueError('the frozen source default uses four GraFormer layers')
        from torchvision.models import resnet34
        backbone = resnet34(weights=None)
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.dataset, self.subcarrier_num = num_person, dataset, subcarrier_num
        self.f_conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.f_bn1, self.f_relu = backbone.bn1, backbone.relu
        self.f_layer1, self.f_layer2 = backbone.layer1, backbone.layer2
        self.f_layer3, self.f_layer4 = backbone.layer3, backbone.layer4
        self.agg = AntTimeAggregator(mode=agg_mode)
        self.gnn_in_dim = 256
        self.graformer = GraFormer(mmfi_adjacency(), num_layers=num_layers)
        self.bn2 = nn.BatchNorm2d(512)

    def _encode_freq(self, x: torch.Tensor) -> torch.Tensor:
        result = self.f_relu(self.f_bn1(self.f_conv1(x)))
        for layer in (self.f_layer1, self.f_layer2, self.f_layer3, self.f_layer4):
            result = layer(result)
        return result

    def forward(self, x: torch.Tensor):
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 4 or x.shape[1:3] != (3, 114) or x.shape[3] != 10:
            raise ValueError('GraphPose-Fi expects MM-Fi CSI [B,3,114,10] (optional final singleton)')
        # Retain the source permutation/contiguity as well as its logical
        # shape: singleton-channel strides can select different CPU kernels.
        native_layout = x.unsqueeze(-1).permute(0, 4, 1, 2, 3).contiguous()
        encoded = []
        for antenna in range(3):
            # torchvision.transforms.Resize([136,32]) defaults to bilinear,
            # align_corners=False and antialias=True in the source environment.
            resized = F.interpolate(native_layout[:, :, antenna], size=(136, 32),
                                    mode='bilinear', align_corners=False, antialias=True)
            encoded.append(self._encode_freq(resized))
        features = self.agg(self.bn2(torch.cat(encoded, dim=3)))
        return self.graformer(features).unsqueeze(1), features


def graphpose_init(module: nn.Module) -> None:
    """Author post-construction initialization; caller invokes model.apply()."""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
