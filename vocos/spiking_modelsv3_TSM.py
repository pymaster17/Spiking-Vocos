import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Optional
import random

# 假设 spikingjelly 已经安装
# pip install spikingjelly
from spikingjelly.activation_based.neuron import IFNode
from spikingjelly.activation_based.neuron import ParametricLIFNode
from spikingjelly.activation_based import functional

# 从项目原有模块中导入基类和辅助模块
from vocos.models import Backbone
from vocos.modules import AdaLayerNorm


class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM) as described in the TS-SNN paper.
    This version uses a fixed shift strategy for increased training stability.
    """
    def __init__(self, channel_folding_factor: int = 32):
        super().__init__()
        self.channel_folding_factor = channel_folding_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [T, B, C, L].
        Returns:
            torch.Tensor: Shifted tensor of the same shape.
        """
        T, B, C, L = x.shape
        
        if T < 2: # Cannot shift if there's only one timestep
            return x

        # C_k in the paper
        num_segments = self.channel_folding_factor
        if C % num_segments != 0:
            raise ValueError(f"Number of channels ({C}) must be divisible by channel_folding_factor ({num_segments}).")
        
        # C_fold in the paper
        fold_size = C // num_segments
        
        # Use fixed split points for stability as requested.
        # A common strategy is to shift 1/4 of channels left, 1/4 right, and keep 1/2 unchanged.
        g1 = num_segments // 4
        g2 = num_segments // 2

        # Initialize a zero tensor for the output
        z = torch.zeros_like(x)

        # Part 1: Shift Left (past information)
        # From group 0 to g1-1
        c_split1 = g1 * fold_size
        if c_split1 > 0:
            z[:-1, :, :c_split1, :] = x[1:, :, :c_split1, :]

        # Part 2: Shift Right (future information)
        # From group g1 to g2-1
        c_split2 = g2 * fold_size
        if c_split2 > c_split1:
            z[1:, :, c_split1:c_split2, :] = x[:-1, :, c_split1:c_split2, :]

        # Part 3: No Shift (present information)
        # From group g2 to the end
        if C > c_split2:
            z[:, :, c_split2:, :] = x[:, :, c_split2:, :]
            
        return z


class MembraneOutputLayer(nn.Module):
    """
    对SNN在时间步上的输出膜电位进行加权积分，以恢复连续值输出。
    """
    def __init__(self, timestep: int):  # 移除默认值
        super().__init__()
        self.n_steps = timestep
        # 注册一个固定的衰减系数buffer
        arr = torch.arange(self.n_steps - 1, -1, -1, dtype=torch.float32)
        coef = torch.pow(0.9, arr)
        self.register_buffer('coef', coef.view(self.n_steps, 1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (T, B, C, L)。
        Returns:
            torch.Tensor: 输出张量，形状为 (B, C, L)。
        """
        out = torch.sum(x * self.coef, dim=0)
        return out


class SpikingConvNeXtBlock(nn.Module):
    """
    遵循“Pre-spike Residual Learning”原则的脉冲版ConvNeXt模块。
    默认集成Temporal Shift Module以增强时序建模能力。
    """
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
        channel_folding_factor: int = 32,
        tsm_penalty_factor: float = 0.5,
    ):
        super().__init__()
        self.temporal_shift = TemporalShift(channel_folding_factor)
        # Penalty factor alpha from the paper, made learnable
        self.alpha = nn.Parameter(torch.tensor(tsm_penalty_factor))

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.neuron1 = ParametricLIFNode(v_threshold=1.0, step_mode='m', backend='torch')
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.neuron2 = ParametricLIFNode(v_threshold=1.0, step_mode='m', backend='torch')
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, potential: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        T, B, C, L = potential.shape
        residual_potential = potential

        # Apply Temporal Shift Module with residual connection
        # The paper's ablation study shows applying TSM in both training and inference is better.
        shifted_potential = self.temporal_shift(potential)
        # Equation (7) from TS-SNN paper: Z' = alpha * Z + X
        x_in = self.alpha * shifted_potential + potential
        
        potential_flat = x_in.flatten(0, 1)

        # 连续值路径，用于生成脉冲和幅度重注入
        x_continuous = self.dwconv(potential_flat)
        x_continuous = x_continuous.transpose(1, 2)

        # 归一化
        if self.adanorm:
            assert cond_embedding_id is not None
            x_continuous_reshaped = x_continuous.reshape(T, B, L, C)
            cond_embedding_id_expanded = cond_embedding_id.unsqueeze(0).repeat(T, 1)
            norm_out = self.norm(x_continuous_reshaped, cond_embedding_id=cond_embedding_id_expanded.flatten(0,1))
            x_continuous = norm_out.flatten(0, 1)
        else:
            x_continuous = self.norm(x_continuous)

        # 脉冲路径
        spikes = self.neuron1(x_continuous.reshape(T, B, L, C))
        spikes_flat = spikes.flatten(0, 1)
        h_spikes = self.pwconv1(spikes_flat)
        h_spikes = self.act(h_spikes)
        h_spikes = h_spikes.reshape(T, B, L, -1)
        
        spikes = self.neuron2(h_spikes)
        spikes_flat = spikes.flatten(0,1)
        h_spikes = self.pwconv2(spikes_flat)

        # 幅度信息重注入
        modulated_out = h_spikes * x_continuous

        if self.gamma is not None:
            modulated_out = self.gamma * modulated_out
            
        modulated_out = modulated_out.transpose(1, 2)
        residual_inc = modulated_out.reshape(T, B, C, L)
        
        # 残差连接
        new_potential = residual_potential + residual_inc
        return new_potential


class SpikingVocosBackbone(Backbone):
    """
    SpikingVocos的主干网络。它将输入特征视为初始膜电位，
    通过一系列SpikingConvNeXtBlock进行处理，并最终输出一个连续值的特征图。
    """
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        snn_timestep: int,  # 移除默认值，必须由配置文件提供
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        # TSM specific parameters
        channel_folding_factor: int = 32,
        tsm_penalty_factor: float = 0.5,
    ):
        super().__init__()
        self.snn_timestep = snn_timestep
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList()
        for i in range(num_layers):
            self.convnext.append(
                SpikingConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                    channel_folding_factor=channel_folding_factor,
                    tsm_penalty_factor=tsm_penalty_factor,
                )
            )

        self.membrane_output = MembraneOutputLayer(timestep=self.snn_timestep)
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        
        # Reset SNN states before processing a new batch
        functional.reset_net(self.convnext)
        
        # 1. 输入嵌入并视为初始膜电位
        x = self.embed(x)
        
        # 2. 扩展时间维度以进行SNN仿真
        potential = x.unsqueeze(0).repeat(self.snn_timestep, 1, 1, 1)

        # 3. 通过脉冲主干网络
        for conv_block in self.convnext:
            potential = conv_block(potential, cond_embedding_id=bandwidth_id)
            
        # 4. 膜电位积分输出
        x = self.membrane_output(potential)
        
        # 5. 最终归一化和维度调整
        x = self.final_layer_norm(x.transpose(1, 2))
        return x
