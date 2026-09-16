import torch
import torch.nn as nn
import torch.nn.functional as F
from models.expertset import Expertset  # 你的 Expertset 类
# this have 2 inputs
class Expert(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, in_channels // 2, kernel_size=1)
        )

    def forward(self, x):
        return self.fuse(x)

class GatedSparseExpertFusion(nn.Module):
    def __init__(self, in_channels, num_experts=4, hidden_channels=64):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels

        # 门控网络
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_experts, kernel_size=1)
        )

        # 多个专家网络
        self.experts = nn.ModuleList([
            Expert(in_channels * 2, hidden_channels) for _ in range(num_experts)
        ])

    def forward(self, ir_feat, vis_feat):
        """
        ir_feat:  [B, T, C, H, W]
        vis_feat: [B, T, C, H, W]
        return:   [B, T, C, H, W]
        """
        B, T, C, H, W = ir_feat.shape
        x = torch.cat([ir_feat, vis_feat], dim=2)  # [B, T, 2C, H, W]
        x_flat = x.view(B * T, -1, H, W)  # [B*T, 2C, H, W]

        # Gate logits -> softmax
        gate_logits = self.gate(x_flat.unsqueeze(2))  # [B*T, num_experts, 1, 1, 1]
        gate_logits = gate_logits.squeeze(-1).squeeze(-1).squeeze(-1)  # [B*T, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B*T, num_experts]

        # Expert 分派
        expert_selector = Expertset(self.num_experts, gate_probs.detach())

        # 将输入分发给专家
        x_input = x_flat.unsqueeze(2)  # [B*T, 2C, 1, H, W]
        x_input_split = expert_selector.es(x_input)  # list of [Ni, 2C, 1, H, W]

        expert_outs = []
        for i, expert in enumerate(self.experts):
            if x_input_split[i].shape[0] == 0:
                expert_outs.append(torch.zeros(0, C, 1, H, W).to(x.device))
                continue
            out = expert(x_input_split[i])  # [Ni, C, 1, H, W]
            expert_outs.append(out)

        # 结果组合
        fused_out = expert_selector.ee(expert_outs)  # [B*T, C]
        fused_out = fused_out.view(B, T, C, H, W)
        return fused_out