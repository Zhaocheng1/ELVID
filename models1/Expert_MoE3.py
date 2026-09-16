import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.expertset import Expertset  # 你的 Expertset 类
import numpy as np
import math
# this have 3 inputs


def gumbel_softmax(logits, tau=1.0, eps=1e-8):
    noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    return F.softmax((logits + noise) / tau, dim=2)


# class CrossLayerExpertFusion(nn.Module):
#     def __init__(self, in_channel=64, target_channel=128, num_experts=4, gumbel_tau=0.5):
#         super().__init__()
#         self.num_experts = num_experts
#         self.gumbel_tau = gumbel_tau  # 可调温度参数
#
#         # 下采样模块
#         self.downsample = nn.Sequential(
#             nn.Conv3d(in_channel, target_channel, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
#             nn.BatchNorm3d(target_channel),
#             nn.ReLU(inplace=True),
#         )
#
#         # 门控网络
#         self.gate = nn.Sequential(
#             nn.Conv3d(target_channel * 3, in_channel, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.GroupNorm(4, in_channel),  # 更稳定的小 batch norm
#             nn.Conv3d(in_channel, num_experts, kernel_size=1)
#         )
#
#         # 多专家网络
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv3d(target_channel * 3, target_channel, kernel_size=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(target_channel, target_channel, kernel_size=3, padding=1)
#             )
#             for _ in range(num_experts)
#         ])
#
#     def forward(self, fusion_layer1, ir_feat, vis_feat):
#         B, T, _, _, _ = fusion_layer1.shape
#         B, T, C, H, W = ir_feat.shape
#
#         # 统一通道，拼接特征
#         fusion_feat = self.downsample(fusion_layer1.permute(0, 2, 1, 3, 4))  # → [B, target_C, T, H//2, W//2]
#         fusion_feat = fusion_feat.permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']
#         x = torch.cat([fusion_feat, ir_feat, vis_feat], dim=2)  # → [B, T, 3*C, H, W]
#
#         # 计算门控
#         gate_input = x.permute(0, 2, 1, 3, 4)  # [B, 3C, T, H, W]
#         gates_raw = self.gate(gate_input)     # [B, E, T, H, W]
#
#         # 使用 Gumbel Softmax 计算门控
#         if self.training:
#             gates = F.gumbel_softmax(gates_raw.permute(0, 2, 1, 3, 4), tau=self.gumbel_tau, dim=2)
#         else:
#             gates = F.softmax(gates_raw.permute(0, 2, 1, 3, 4), dim=2)
#
#         gates = gates.permute(0, 2, 1, 3, 4).contiguous()  # [B, E, T, H, W]
#         gates = gates.view(-1, self.num_experts)  # [N, E]
#
#
#         # === Load Balancing Loss ===
#         avg_gate = gates.mean(dim=0)  # [E]
#         load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
#
#         # === Sparsity Loss ===
#         entropy_per_sample = - (gates * torch.log(gates + 1e-8)).sum(dim=1)
#         sparsity_loss = entropy_per_sample.mean()
#
#         # Expert 分发 + 聚合
#         expertset = Expertset(self.num_experts, gates)
#         x = x.permute(0, 2, 1, 3, 4).reshape(B, 3 * C, -1).permute(2, 0, 1).reshape(-1, 3 * C)  # [N, 3C]
#         expert_inputs = expertset.es(x)
#
#         expert_outputs = []
#         for i, expert in enumerate(self.experts):
#             input_i = expert_inputs[i].view(-1, 3 * C, 1, 1, 1)
#             out_i = expert(input_i).view(-1, C)
#             expert_outputs.append(out_i)
#
#         out = expertset.ee(expert_outputs)  # [N, C]
#         out = out.view(B, T, C, H, W)
#
#         return {
#             "fused_output": out,
#             "load_balancing_loss": load_balancing_loss,
#             "sparsity_loss": sparsity_loss
# class CrossLayerExpertFusion(nn.Module):
#     def __init__(self, in_channel=64, target_channel=128, num_experts=4, use_experts=2, noise_epsilon=1e-2):
#         super().__init__()
#         self.num_experts = num_experts
#         self.use_experts = use_experts
#         self.noise_epsilon = noise_epsilon
#
#         self.mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
#
#         # 下采样模块
#         self.downsample = nn.Sequential(
#             nn.Conv3d(in_channel, target_channel, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
#             nn.BatchNorm3d(target_channel),
#             nn.ReLU(inplace=True),
#         )
#
#         # 门控参数（用于 noisy-top-k）
#         self.w_g = nn.Parameter(torch.randn(3 * target_channel, num_experts))
#         self.w_n = nn.Parameter(torch.zeros(3 * target_channel, num_experts))
#         self.softplus = nn.Softplus()
#
#         # 多专家网络
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(3 * target_channel, target_channel),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(target_channel, target_channel)
#             )
#             for _ in range(num_experts)
#         ])
#
#     def forward(self, fusion_layer1, ir_feat, vis_feat):
#         B, T, _, _, _ = fusion_layer1.shape
#         B, T, C, H, W = ir_feat.shape
#
#         # 特征准备
#         fusion_feat = self.downsample(fusion_layer1.permute(0, 2, 1, 3, 4))
#         fusion_feat = fusion_feat.permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']
#         x = torch.cat([fusion_feat, ir_feat, vis_feat], dim=2)  # → [B, T, 3*C, H, W]
#
#         x = x.permute(0, 2, 1, 3, 4).reshape(B, 3 * C, -1).permute(2, 0, 1).reshape(-1, 3 * C)  # [N, 3C]
#
#         # clean logits
#         clean_logits = x @ self.w_g
#
#         if self.training:
#             noise_stddev = self.softplus(x @ self.w_n) + self.noise_epsilon
#             noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
#             logits = noisy_logits
#         else:
#             logits = clean_logits
#
#         top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
#         top_gates = F.softmax(top_logits, dim=-1)
#
#         gates = torch.zeros_like(logits)
#         gates.scatter_(1, top_indices, top_gates)
#
#         # === Loss ===
#         avg_gate = gates.mean(dim=0)
#         load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
#         entropy_per_sample = -(gates * torch.log(gates + 1e-8)).sum(dim=1)#熵最大化 —— 所有专家都参与（这可能与 sp_loss 初衷不符） -(gates * torch.log(gates + 1e-8)).sum(dim=1)熵最小化 让门控输出更稀疏（熵越小越好 → 更偏向 one-hot）
#         sparsity_loss = entropy_per_sample.mean()
#         # max_entropy = math.log(gates.shape[1])
#         # sparsity_loss = (max_entropy - entropy_per_sample).mean()  # 正数，稀疏越大，loss 越大
#
#         # Expert 路由
#         expertset = Expertset(self.num_experts, gates)
#         expert_inputs = expertset.es(x)
#
#         expert_outputs = []
#         for i, expert in enumerate(self.experts):
#             out_i = expert(expert_inputs[i])
#             expert_outputs.append(out_i)
#
#         out = expertset.ee(expert_outputs)
#         out = out.view(B, T, C, H, W)
#
#         return {
#             "fused_output": out,
#             "load_balancing_loss": load_balancing_loss,
#             "sparsity_loss": sparsity_loss
#         }

class CrossLayerExpertFusion(nn.Module):
    def __init__(self, in_channel=64, target_channel=128, num_experts=3, use_experts=2, noise_epsilon=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.use_experts = use_experts
        self.noise_epsilon = noise_epsilon
        self.register_buffer("expert_epoch_accumulator", torch.zeros(num_experts))

        # 通道对齐
        self.ir_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )
        self.vis_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )
        self.fused_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )

        # 门控权重
        self.w_g = nn.Parameter(torch.randn(target_channel, num_experts))  # 路由权重
        self.w_n = nn.Parameter(torch.zeros(target_channel, num_experts))  # 路由噪声
        self.softplus = nn.Softplus()

        # === 三个模态的专家 ===
        self.experts = nn.ModuleList([
            nn.Sequential(  # Expert 0: IR
                nn.Linear(target_channel, target_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(target_channel // 2, target_channel)
            ),
            nn.Sequential(  # Expert 1: VIS
                nn.Linear(target_channel, target_channel),
                nn.Tanh(),
                nn.Linear(target_channel, target_channel)
            ),
            nn.Sequential(  # Expert 2: Fused
                nn.Linear(target_channel, target_channel),
                nn.ReLU(inplace=True),
                nn.Linear(target_channel, target_channel)
            )
        ])

    def forward(self, ir_feat, vis_feat, fused_feat):
        B, T, C, H, W = fused_feat.shape

        # Step 1: 通道对齐（B, C', T, H, W）
        ir = self.ir_proj(ir_feat.permute(0, 2, 1, 3, 4))
        vis = self.vis_proj(vis_feat.permute(0, 2, 1, 3, 4))
        fused = self.fused_proj(fused_feat.permute(0, 2, 1, 3, 4))

        # Step 2: 使用 Fused 特征做全局池化 → Gating
        pooled = F.adaptive_avg_pool3d(fused, (1, 1, 1)).view(B, -1)
        clean_logits = pooled @ self.w_g

        if self.training:
            noise_stddev = self.softplus(pooled @ self.w_n) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
        top_gates = F.softmax(top_logits, dim=-1)

        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_indices, top_gates)
        # expert_selection_count = gates.sum(dim=0)  # shape: [num_experts]
        # print(f"Expert selection count (batch): {expert_selection_count.tolist()}")
        with torch.no_grad():
            expert_counts = top_indices.detach().flatten()
            for idx in expert_counts:
                self.expert_epoch_accumulator[idx] += 1

        # === 路由损失 ===
        avg_gate = gates.mean(dim=0)
        load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
        sparsity_loss = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()

        # Step 3: 每个专家提取不同模态特征，reshape 为 [B*T*H*W, C']
        ir_flat = ir.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        vis_flat = vis.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        fused_flat = fused.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)

        # === 分别送入三个专家 ===
        expert_inputs = [ir_flat, vis_flat, fused_flat]
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(expert_inputs[i]))  # 每个专家处理其模态

        # === 合并输出（使用 gating 融合） ===
        gates_exp = gates.repeat_interleave(T * H * W, dim=0)  # [B*T*H*W, E]
        stacked_output = torch.stack(expert_outputs, dim=1)  # [B*T*H*W, E, C']
        fused_output = torch.sum(gates_exp.unsqueeze(-1) * stacked_output, dim=1)  # 加权融合

        out = fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()

        return {
            "fused_output": out,
            "load_balancing_loss": load_balancing_loss,
            "sparsity_loss": sparsity_loss
        }
class CrossLayerExpertFusion_MultiGate(nn.Module):
    def __init__(self, in_channel=64, target_channel=128, num_experts=3, use_experts=2, noise_epsilon=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.use_experts = use_experts
        self.noise_epsilon = noise_epsilon
        self.register_buffer("expert_epoch_accumulator", torch.zeros(num_experts))

        # === 特征投影（对齐通道） ===
        self.ir_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )
        self.vis_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )
        self.fused_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )

        # === 路由网络（每个模态一个） ===
        self.w_g_ir = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_ir = nn.Parameter(torch.zeros(target_channel, num_experts))

        self.w_g_vis = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_vis = nn.Parameter(torch.zeros(target_channel, num_experts))

        self.w_g_fused = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_fused = nn.Parameter(torch.zeros(target_channel, num_experts))

        self.softplus = nn.Softplus()

        # === 专家模块 ===
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(target_channel, target_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(target_channel // 2, target_channel)
            ),
            nn.Sequential(
                nn.Linear(target_channel, target_channel),
                nn.Tanh(),
                nn.Linear(target_channel, target_channel)
            ),
            nn.Sequential(
                nn.Linear(target_channel, target_channel),
                nn.ReLU(inplace=True),
                nn.Linear(target_channel, target_channel)
            )
        ])

    def compute_gates(self, pooled, w_g, w_n):
        clean_logits = pooled @ w_g
        if self.training:
            noise_stddev = self.softplus(pooled @ w_n) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
        top_gates = F.softmax(top_logits, dim=-1)

        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_indices, top_gates)
        return gates, top_indices

    def forward(self, ir_feat, vis_feat, fused_feat):
        B, T, C, H, W = fused_feat.shape

        ir = self.ir_proj(ir_feat.permute(0, 2, 1, 3, 4))
        vis = self.vis_proj(vis_feat.permute(0, 2, 1, 3, 4))
        fused = self.fused_proj(fused_feat.permute(0, 2, 1, 3, 4))

        pooled_ir = F.adaptive_avg_pool3d(ir, (1, 1, 1)).view(B, -1)
        pooled_vis = F.adaptive_avg_pool3d(vis, (1, 1, 1)).view(B, -1)
        pooled_fused = F.adaptive_avg_pool3d(fused, (1, 1, 1)).view(B, -1)

        gates_ir, top_idx_ir = self.compute_gates(pooled_ir, self.w_g_ir, self.w_n_ir)
        gates_vis, top_idx_vis = self.compute_gates(pooled_vis, self.w_g_vis, self.w_n_vis)
        gates_fused, top_idx_fused = self.compute_gates(pooled_fused, self.w_g_fused, self.w_n_fused)

        # === 最终 gate：平均模态投票（也可用 attention 加权）===
        gates = (gates_ir + gates_vis + gates_fused) / 3.0

        # === 路由统计 ===
        with torch.no_grad():
            expert_counts = torch.cat([top_idx_ir, top_idx_vis, top_idx_fused], dim=1).flatten()
            for idx in expert_counts:
                self.expert_epoch_accumulator[idx] += 1

        # === 路由 loss ===
        avg_gate = gates.mean(dim=0)
        load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
        sparsity_loss = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()

        # === 每个专家处理各自输入 ===
        ir_flat = ir.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        vis_flat = vis.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        fused_flat = fused.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)

        expert_inputs = [ir_flat, vis_flat, fused_flat]
        expert_outputs = [expert(expert_inputs[i]) for i, expert in enumerate(self.experts)]

        # === 融合输出 ===
        gates_exp = gates.repeat_interleave(T * H * W, dim=0)
        stacked_output = torch.stack(expert_outputs, dim=1)
        fused_output = torch.sum(gates_exp.unsqueeze(-1) * stacked_output, dim=1)

        out = fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()

        return {
            "fused_output": out,
            "load_balancing_loss": load_balancing_loss,
            "sparsity_loss": sparsity_loss
        }
class CrossLayerExpertFusion_MultiGate_TokenMoE(nn.Module):
    def __init__(self, in_channel=64, target_channel=128, num_experts=3, use_experts=2, noise_epsilon=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.use_experts = use_experts
        self.noise_epsilon = noise_epsilon
        self.register_buffer("expert_epoch_accumulator", torch.zeros(num_experts))
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 可学习融合权重

        # 通道对齐
        self.ir_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )
        self.vis_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )
        self.fused_proj = nn.Sequential(
            nn.Conv3d(in_channel, target_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(target_channel),
            nn.ReLU(inplace=True)
        )

        # 路由网络（每模态各一组）
        self.w_g_ir = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_ir = nn.Parameter(torch.zeros(target_channel, num_experts))

        self.w_g_vis = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_vis = nn.Parameter(torch.zeros(target_channel, num_experts))

        self.w_g_fused = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_fused = nn.Parameter(torch.zeros(target_channel, num_experts))

        self.softplus = nn.Softplus()

        # 专家模块（共用）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(target_channel, target_channel // 2),
                nn.ReLU(inplace=True),
                nn.Linear(target_channel // 2, target_channel)
            ),
            nn.Sequential(
                nn.Linear(target_channel, target_channel),
                nn.Tanh(),
                nn.Linear(target_channel, target_channel)
            ),
            nn.Sequential(
                nn.Linear(target_channel, target_channel),
                nn.ReLU(inplace=True),
                nn.Linear(target_channel, target_channel)
            )
        ])

    def compute_gates(self, pooled, w_g, w_n):
        clean_logits = pooled @ w_g
        if self.training:
            noise_stddev = self.softplus(pooled @ w_n) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
        top_gates = F.softmax(top_logits, dim=-1)

        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_indices, top_gates)
        return gates, top_indices

    def forward(self, ir_feat, vis_feat, fused_feat):
        B, T, C, H, W = fused_feat.shape

        # === 通道对齐 ===
        ir = self.ir_proj(ir_feat.permute(0, 2, 1, 3, 4))
        vis = self.vis_proj(vis_feat.permute(0, 2, 1, 3, 4))
        fused = self.fused_proj(fused_feat.permute(0, 2, 1, 3, 4))

        # === 全局池化（各模态） ===
        pooled_ir = F.adaptive_avg_pool3d(ir, (1, 1, 1)).view(B, -1)
        pooled_vis = F.adaptive_avg_pool3d(vis, (1, 1, 1)).view(B, -1)
        pooled_fused = F.adaptive_avg_pool3d(fused, (1, 1, 1)).view(B, -1)

        # === 计算 gating 权重 ===
        gates_ir, top_idx_ir = self.compute_gates(pooled_ir, self.w_g_ir, self.w_n_ir)
        gates_vis, top_idx_vis = self.compute_gates(pooled_vis, self.w_g_vis, self.w_n_vis)
        gates_fused, top_idx_fused = self.compute_gates(pooled_fused, self.w_g_fused, self.w_n_fused)

        # === 平均投票 gating ===
        gates = (gates_ir + gates_vis + gates_fused) / 3.0

        # === 统计专家选择 ===
        with torch.no_grad():
            expert_counts = torch.cat([top_idx_ir, top_idx_vis, top_idx_fused], dim=1).flatten()
            for idx in expert_counts:
                self.expert_epoch_accumulator[idx] += 1

        # === Loss ===
        avg_gate = gates.mean(dim=0)
        load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
        sparsity_loss = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()

        # === 展平输入 ===
        ir_flat = ir.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        vis_flat = vis.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        fused_flat = fused.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)

        # === 模态专家输出（soft gate） ===
        expert_inputs = [ir_flat, vis_flat, fused_flat]
        expert_outputs = [expert(expert_inputs[i]) for i, expert in enumerate(self.experts)]

        gates_exp = gates.repeat_interleave(T * H * W, dim=0)
        stacked_output = torch.stack(expert_outputs, dim=1)
        soft_fused_output = torch.sum(gates_exp.unsqueeze(-1) * stacked_output, dim=1)
        soft_fused_output = soft_fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()

        # === Token-Level MoE 分支 ===
        token_moe_gates, token_top_idx = self.compute_gates(pooled_fused, self.w_g_fused, self.w_n_fused)
        token_moe_gates_exp = token_moe_gates.repeat_interleave(T * H * W, dim=0)

        token_expertset = Expertset(self.num_experts, token_moe_gates_exp)
        token_inputs = token_expertset.es(fused_flat)

        token_outputs = []
        for i, expert in enumerate(self.experts):
            if len(token_inputs[i]) > 0:
                token_outputs.append(expert(token_inputs[i]))
            else:
                token_outputs.append(torch.zeros((0, fused_flat.shape[1]), device=fused_flat.device))

        token_fused_output = token_expertset.ee(token_outputs)
        token_fused_output = token_fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()

        # === 加权融合两个输出 ===
        out = self.fusion_weight * soft_fused_output + (1 - self.fusion_weight) * token_fused_output

        return {
            "fused_output": out,
            "load_balancing_loss": load_balancing_loss,
            "sparsity_loss": sparsity_loss
        }
# def compute_edge_map(x: Tensor) -> Tensor:
#     """x: [B, C, T, H, W] → returns [B, 1, T, H, W]"""
#     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
#     sobel_y = sobel_x.permute(0, 1, 3, 2)
#     x_gray = x.mean(1, keepdim=True).flatten(2, 3)  # [B, 1, T*H, W]
#     grad_x = F.conv2d(x_gray, sobel_x, padding=1)
#     grad_y = F.conv2d(x_gray, sobel_y, padding=1)
#     edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)
#     return edge.reshape(x.size(0), 1, x.size(2), x.size(3), x.size(4))
#
# def compute_motion_map(x: Tensor) -> Tensor:
#     """x: [B, C, T, H, W] → returns [B, 1, T, H, W]"""
#     diff = x[:, :, 1:] - x[:, :, :-1]
#     motion = torch.norm(diff, dim=1, keepdim=True)
#     return F.pad(motion, (0, 0, 0, 0, 1, 0), mode='replicate')
#
# class CrossLayerExpertFusion_MultiGate_TokenMoE_ExpertSet(nn.Module):
#     def __init__(self, in_channel=64, target_channel=128, num_experts=3, use_experts=2, noise_epsilon=1e-2):
#         super().__init__()
#         self.num_experts = num_experts
#         self.use_experts = use_experts
#         self.noise_epsilon = noise_epsilon
#
#         self.register_buffer("expert_epoch_accumulator", torch.zeros(num_experts))
#
#         # 模态通道对齐
#         self.ir_proj = nn.Conv3d(in_channel, target_channel, 1)
#         self.vis_proj = nn.Conv3d(in_channel, target_channel, 1)
#         self.fused_proj = nn.Conv3d(in_channel, target_channel, 1)
#
#         # 路由权重（IR/VIS/Fused）
#         self.w_g_ir = nn.Parameter(torch.randn(target_channel, num_experts))
#         self.w_n_ir = nn.Parameter(torch.zeros(target_channel, num_experts))
#         self.w_g_vis = nn.Parameter(torch.randn(target_channel, num_experts))
#         self.w_n_vis = nn.Parameter(torch.zeros(target_channel, num_experts))
#         self.w_g_fused = nn.Parameter(torch.randn(target_channel, num_experts))
#         self.w_n_fused = nn.Parameter(torch.zeros(target_channel, num_experts))
#         self.softplus = nn.Softplus()
#
#         # IR Expert（Motion attention）
#         self.expert_ir = nn.Sequential(
#             nn.Linear(target_channel, target_channel // 2),
#             nn.ReLU(),
#             nn.Linear(target_channel // 2, target_channel)
#         )
#
#         # VIS Expert（Edge attention）
#         self.expert_vis = nn.Sequential(
#             nn.Linear(target_channel, target_channel),
#             nn.Tanh(),
#             nn.Linear(target_channel, target_channel)
#         )
#
#         # Fused Expert（融合感知）
#         self.expert_fused = nn.Sequential(
#             nn.Linear(target_channel, target_channel),
#             nn.ReLU(),
#             nn.Linear(target_channel, target_channel)
#         )
#
#         self.experts = nn.ModuleList([self.expert_ir, self.expert_vis, self.expert_fused])
#
#         # Token-level routing 用于 fused_feat
#         self.token_w_g = nn.Parameter(torch.randn(target_channel, num_experts))
#         self.token_experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(target_channel, target_channel),
#                 nn.ReLU(),
#                 nn.Linear(target_channel, target_channel)
#             ) for _ in range(num_experts)
#         ])
#
#         self.final_fusion_conv = nn.Conv3d(2 * target_channel, target_channel, kernel_size=1)
#
#     def compute_gates(self, pooled, w_g, w_n):
#         clean_logits = pooled @ w_g
#         if self.training:
#             noise_std = self.softplus(pooled @ w_n) + self.noise_epsilon
#             logits = clean_logits + torch.randn_like(clean_logits) * noise_std
#         else:
#             logits = clean_logits
#         top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
#         top_gates = F.softmax(top_logits, dim=-1)
#         gates = torch.zeros_like(logits).scatter(1, top_indices, top_gates)
#         return gates, top_indices
#
#     def forward(self, ir_feat, vis_feat, fused_feat):
#         B, T, C, H, W = fused_feat.shape
#
#         # === 通道对齐 ===
#         ir = self.ir_proj(ir_feat.permute(0, 2, 1, 3, 4))
#         vis = self.vis_proj(vis_feat.permute(0, 2, 1, 3, 4))
#         fused = self.fused_proj(fused_feat.permute(0, 2, 1, 3, 4))
#
#         # === 路由 gating ===
#         pooled_ir = F.adaptive_avg_pool3d(ir, 1).view(B, -1)
#         pooled_vis = F.adaptive_avg_pool3d(vis, 1).view(B, -1)
#         pooled_fused = F.adaptive_avg_pool3d(fused, 1).view(B, -1)
#
#         gates_ir, idx_ir = self.compute_gates(pooled_ir, self.w_g_ir, self.w_n_ir)
#         gates_vis, idx_vis = self.compute_gates(pooled_vis, self.w_g_vis, self.w_n_vis)
#         gates_fused, idx_fused = self.compute_gates(pooled_fused, self.w_g_fused, self.w_n_fused)
#
#         gates = (gates_ir + gates_vis + gates_fused) / 3
#
#         with torch.no_grad():
#             all_idx = torch.cat([idx_ir, idx_vis, idx_fused], dim=1).flatten()
#             for i in all_idx:
#                 self.expert_epoch_accumulator[i] += 1
#
#         avg_gate = gates.mean(dim=0)
#         load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
#         sparsity_loss = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()
#
#         # === 模态专家执行 ===
#
#         edge_map = compute_edge_map(vis).reshape(B * T * H * W, 1)
#         motion_map = compute_motion_map(ir).reshape(B * T * H * W, 1)
#
#         ir_flat = ir.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
#         vis_flat = vis.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
#         fused_flat = fused.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
#
#
#         expert_outputs = [
#             self.expert_ir(ir_flat) * motion_map,
#             self.expert_vis(vis_flat) * edge_map,
#             self.expert_fused(fused_flat)
#         ]
#
#         gates_exp = gates.repeat_interleave(T * H * W, dim=0)
#         stacked_output = torch.stack(expert_outputs, dim=1)
#         modal_fused_output = torch.sum(gates_exp.unsqueeze(-1) * stacked_output, dim=1)
#         modal_fused_output = modal_fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)
#
#         # === Token-MoE fused 路由 ===
#         fused_tokens = fused_flat  # [B*T*H*W, C]
#         token_logits = fused_tokens @ self.token_w_g
#         token_gates = F.softmax(token_logits, dim=1)
#         moe_router = Expertset(self.num_experts, token_gates)
#         expert_inputs = moe_router.es(fused_tokens)
#         token_outputs = [self.token_experts[i](x) for i, x in enumerate(expert_inputs)]
#         token_out = moe_router.ee(token_outputs).reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)
#
#         # === 融合两个输出路径 ===
#         combined = torch.cat([modal_fused_output, token_out], dim=2)
#         out = self.final_fusion_conv(combined.permute(0, 2, 1, 3, 4))
#         out = out.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()
#         # print(out.shape)
#
#         return {
#             "fused_output": out,
#             "load_balancing_loss": load_balancing_loss,
#             "sparsity_loss": sparsity_loss,
#             "expert_usage": self.expert_epoch_accumulator.clone()
#         }
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# =============== Utility Functions ===============

def compute_edge_map(x: Tensor) -> Tensor:
    """x: [B, C, T, H, W] → returns [B, 1, T, H, W]"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    x_gray = x.mean(1, keepdim=True).flatten(2, 3)
    grad_x = F.conv2d(x_gray, sobel_x, padding=1)
    grad_y = F.conv2d(x_gray, sobel_y, padding=1)
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return edge.reshape(x.size(0), 1, x.size(2), x.size(3), x.size(4))


def compute_motion_map(x: torch.Tensor) -> torch.Tensor:
    """x: [B, C, T, H, W] → returns [B, 1, T, H, W]"""
    # 一阶时间差
    diff1 = x[:, :, 1:] - x[:, :, :-1]         # [B, C, T-1, H, W]
    motion1 = torch.norm(diff1, dim=1, keepdim=True)  # [B, 1, T-1, H, W]
    motion1 = F.pad(motion1, (0, 0, 0, 0, 1, 0), mode='replicate')  # [B, 1, T, H, W]

    # 二阶时间差（增强动态信息）
    diff2 = x[:, :, 2:] - x[:, :, :-2]         # [B, C, T-2, H, W]
    motion2 = torch.norm(diff2, dim=1, keepdim=True)  # [B, 1, T-2, H, W]
    motion2 = F.pad(motion2, (0, 0, 0, 0, 1, 1), mode='replicate')  # [B, 1, T, H, W]

    # 合并两个动态特征图
    motion = motion1 + 0.5 * motion2
    return motion
# =============== Core Module ===============

class CrossLayerExpertFusion_MultiGate_TokenMoE_ExpertSet(nn.Module):
    def __init__(self, in_channel=64, target_channel=128, num_experts=3, use_experts=2, noise_epsilon=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.use_experts = use_experts
        self.noise_epsilon = noise_epsilon

        self.register_buffer("expert_epoch_accumulator", torch.zeros(num_experts))

        self.ir_proj = nn.Conv3d(in_channel, target_channel, 1)
        self.vis_proj = nn.Conv3d(in_channel, target_channel, 1)
        self.fused_proj = nn.Conv3d(in_channel, target_channel, 1)

        # Gating weights
        self.w_g_ir = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_ir = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.w_g_vis = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_vis = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.w_g_fused = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_fused = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.softplus = nn.Softplus()

        # Experts
        self.expert_ir = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel)
        )

        self.expert_vis = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.Tanh(),
            nn.Linear(target_channel, target_channel)
        )

        self.expert_fused = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel)
        )

        self.experts = nn.ModuleList([self.expert_ir, self.expert_vis, self.expert_fused])

        # Token-level MoE
        self.token_w_g = nn.Parameter(torch.randn(target_channel, num_experts))
        self.token_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(target_channel, target_channel),
                nn.ReLU(),
                nn.Linear(target_channel, target_channel)
            ) for _ in range(num_experts)
        ])

        self.final_fusion_conv = nn.Conv3d(2 * target_channel, target_channel, kernel_size=1)

    def compute_gates(self, pooled, w_g, w_n):
        clean_logits = pooled @ w_g
        if self.training:
            noise_std = self.softplus(pooled @ w_n) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
        top_gates = F.softmax(top_logits, dim=-1)
        gates = torch.zeros_like(logits).scatter(1, top_indices, top_gates)
        return gates, top_indices

    def forward(self, ir_feat, vis_feat, fused_feat):
        B, T, C, H, W = fused_feat.shape

        ir = self.ir_proj(ir_feat.permute(0, 2, 1, 3, 4))
        vis = self.vis_proj(vis_feat.permute(0, 2, 1, 3, 4))
        fused = self.fused_proj(fused_feat.permute(0, 2, 1, 3, 4))

        # Gating
        pooled_ir = F.adaptive_avg_pool3d(ir, 1).view(B, -1)
        pooled_vis = F.adaptive_avg_pool3d(vis, 1).view(B, -1)
        pooled_fused = F.adaptive_avg_pool3d(fused, 1).view(B, -1)

        gates_ir, idx_ir = self.compute_gates(pooled_ir, self.w_g_ir, self.w_n_ir)
        gates_vis, idx_vis = self.compute_gates(pooled_vis, self.w_g_vis, self.w_n_vis)
        gates_fused, idx_fused = self.compute_gates(pooled_fused, self.w_g_fused, self.w_n_fused)

        gates = (gates_ir + gates_vis + gates_fused) / 3

        with torch.no_grad():
            all_idx = torch.cat([idx_ir, idx_vis, idx_fused], dim=1).flatten()
            for i in all_idx:
                self.expert_epoch_accumulator[i] += 1

        avg_gate = gates.mean(dim=0)
        load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
        sparsity_loss = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()

        # ========== Expert execution ==========
        edge_map = compute_edge_map(vis).reshape(B * T * H * W, 1)
        motion_map = compute_motion_map(ir).reshape(B * T * H * W, 1)

        ir_flat = ir.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        vis_flat = vis.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        fused_flat = fused.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)

        expert_outputs = [
            self.expert_ir(ir_flat) * motion_map * 1.5,
            self.expert_vis(vis_flat) * edge_map,
            self.expert_fused(fused_flat)
        ]

        gates_exp = gates.repeat_interleave(T * H * W, dim=0)
        stacked_output = torch.stack(expert_outputs, dim=1)
        modal_fused_output = torch.sum(gates_exp.unsqueeze(-1) * stacked_output, dim=1)
        modal_fused_output = modal_fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        # ========== Token-level MoE ==========
        token_logits = fused_flat @ self.token_w_g
        token_gates = F.softmax(token_logits, dim=1)


        moe_router = Expertset(self.num_experts, token_gates)
        expert_inputs = moe_router.es(fused_flat)
        token_outputs = [self.token_experts[i](x) for i, x in enumerate(expert_inputs)]
        token_out = moe_router.ee(token_outputs).reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        # ========== Fusion ==========
        combined = torch.cat([modal_fused_output, token_out], dim=2)
        out = self.final_fusion_conv(combined.permute(0, 2, 1, 3, 4))
        out = out.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()

        return {
            "fused_output": out,
            "load_balancing_loss": load_balancing_loss,
            "sparsity_loss": sparsity_loss,
            "expert_usage": self.expert_epoch_accumulator.clone()
        }

class CrossLayerExpertFusion_MultiGate_TokenMoE_ExpertSet_no_motion(nn.Module):
    def __init__(self, in_channel=64, target_channel=128, num_experts=3, use_experts=2, noise_epsilon=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.use_experts = use_experts
        self.noise_epsilon = noise_epsilon

        self.register_buffer("expert_epoch_accumulator", torch.zeros(num_experts))

        self.ir_proj = nn.Conv3d(in_channel, target_channel, 1)
        self.vis_proj = nn.Conv3d(in_channel, target_channel, 1)
        self.fused_proj = nn.Conv3d(in_channel, target_channel, 1)

        # Gating weights
        self.w_g_ir = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_ir = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.w_g_vis = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_vis = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.w_g_fused = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_fused = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.softplus = nn.Softplus()

        # Experts
        self.expert_ir = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel)
        )

        self.expert_vis = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.Tanh(),
            nn.Linear(target_channel, target_channel)
        )

        self.expert_fused = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel)
        )

        self.experts = nn.ModuleList([self.expert_ir, self.expert_vis, self.expert_fused])

        # Token-level MoE
        self.token_w_g = nn.Parameter(torch.randn(target_channel, num_experts))
        self.token_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(target_channel, target_channel),
                nn.ReLU(),
                nn.Linear(target_channel, target_channel)
            ) for _ in range(num_experts)
        ])

        self.final_fusion_conv = nn.Conv3d(2 * target_channel, target_channel, kernel_size=1)

    def compute_gates(self, pooled, w_g, w_n):
        clean_logits = pooled @ w_g
        if self.training:
            noise_std = self.softplus(pooled @ w_n) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
        top_gates = F.softmax(top_logits, dim=-1)
        gates = torch.zeros_like(logits).scatter(1, top_indices, top_gates)
        return gates, top_indices

    def forward(self, ir_feat, vis_feat, fused_feat):
        B, T, C, H, W = fused_feat.shape

        ir = self.ir_proj(ir_feat.permute(0, 2, 1, 3, 4))
        vis = self.vis_proj(vis_feat.permute(0, 2, 1, 3, 4))
        fused = self.fused_proj(fused_feat.permute(0, 2, 1, 3, 4))

        # Gating
        pooled_ir = F.adaptive_avg_pool3d(ir, 1).view(B, -1)
        pooled_vis = F.adaptive_avg_pool3d(vis, 1).view(B, -1)
        pooled_fused = F.adaptive_avg_pool3d(fused, 1).view(B, -1)

        gates_ir, idx_ir = self.compute_gates(pooled_ir, self.w_g_ir, self.w_n_ir)
        gates_vis, idx_vis = self.compute_gates(pooled_vis, self.w_g_vis, self.w_n_vis)
        gates_fused, idx_fused = self.compute_gates(pooled_fused, self.w_g_fused, self.w_n_fused)

        gates = (gates_ir + gates_vis + gates_fused) / 3

        with torch.no_grad():
            all_idx = torch.cat([idx_ir, idx_vis, idx_fused], dim=1).flatten()
            for i in all_idx:
                self.expert_epoch_accumulator[i] += 1

        avg_gate = gates.mean(dim=0)
        load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
        sparsity_loss = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()

        # ========== Expert execution ==========
        edge_map = compute_edge_map(vis).reshape(B * T * H * W, 1)
        edge_map_ir = compute_edge_map(ir).reshape(B * T * H * W, 1)
        motion_map = compute_motion_map(ir).reshape(B * T * H * W, 1)

        ir_flat = ir.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        vis_flat = vis.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        fused_flat = fused.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)

        expert_outputs = [
            self.expert_ir(ir_flat) * motion_map * 3  + edge_map_ir,
            self.expert_vis(vis_flat) * edge_map,
            self.expert_fused(fused_flat)
        ]

        gates_exp = gates.repeat_interleave(T * H * W, dim=0)
        stacked_output = torch.stack(expert_outputs, dim=1)
        modal_fused_output = torch.sum(gates_exp.unsqueeze(-1) * stacked_output, dim=1)
        modal_fused_output = modal_fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        # ========== Token-level MoE ==========
        token_logits = fused_flat @ self.token_w_g
        token_gates = F.softmax(token_logits, dim=1)


        moe_router = Expertset(self.num_experts, token_gates)
        expert_inputs = moe_router.es(fused_flat)
        token_outputs = [self.token_experts[i](x) for i, x in enumerate(expert_inputs)]
        token_out = moe_router.ee(token_outputs).reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        # ========== Fusion ==========
        combined = torch.cat([modal_fused_output, token_out], dim=2)
        out = self.final_fusion_conv(combined.permute(0, 2, 1, 3, 4))
        out = out.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()

        return {
            "fused_output": out,
            "load_balancing_loss": load_balancing_loss,
            "sparsity_loss": sparsity_loss,
            "expert_usage": self.expert_epoch_accumulator.clone()
        }

class CrossLayerExpertFusion_MultiGate_TokenMoE_ExpertSet_edgeir(nn.Module):
    def __init__(self, in_channel=64, target_channel=128, num_experts=3, use_experts=2, noise_epsilon=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.use_experts = use_experts
        self.noise_epsilon = noise_epsilon

        self.register_buffer("expert_epoch_accumulator", torch.zeros(num_experts))

        self.ir_proj = nn.Conv3d(in_channel, target_channel, 1)
        self.vis_proj = nn.Conv3d(in_channel, target_channel, 1)
        self.fused_proj = nn.Conv3d(in_channel, target_channel, 1)

        # Gating weights
        self.w_g_ir = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_ir = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.w_g_vis = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_vis = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.w_g_fused = nn.Parameter(torch.randn(target_channel, num_experts))
        self.w_n_fused = nn.Parameter(torch.zeros(target_channel, num_experts))
        self.softplus = nn.Softplus()

        # Experts
        self.expert_ir = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel)
        )

        self.expert_vis = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.Tanh(),
            nn.Linear(target_channel, target_channel)
        )

        self.expert_fused = nn.Sequential(
            nn.Linear(target_channel, target_channel),
            nn.ReLU(),
            nn.Linear(target_channel, target_channel)
        )

        self.experts = nn.ModuleList([self.expert_ir, self.expert_vis, self.expert_fused])

        # Token-level MoE
        self.token_w_g = nn.Parameter(torch.randn(target_channel, num_experts))
        self.token_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(target_channel, target_channel),
                nn.ReLU(),
                nn.Linear(target_channel, target_channel)
            ) for _ in range(num_experts)
        ])

        self.final_fusion_conv = nn.Conv3d(2 * target_channel, target_channel, kernel_size=1)

    def compute_gates(self, pooled, w_g, w_n):
        clean_logits = pooled @ w_g
        if self.training:
            noise_std = self.softplus(pooled @ w_n) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.use_experts, self.num_experts), dim=1)
        top_gates = F.softmax(top_logits, dim=-1)
        gates = torch.zeros_like(logits).scatter(1, top_indices, top_gates)
        return gates, top_indices

    def forward(self, ir_feat, vis_feat, fused_feat):
        B, T, C, H, W = fused_feat.shape

        ir = self.ir_proj(ir_feat.permute(0, 2, 1, 3, 4))
        vis = self.vis_proj(vis_feat.permute(0, 2, 1, 3, 4))
        fused = self.fused_proj(fused_feat.permute(0, 2, 1, 3, 4))

        # Gating
        pooled_ir = F.adaptive_avg_pool3d(ir, 1).view(B, -1)
        pooled_vis = F.adaptive_avg_pool3d(vis, 1).view(B, -1)
        pooled_fused = F.adaptive_avg_pool3d(fused, 1).view(B, -1)

        gates_ir, idx_ir = self.compute_gates(pooled_ir, self.w_g_ir, self.w_n_ir)
        gates_vis, idx_vis = self.compute_gates(pooled_vis, self.w_g_vis, self.w_n_vis)
        gates_fused, idx_fused = self.compute_gates(pooled_fused, self.w_g_fused, self.w_n_fused)

        gates = (gates_ir + gates_vis + gates_fused) / 3

        with torch.no_grad():
            all_idx = torch.cat([idx_ir, idx_vis, idx_fused], dim=1).flatten()
            for i in all_idx:
                self.expert_epoch_accumulator[i] += 1

        avg_gate = gates.mean(dim=0)
        load_balancing_loss = -(avg_gate * torch.log(avg_gate + 1e-8)).sum()
        sparsity_loss = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()

        # ========== Expert execution ==========
        edge_map = compute_edge_map(vis).reshape(B * T * H * W, 1)
        edge_map_ir = compute_edge_map(ir).reshape(B * T * H * W, 1)
        motion_map = compute_motion_map(ir).reshape(B * T * H * W, 1)

        ir_flat = ir.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        vis_flat = vis.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)
        fused_flat = fused.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, -1)

        alpha = 1.5
        beta = 0.7
        
        expert_outputs = [
            self.expert_ir(ir_flat) * (alpha * motion_map + beta * edge_map_ir),
            self.expert_vis(vis_flat) * edge_map,
            self.expert_fused(fused_flat)
        ]

        gates_exp = gates.repeat_interleave(T * H * W, dim=0)
        stacked_output = torch.stack(expert_outputs, dim=1)
        modal_fused_output = torch.sum(gates_exp.unsqueeze(-1) * stacked_output, dim=1)
        modal_fused_output = modal_fused_output.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        # ========== Token-level MoE ==========
        token_logits = fused_flat @ self.token_w_g
        token_gates = F.softmax(token_logits, dim=1)


        moe_router = Expertset(self.num_experts, token_gates)
        expert_inputs = moe_router.es(fused_flat)
        token_outputs = [self.token_experts[i](x) for i, x in enumerate(expert_inputs)]
        token_out = moe_router.ee(token_outputs).reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        # ========== Fusion ==========
        combined = torch.cat([modal_fused_output, token_out], dim=2)
        out = self.final_fusion_conv(combined.permute(0, 2, 1, 3, 4))
        out = out.reshape(B, T, H, W, -1).permute(0, 1, 4, 2, 3).contiguous()

        return {
            "fused_output": out,
            "load_balancing_loss": load_balancing_loss,
            "sparsity_loss": sparsity_loss,
            "expert_usage": self.expert_epoch_accumulator.clone()
        }
