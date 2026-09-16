import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianEdge(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        elif kernel_size == 5:
            kernel = torch.tensor([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]], dtype=torch.float32)
        else:
            raise ValueError("Unsupported kernel size")
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        weight = self.kernel.to(device).repeat(C, 1, 1, 1)
        edge = F.conv2d(x, weight, padding=self.kernel.shape[-1] // 2, groups=C)
        return edge

class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        weight_x = self.sobel_x.to(device).repeat(C, 1, 1, 1)
        weight_y = self.sobel_y.to(device).repeat(C, 1, 1, 1)
        edge_x = F.conv2d(x, weight_x, padding=1, groups=C)
        edge_y = F.conv2d(x, weight_y, padding=1, groups=C)
        magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return magnitude

class SpatialPyramidAttention(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(out_channels * 3, 1, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        s1 = F.adaptive_avg_pool2d(x, 1)
        s2 = F.adaptive_avg_pool2d(x, 2)
        s3 = F.adaptive_avg_pool2d(x, 4)
        spp_feat = torch.cat([
            F.interpolate(s1, size=size, mode='bilinear', align_corners=False),
            F.interpolate(s2, size=size, mode='bilinear', align_corners=False),
            F.interpolate(s3, size=size, mode='bilinear', align_corners=False)
        ], dim=1)
        attn = torch.sigmoid(self.conv(spp_feat))
        return attn

class MultiScaleFusionBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.attn = SpatialPyramidAttention(in_channels)
        self.edge3 = LaplacianEdge(kernel_size=3)
        self.edge5 = LaplacianEdge(kernel_size=5)
        self.sobel = SobelEdge()
        self.downsample_F = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def enhance_contrast_gamma(self, x, gamma=0.85):
        # gamma < 1 变亮；>1 变暗
        x = x.clamp(0, 1)
        return x.pow(gamma)

    def enhance_contrast_sigmoid(self, x, alpha=4.0):
        # beta 自适应为每通道均值，更稳
        beta = x.mean(dim=(2, 3), keepdim=True)  # [B*T, C, 1, 1]
        return torch.sigmoid(alpha * (x - beta))

    def forward(self, ir_feat, vis_feat, prev_fused_feat=None,
                use_gamma=True, gamma=0.85, alpha=4.0,
                w_sobel=0.15, w_edge_mix=0.6):
        """
        可调参数解释：
          use_gamma: True 用 gamma 增强；False 用 sigmoid 门控
          gamma:     0.8~0.95（偏亮）；>1 会变暗
          alpha:     2~6 之间；过大会两极化
          w_sobel:   0.1~0.2 细节量
          w_edge_mix:0.4~0.8 边缘凸组合的强度
        """
        B, T, C, H, W = ir_feat.shape
        ir = ir_feat.view(B * T, C, H, W)
        vis = vis_feat.view(B * T, C, H, W)

        # ---------- 1) 可见光增强（二选一） ----------
        if use_gamma:
            vis_contrast = self.enhance_contrast_gamma(vis, gamma=gamma)
        else:
            gate = self.enhance_contrast_sigmoid(vis, alpha=alpha)  # [0,1]
            vis_contrast = (vis * gate).clamp(0, 1)

        # ---------- 2) 细节项（归一后小系数混入） ----------
        sob_v = self.sobel(vis_contrast)
        sob_v = sob_v / (sob_v.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        vis_detail = (vis_contrast + w_sobel * sob_v).clamp(0, 1)

        # ---------- 3) 注意力引导的凸组合（带安全带） ----------
        diff = torch.abs(ir - vis_detail)
        attn_map = torch.sigmoid(self.attn(diff)) * 0.8 + 0.1  # ∈[0.1,0.9]
        s1 = ir * attn_map + vis_detail * (1 - attn_map)

        # ---------- 4) 边缘引导的 softmax 权重 ----------
        edge_ir = self.sobel(ir).abs().mean(dim=1, keepdim=True)  # [B*T,1,H,W]
        edge_vis = self.sobel(vis_detail).abs().mean(dim=1, keepdim=True)  # [B*T,1,H,W]
        w = torch.stack([edge_ir, edge_vis], dim=1)  # [B*T,2,1,H,W]
        w = torch.softmax(w, dim=1)
        ir_weighted = ir * w[:, 0]
        vis_weighted = vis_detail * w[:, 1]

        fused_bt = s1 + w_edge_mix * (ir_weighted + vis_weighted)  # [B*T,C,H,W]
        fused = fused_bt.view(B, T, C, H, W)

        # ---------- 5) 上一层残差：修正 view/通道对齐 ----------
        # 融合上一层特征（下采样）
        # 注意 prev_fused_feat 输入是 [B, T, C_prev, H_prev, W_prev]
        B, T, C1, H1, W1 = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C1, H1, W1)
        # prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']
        fused = fused +  prev_fused_feat
        return fused
class MultiScaleFusionBlock1(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.attn = SpatialPyramidAttention(in_channels)
        self.edge3 = LaplacianEdge(kernel_size=3)
        self.edge5 = LaplacianEdge(kernel_size=5)
        self.sobel = SobelEdge()
        self.downsample_F = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def enhance_contrast_gamma(self, x, gamma=0.85):
        # gamma < 1 变亮；>1 变暗
        x = x.clamp(0, 1)
        return x.pow(gamma)

    def enhance_contrast_sigmoid(self, x, alpha=4.0):
        # beta 自适应为每通道均值，更稳
        beta = x.mean(dim=(2, 3), keepdim=True)  # [B*T, C, 1, 1]
        return torch.sigmoid(alpha * (x - beta))
    def forward(self, ir_feat, vis_feat, prev_fused_feat=None,
                use_gamma=True, gamma=0.85, alpha=4.0,
                w_sobel=0.15, w_edge_mix=0.6):
        """
        可调参数解释：
          use_gamma: True 用 gamma 增强；False 用 sigmoid 门控
          gamma:     0.8~0.95（偏亮）；>1 会变暗
          alpha:     2~6 之间；过大会两极化
          w_sobel:   0.1~0.2 细节量
          w_edge_mix:0.4~0.8 边缘凸组合的强度
        """
        B, T, C, H, W = ir_feat.shape
        ir = ir_feat.view(B * T, C, H, W)
        vis = vis_feat.view(B * T, C, H, W)

        # ---------- 1) 可见光增强（二选一） ----------
        if use_gamma:
            vis_contrast = self.enhance_contrast_gamma(vis, gamma=gamma)
        else:
            gate = self.enhance_contrast_sigmoid(vis, alpha=alpha)  # [0,1]
            vis_contrast = (vis * gate).clamp(0, 1)

        # ---------- 2) 细节项（归一后小系数混入） ----------
        sob_v = self.sobel(vis_contrast)
        sob_v = sob_v / (sob_v.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        vis_detail = (vis_contrast + w_sobel * sob_v).clamp(0, 1)

        # ---------- 3) 注意力引导的凸组合（带安全带） ----------
        diff = torch.abs(ir - vis_detail)
        attn_map = torch.sigmoid(self.attn(diff)) * 0.8 + 0.1  # ∈[0.1,0.9]
        s1 = ir * attn_map + vis_detail * (1 - attn_map)

        # ---------- 4) 边缘引导的 softmax 权重 ----------
        edge_ir = self.sobel(ir).abs().mean(dim=1, keepdim=True)  # [B*T,1,H,W]
        edge_vis = self.sobel(vis_detail).abs().mean(dim=1, keepdim=True)  # [B*T,1,H,W]
        w = torch.stack([edge_ir, edge_vis], dim=1)  # [B*T,2,1,H,W]
        w = torch.softmax(w, dim=1)
        ir_weighted = ir * w[:, 0]
        vis_weighted = vis_detail * w[:, 1]

        fused_bt = s1 + w_edge_mix * (ir_weighted + vis_weighted)  # [B*T,C,H,W]
        fused = fused_bt.view(B, T, C, H, W)

        # ---------- 5) 上一层残差：修正 view/通道对齐 ----------
        B, T, C1, H1, W1 = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C1, H1, W1)
        prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']
        fused = fused + prev_fused_feat
        return fused
class MultiScaleFusionBlock2(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.attn = SpatialPyramidAttention(out_channels)
        self.edge3 = LaplacianEdge(kernel_size=3)
        self.edge5 = LaplacianEdge(kernel_size=5)
        self.sobel = SobelEdge()
        self.downsample_F = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def enhance_contrast_gamma(self, x, gamma=0.85):
        # gamma < 1 变亮；>1 变暗
        x = x.clamp(0, 1)
        return x.pow(gamma)

    def enhance_contrast_sigmoid(self, x, alpha=4.0):
        # beta 自适应为每通道均值，更稳
        beta = x.mean(dim=(2, 3), keepdim=True)  # [B*T, C, 1, 1]
        return torch.sigmoid(alpha * (x - beta))
    def forward(self, ir_feat, vis_feat, prev_fused_feat=None,
                use_gamma=True, gamma=0.6, alpha=2.0,
                w_sobel=0.15, w_edge_mix=0.6):
        """
        可调参数解释：
          use_gamma: True 用 gamma 增强；False 用 sigmoid 门控
          gamma:     0.8~0.95（偏亮）；>1 会变暗
          alpha:     2~6 之间；过大会两极化
          w_sobel:   0.1~0.2 细节量
          w_edge_mix:0.4~0.8 边缘凸组合的强度
        """
        B, T, C, H, W = ir_feat.shape
        ir = ir_feat.view(B * T, C, H, W)
        vis = vis_feat.view(B * T, C, H, W)

        # ---------- 1) 可见光增强（二选一） ----------
        if use_gamma:
            vis_contrast = self.enhance_contrast_gamma(vis, gamma=gamma)
        else:
            gate = self.enhance_contrast_sigmoid(vis, alpha=alpha)  # [0,1]
            vis_contrast = (vis * gate).clamp(0, 1)

        # ---------- 2) 细节项（归一后小系数混入） ----------
        sob_v = self.sobel(vis_contrast)
        sob_v = sob_v / (sob_v.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        vis_detail = (vis_contrast + w_sobel * sob_v).clamp(0, 1)

        # ---------- 3) 注意力引导的凸组合（带安全带） ----------
        diff = torch.abs(ir - vis_detail)
        attn_map = torch.sigmoid(self.attn(diff)) * 0.8 + 0.1  # ∈[0.1,0.9]
        s1 = ir * attn_map + vis_detail * (1 - attn_map)

        # ---------- 4) 边缘引导的 softmax 权重 ----------
        edge_ir = self.sobel(ir).abs().mean(dim=1, keepdim=True)  # [B*T,1,H,W]
        edge_vis = self.sobel(vis_detail).abs().mean(dim=1, keepdim=True)  # [B*T,1,H,W]
        w = torch.stack([edge_ir, edge_vis], dim=1)  # [B*T,2,1,H,W]
        w = torch.softmax(w, dim=1)
        ir_weighted = ir * w[:, 0]
        vis_weighted = vis_detail * w[:, 1]

        fused_bt = s1 + w_edge_mix * (ir_weighted + vis_weighted)  # [B*T,C,H,W]
        fused = fused_bt.view(B, T, C, H, W)

        # ---------- 5) 上一层残差：修正 view/通道对齐 ----------
        B, T, C1, H1, W1 = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C1, H1, W1)
        prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']
        fused = fused + prev_fused_feat
        return fused

class ConfidenceWeightedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 上一层融合特征 F 的通道数
            out_channels: 当前 ir/vis 特征的通道数（融合后的通道数）
        """
        super(ConfidenceWeightedFusion, self).__init__()
        self.downsample_F = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, ir_feat, vis_feat, conf_ir, conf_vis, prev_fused_feat):
        """
        Args:
            ir_feat:         [B, T, C, H, W]
            vis_feat:        [B, T, C, H, W]
            conf_ir:         [B, T, H, W]
            conf_vis:        [B, T, H, W]
            prev_fused_feat: [B, T, C_F, H_F, W_F] 上一层融合特征
        Returns:
            fused_feat: [B, T, C, H, W]
        """
        B, T, C, H, W = ir_feat.shape
        B, T, C1, H1, W1 = prev_fused_feat.shape


        # 归一化置信度（防止溢出）
        eps = 1e-6
        conf_ir = conf_ir.expand(B, T, C, H, W)
        conf_vis = conf_vis.expand(B, T, C, H, W)
        alpha_ir = conf_ir / (conf_ir + conf_vis + eps)    # [B, T, H, W]
        alpha_vis = conf_vis / (conf_ir + conf_vis + eps)  # [B, T, H, W]
        print("alpha_vis range:", alpha_vis.min().item(), alpha_vis.max().item())
        print("alpha_ir range:", alpha_ir.min().item(), alpha_ir.max().item())


        # 置信度加权融合 → [B, T, C, H, W]
        fused = alpha_ir * ir_feat + alpha_vis * vis_feat

        # 融合上一层特征（下采样）
        # 注意 prev_fused_feat 输入是 [B, T, C_prev, H_prev, W_prev]
        B, T, C1, H1, W1 = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B*T,C1,H1,W1)
        prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)   # → [B, T, target_C, H', W']

        fused = fused +  prev_fused_feat # 残差连接（可选）

        return fused

class ConfidenceWeightedFusion1(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 上一层融合特征 F 的通道数
            out_channels: 当前 ir/vis 特征的通道数（融合后的通道数）
        """
        super(ConfidenceWeightedFusion1, self).__init__()
        self.out_channels = out_channels

        self.downsample_F = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ir_feat, vis_feat, conf_ir, conf_vis, prev_fused_feat):
        """
        Args:
            ir_feat:         [B, T, C, H, W]
            vis_feat:        [B, T, C, H, W]
            conf_ir:         [B, T, H, W]
            conf_vis:        [B, T, H, W]
            prev_fused_feat: [B, T, C_F, H_F, W_F]
        Returns:
            fused_feat:      [B, T, C, H, W]
        """
        B, T, C, H, W = ir_feat.shape

        # ==== 置信度加权 ====
        eps = 1e-6
        # conf_ir = conf_ir.unsqueeze(2)  # → [B, T, 1, H, W]
        # conf_vis = conf_vis.unsqueeze(2)

        alpha_ir = conf_ir / (conf_ir + conf_vis + eps)    # [B, T, 1, H, W]
        alpha_vis = conf_vis / (conf_ir + conf_vis + eps)
        print("alpha_vis range:", alpha_vis.min().item(), alpha_vis.max().item())
        print("alpha_ir range:", alpha_ir.min().item(), alpha_ir.max().item())

        fused = alpha_ir * ir_feat + alpha_vis * vis_feat  # [B, T, C, H, W]

        # ==== 下采样上一层特征 ====
        B, T, C_prev, H_prev, W_prev = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B*T, C_prev, H_prev, W_prev)
        prev_fused_feat = self.downsample_F(prev_fused_feat)  # → [B*T, out_channels, H, W]
        prev_fused_feat = prev_fused_feat.view(B, T, self.out_channels, H, W)

        # ==== 残差加和 ====
        fused = fused + prev_fused_feat  # [B, T, C, H, W]

        return fused

# class DualModalGraphCrossAttention(nn.Module):
#     def __init__(self, in_channels, patch_size=(2, 2), use_confidence=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.patch_h, self.patch_w = patch_size
#         self.use_confidence = use_confidence
#
#         # GAT-style cross attention
#         self.query_proj = nn.Linear(in_channels, in_channels)
#         self.key_proj = nn.Linear(in_channels, in_channels)
#         self.value_proj = nn.Linear(in_channels, in_channels)
#         self.final_proj = nn.Linear(in_channels, in_channels)
#
#     def _patchify(self, x):
#         B, T, C, H, W = x.shape
#         ph, pw = self.patch_h, self.patch_w
#         assert H % ph == 0 and W % pw == 0
#
#         x = x.permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]
#         x = x.reshape(B, T, H // ph, ph, W // pw, pw, C)
#         x = x.permute(0, 1, 2, 4, 3, 5, 6)  # [B, T, H', W', ph, pw, C]
#         x = x.reshape(B, T, -1, ph * pw * C)  # [B, T, N, D]
#         return x  # 每帧的 N 个图节点，每个节点 D 维特征
#
#     def forward(self, vis_feat, ir_feat, conf_vis=None, conf_ir=None):
#         # vis_feat, ir_feat: [B, T, C, H, W]
#         B, T, C, H, W = vis_feat.shape
#
#         vis_nodes = self._patchify(vis_feat)  # [B, T, N, D]
#         ir_nodes = self._patchify(ir_feat)    # [B, T, N, D]
#         B, T, N, D = vis_nodes.shape
#
#         vis_nodes = vis_nodes.view(B * T * N, D)
#         ir_nodes = ir_nodes.view(B * T * N, D)
#
#         q_vis = self.query_proj(vis_nodes)
#         k_ir = self.key_proj(ir_nodes)
#         v_ir = self.value_proj(ir_nodes)
#
#         attn_logits = torch.bmm(q_vis.unsqueeze(1), k_ir.unsqueeze(2)).squeeze(1)  # [B*T*N, 1]
#         attn_weights = F.softmax(attn_logits, dim=-1)  # [B*T*N, 1]
#
#         if self.use_confidence and conf_ir is not None and conf_vis is not None:
#             # conf: [B, T, H, W] → patch-wise → [B, T, N]
#             conf_ir_patch = self._patchify(conf_ir.unsqueeze(2)).squeeze(-1)  # [B, T, N]
#             conf_vis_patch = self._patchify(conf_vis.unsqueeze(2)).squeeze(-1)
#             conf_ir_flat = conf_ir_patch.view(B * T * N, 1)
#             conf_vis_flat = conf_vis_patch.view(B * T * N, 1)
#
#             total_conf = conf_ir_flat + conf_vis_flat + 1e-6
#             alpha = conf_vis_flat / total_conf
#         else:
#             alpha = torch.tensor(0.5, device=vis_feat.device)
#
#         attended_ir = attn_weights * v_ir  # [B*T*N, D]
#         fused_nodes = alpha * vis_nodes + (1 - alpha) * attended_ir  # [B*T*N, D]
#
#         fused_nodes = self.final_proj(fused_nodes)  # [B*T*N, D]
#         fused_nodes = fused_nodes.view(B, T, N, D)
#
#         # unpatchify
#         H_p, W_p = H // self.patch_h, W // self.patch_w
#         fused_nodes = fused_nodes.view(B, T, H_p, W_p, self.patch_h, self.patch_w, C)
#         fused_nodes = fused_nodes.permute(0, 1, 2, 4, 3, 5, 6)
#         fused_nodes = fused_nodes.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
#
#         return fused_nodes
class CrossWeightedFusion(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=(2, 2)):
        super(CrossWeightedFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_h, self.patch_w = patch_size
        patch_dim = patch_size[0] * patch_size[1] * in_channels

        # VIS → IR
        self.query_proj_vis = nn.Linear(patch_dim, in_channels)
        self.key_proj_ir = nn.Linear(patch_dim, in_channels)
        self.value_proj_ir = nn.Linear(patch_dim, in_channels)
        # IR → VIS
        self.query_proj_ir = nn.Linear(patch_dim, in_channels)
        self.key_proj_vis = nn.Linear(patch_dim, in_channels)
        self.value_proj_vis = nn.Linear(patch_dim, in_channels)

        self.final_proj = nn.Linear(in_channels, patch_size[0] * patch_size[1] * out_channels)

        self.downsample_F = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def _patchify(self, x):
        B, T, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        x = x.permute(0, 1, 3, 4, 2).reshape(B, T, H // ph, ph, W // pw, pw, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, -1, ph * pw * C)
        return x

    def forward(self, ir_feat, vis_feat, prev_fused_feat):
        B, T, C, H, W = ir_feat.shape
        # ir_feat = torch.nan_to_num(ir_feat)
        # vis_feat = torch.nan_to_num(vis_feat)

        ir_patch = self._patchify(ir_feat)  # [B, T, N, D]
        vis_patch = self._patchify(vis_feat)

        B, T, N, D = vis_patch.shape
        ir_flat = ir_patch.view(B * T * N, D)
        vis_flat = vis_patch.view(B * T * N, D)

        # VIS → IR
        q_vis = self.query_proj_vis(vis_flat)
        k_ir = self.key_proj_ir(ir_flat)
        v_ir = self.value_proj_ir(ir_flat)
        attn_v2i = F.softmax(torch.bmm(q_vis.unsqueeze(1), k_ir.unsqueeze(2)).squeeze(1), dim=-1)
        vis_to_ir = attn_v2i * v_ir

        # IR → VIS
        q_ir = self.query_proj_ir(ir_flat)
        k_vis = self.key_proj_vis(vis_flat)
        v_vis = self.value_proj_vis(vis_flat)
        attn_i2v = F.softmax(torch.bmm(q_ir.unsqueeze(1), k_vis.unsqueeze(2)).squeeze(1), dim=-1)
        ir_to_vis = attn_i2v * v_vis

        # 融合（可选 learnable 或平均）
        fused_flat = 0.5 * (vis_to_ir + ir_to_vis)
        fused_flat = self.final_proj(fused_flat)  # [B*T*N, patch_dim_out]
        fused = fused_flat.view(B, T, N, self.patch_h, self.patch_w, self.out_channels)

        # Unpatchify
        H_p, W_p = H // self.patch_h, W // self.patch_w
        fused = fused.view(B, T, H_p, W_p, self.patch_h, self.patch_w, self.out_channels)
        fused = fused.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, H, W, self.out_channels)
        fused = fused.permute(0, 1, 4, 2, 3)  # [B, T, C_out, H, W]

        # 下采样上一层特征并加残差
        B, T, C_prev, H_prev, W_prev = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C_prev, H_prev, W_prev)
        print( prev_fused_feat.shape)
        prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, T, self.out_channels, H, W)
        fused = fused + prev_fused_feat

        return fused

class CrossWeightedFusion1(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=(2, 2)):
        super(CrossWeightedFusion1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_h, self.patch_w = patch_size
        patch_dim = patch_size[0] * patch_size[1] * out_channels

        # VIS → IR
        self.query_proj_vis = nn.Linear(patch_dim, out_channels)
        self.key_proj_ir = nn.Linear(patch_dim, out_channels)
        self.value_proj_ir = nn.Linear(patch_dim, out_channels)
        # IR → VIS
        self.query_proj_ir = nn.Linear(patch_dim, out_channels)
        self.key_proj_vis = nn.Linear(patch_dim, out_channels)
        self.value_proj_vis = nn.Linear(patch_dim, out_channels)

        self.final_proj = nn.Linear(out_channels, patch_size[0] * patch_size[1] * out_channels)

        self.downsample_F = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def _patchify(self, x):
        B, T, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        x = x.permute(0, 1, 3, 4, 2).reshape(B, T, H // ph, ph, W // pw, pw, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, -1, ph * pw * C)
        return x

    def forward(self, ir_feat, vis_feat, prev_fused_feat):
        B, T, C, H, W = ir_feat.shape
        # ir_feat = torch.nan_to_num(ir_feat)
        # vis_feat = torch.nan_to_num(vis_feat)

        ir_patch = self._patchify(ir_feat)  # [B, T, N, D]
        vis_patch = self._patchify(vis_feat)

        B, T, N, D = vis_patch.shape
        ir_flat = ir_patch.view(B * T * N, D)
        vis_flat = vis_patch.view(B * T * N, D)

        # VIS → IR
        q_vis = self.query_proj_vis(vis_flat)
        k_ir = self.key_proj_ir(ir_flat)
        v_ir = self.value_proj_ir(ir_flat)
        attn_v2i = F.softmax(torch.bmm(q_vis.unsqueeze(1), k_ir.unsqueeze(2)).squeeze(1), dim=-1)
        vis_to_ir = attn_v2i * v_ir

        # IR → VIS
        q_ir = self.query_proj_ir(ir_flat)
        k_vis = self.key_proj_vis(vis_flat)
        v_vis = self.value_proj_vis(vis_flat)
        attn_i2v = F.softmax(torch.bmm(q_ir.unsqueeze(1), k_vis.unsqueeze(2)).squeeze(1), dim=-1)
        ir_to_vis = attn_i2v * v_vis

        # 融合（可选 learnable 或平均）
        fused_flat = 0.5 * (vis_to_ir + ir_to_vis)
        fused_flat = self.final_proj(fused_flat)  # [B*T*N, patch_dim_out]
        fused = fused_flat.view(B, T, N, self.patch_h, self.patch_w, self.out_channels)

        # Unpatchify
        H_p, W_p = H // self.patch_h, W // self.patch_w
        fused = fused.view(B, T, H_p, W_p, self.patch_h, self.patch_w, self.out_channels)
        fused = fused.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, H, W, self.out_channels)
        fused = fused.permute(0, 1, 4, 2, 3)  # [B, T, C_out, H, W]

        # 下采样上一层特征并加残差
        B, T, C_prev, H_prev, W_prev = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C_prev, H_prev, W_prev)
        prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, T, self.out_channels, H, W)
        fused = fused + prev_fused_feat

        return fused