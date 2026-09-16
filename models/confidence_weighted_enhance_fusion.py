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

    def enhance_contrast(self, x, alpha=4, beta=0.5):
        return 1 / (1 + torch.exp(-alpha * (x - beta)))

    def forward(self, ir_feat, vis_feat, prev_fused_feat=None):
        B, T, C, H, W = ir_feat.shape
        ir = ir_feat.view(B * T, C, H, W)
        vis = vis_feat.view(B * T, C, H, W)

        # 对可见光增强
        vis_contrast = vis * self.enhance_contrast(vis)
        vis_detail = vis_contrast + 0.1 * self.sobel(vis_contrast)

        diff = torch.abs(ir -   vis_detail)
        attn_map = self.attn(diff)
        s1 = ir * attn_map +   vis_detail * (1 - attn_map)

        edge_ir_lap = (self.edge3(ir) + self.edge5(ir)) / 2
        edge_vis_lap = (self.edge3(vis_detail) + self.edge5(vis_detail)) / 2
        edge_ir_sobel = self.sobel(ir)
        edge_vis_sobel = self.sobel(vis_detail)

        edge_ir = (edge_ir_lap + edge_ir_sobel) / 2
        edge_vis = (edge_vis_lap + edge_vis_sobel) / 2

        edge_mask = torch.sigmoid(edge_ir - edge_vis)
        ir_weighted = ir * edge_mask
        vis_weighted = vis_detail * (1 - edge_mask)

        fused = ir_weighted + vis_weighted + s1
        fused = fused.view(B, T, C, H, W)

        # 融合上一层特征（下采样）
        # 注意 prev_fused_feat 输入是 [B, T, C_prev, H_prev, W_prev]
        B, T, C1, H1, W1 = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C1, H1, W1)
        # prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']

        fused = fused + prev_fused_feat  # 残差连接（可选）

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

    def enhance_contrast(self, x, alpha=4, beta=0.5):
        return 1 / (1 + torch.exp(-alpha * (x - beta)))

    def forward(self, ir_feat, vis_feat, prev_fused_feat=None):
        B, T, C, H, W = ir_feat.shape
        ir = ir_feat.view(B * T, C, H, W)
        vis = vis_feat.view(B * T, C, H, W)

        # 对可见光增强
        vis_contrast = vis * self.enhance_contrast(vis)
        vis_detail = vis_contrast + 0.1 * self.sobel(vis_contrast)

        diff = torch.abs(ir - vis_detail)
        attn_map = self.attn(diff)
        s1 = ir * attn_map + vis_detail * (1 - attn_map)

        edge_ir_lap = (self.edge3(ir) + self.edge5(ir)) / 2
        edge_vis_lap = (self.edge3(vis_detail) + self.edge5(vis_detail)) / 2
        edge_ir_sobel = self.sobel(ir)
        edge_vis_sobel = self.sobel(vis_detail)

        edge_ir = (edge_ir_lap + edge_ir_sobel) / 2
        edge_vis = (edge_vis_lap + edge_vis_sobel) / 2

        edge_mask = torch.sigmoid(edge_ir - edge_vis)
        ir_weighted = ir * edge_mask
        vis_weighted = vis_detail * (1 - edge_mask)

        fused = ir_weighted + vis_weighted + s1
        fused = fused.view(B, T, C, H, W)

        # 融合上一层特征（下采样）
        # 注意 prev_fused_feat 输入是 [B, T, C_prev, H_prev, W_prev]
        B, T, C1, H1, W1 = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C1, H1, W1)
        prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']

        fused = fused + prev_fused_feat  # 残差连接（可选）

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

    def enhance_contrast(self, x, alpha=4, beta=0.5):
        return 1 / (1 + torch.exp(-alpha * (x - beta)))

    def forward(self, ir_feat, vis_feat, prev_fused_feat=None):
        B, T, C, H, W = ir_feat.shape
        ir = ir_feat.view(B * T, C, H, W)
        vis = vis_feat.view(B * T, C, H, W)

        # 对可见光增强
        vis_contrast = vis * self.enhance_contrast(vis)
        vis_detail = vis_contrast + 0.1 * self.sobel(vis_contrast)

        diff = torch.abs(ir -  vis_detail)
        attn_map = self.attn(diff)
        s1 = ir * attn_map +  vis_detail * (1 - attn_map)

        edge_ir_lap = (self.edge3(ir) + self.edge5(ir)) / 2
        edge_vis_lap = (self.edge3( vis_detail) + self.edge5( vis_detail)) / 2
        edge_ir_sobel = self.sobel(ir)
        edge_vis_sobel = self.sobel( vis_detail)

        edge_ir = (edge_ir_lap + edge_ir_sobel) / 2
        edge_vis = (edge_vis_lap + edge_vis_sobel) / 2

        edge_mask = torch.sigmoid(edge_ir - edge_vis)
        ir_weighted = ir * edge_mask
        vis_weighted =  vis_detail * (1 - edge_mask)

        fused = ir_weighted + vis_weighted + s1
        fused = fused.view(B, T, C, H, W)

        # 融合上一层特征（下采样）
        # 注意 prev_fused_feat 输入是 [B, T, C_prev, H_prev, W_prev]
        # 注意 prev_fused_feat 输入是 [B, T, C_prev, H_prev, W_prev]
        B, T, C1, H1, W1 = prev_fused_feat.shape
        prev_fused_feat = prev_fused_feat.view(B * T, C1, H1, W1)
        prev_fused_feat = self.downsample_F(prev_fused_feat)
        prev_fused_feat = prev_fused_feat.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)  # → [B, T, target_C, H', W']

        fused = fused + prev_fused_feat  # 残差连接（可选）

        return fused

