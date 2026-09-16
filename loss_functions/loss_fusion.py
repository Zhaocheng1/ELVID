import pytorch_ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
import piq

def ssim_loss_video(fused: torch.Tensor, reference: torch.Tensor, eps=1e-6):
    """
    更稳健的 SSIM 损失函数（视频版本）
    """
    bs1 = fused.shape[0] // 16
    fused = fused.view(bs1, 16, 3, 256, 256)
    reference = reference.view(bs1, 16, 3, 256, 256)

    B, T, C, H, W = fused.shape
    total_ssim = 0.0
    valid_count = 0

    for t in range(T):
        fused_clip = fused[:, t].clamp(0, 1).float()
        ref_clip = reference[:, t].clamp(0, 1).float()

        try:
            ssim_val = piq.ssim(fused_clip, ref_clip, data_range=1.)
            if torch.isnan(ssim_val):
                continue
            total_ssim += (1 - ssim_val)
            valid_count += 1
        except Exception as e:
            print(f"SSIM calculation failed at t={t}: {e}")
            continue

    if valid_count == 0:
        return torch.tensor(0.0, device=fused.device)
    return total_ssim / valid_count

def safe_ssim(x, y, data_range=1.0):
    # x, y: [B, C, H, W]，值需在 [0, data_range] 且不能包含 NaN
    assert torch.isfinite(x).all() and torch.isfinite(y).all(), "Input contains NaN or Inf"
    x = torch.clamp(x, 0, data_range)
    y = torch.clamp(y, 0, data_range)
    return piq.ssim(x, y, data_range=data_range)

# 2. 如果你的图像不是 [0,1] 范围，可以归一化再送入：
def normalize_for_ssim(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

def consistency_loss_video(fused: torch.Tensor,
                           reference: torch.Tensor,
                           alpha=0.85,
                           beta=0.15,
                           use_mse=False,
                           normalize=True):
    """
    视频一致性损失（像素 + 结构）

    参数:
        fused:     [B, T, C, H, W]，融合输出序列
        reference: [B, T, C, H, W]，参考图像序列（如红外或可见光）
        alpha: 像素损失权重（L1/MSE）
        beta: 结构损失权重（SSIM）
        use_mse: 是否使用 MSE 替代 L1（默认=False）
        normalize: 是否归一化到 [0, 1] 再计算 SSIM

    返回:
        consistency_loss: float 标量，越小越好
    """
    bs1 = fused.shape[0] // 16
    fused = fused.view(bs1, 16, 3, 256, 256)

    reference = reference.view(bs1, 16, 3, 256, 256)
    assert fused.shape == reference.shape, "输入和参考形状必须一致"
    B, T, C, H, W = fused.shape

    pixel_loss_total = 0.0
    ssim_loss_total = 0.0

    for t in range(T):
        frame_fused = fused[:, t]
        frame_ref = reference[:, t]

        # 像素级损失
        if use_mse:
            pixel_loss = F.mse_loss(frame_fused, frame_ref)
        else:
            pixel_loss = F.l1_loss(frame_fused, frame_ref)
        pixel_loss_total += pixel_loss

        # 结构损失（SSIM）
        if normalize:
            frame_fused = (frame_fused - frame_fused.amin(dim=(1, 2, 3), keepdim=True)) / \
                          (frame_fused.amax(dim=(1, 2, 3), keepdim=True) - frame_fused.amin(dim=(1, 2, 3),
                                                                                            keepdim=True) + 1e-8)
            frame_ref = (frame_ref - frame_ref.amin(dim=(1, 2, 3), keepdim=True)) / \
                        (frame_ref.amax(dim=(1, 2, 3), keepdim=True) - frame_ref.amin(dim=(1, 2, 3),
                                                                                      keepdim=True) + 1e-8)

            # 保障无 NaN/Inf
        if torch.isnan(frame_fused).any() or torch.isnan(frame_ref).any():
            continue

        ssim_score = safe_ssim(frame_fused, frame_ref, data_range=1.0)
        ssim_loss = 1 - ssim_score
        ssim_loss_total += ssim_loss

        # 平均每帧损失
    avg_pixel_loss = pixel_loss_total / T
    avg_ssim_loss = ssim_loss_total / T

    return alpha * avg_pixel_loss + beta * avg_ssim_loss
    # return  avg_pixel_loss

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, use_l1=True, weight_frame_diff=1.0, weight_grad=1.0):
        super(TemporalConsistencyLoss, self).__init__()
        self.loss_fn = nn.L1Loss() if use_l1 else nn.MSELoss()
        self.weight_frame_diff = weight_frame_diff
        self.weight_grad = weight_grad

    def spatial_gradient(self, x: torch.Tensor):
        """
        计算图像的 Sobel 近似梯度（横向和纵向）
        输入: x: [B, C, H, W]
        输出: grad_x, grad_y
        """
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad_x, grad_y

    def forward(self, fused: torch.Tensor, ref: torch.Tensor = None):
        """
        Args:
            fused: [B, T, C, H, W] — 融合输出
            ref:   [B, T, C, H, W] — 可选：参考输入序列（如IR或VIS），若提供则启用帧差一致性项
        Returns:
            Scalar temporal loss
        """
        bs1 = fused.shape[0] // 16
        fused = fused.view(bs1, 16, 3, 256, 256)

        ref = ref.view(bs1, 16, 3, 256, 256)
        B, T, C, H, W = fused.shape
        loss = 0.0
        for t in range(1, T):
            # --- 1. 时间帧差一致性（参考序列对齐）
            if ref is not None:
                diff_fused = fused[:, t] - fused[:, t - 1]
                diff_ref = ref[:, t] - ref[:, t - 1]
                loss += self.weight_frame_diff * self.loss_fn(diff_fused, diff_ref)

            # --- 2. 空间梯度时序一致性
            gx1, gy1 = self.spatial_gradient(fused[:, t])
            gx0, gy0 = self.spatial_gradient(fused[:, t - 1])
            loss += self.weight_grad * (F.l1_loss(gx1, gx0) + F.l1_loss(gy1, gy0))

        return loss / (T - 1)

class ContentLossVideo(nn.Module):
    def __init__(self):
        super(ContentLossVideo, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fused: torch.Tensor, ref: torch.Tensor):
        """
        fused: [B, T, C, H, W]
        ref:   [B, T, C, H, W]（红外或可见光帧序列）
        """
        bs1 = fused.shape[0] // 16
        fused = fused.view(bs1, 16, 3, 256, 256)

        ref = ref.view(bs1, 16, 3, 256, 256)

        B, T, C, H, W = fused.shape
        loss = 0.0
        for t in range(T):
            loss += self.l1(fused[:, t], ref[:, t])
        return loss / T

class GradientLossVideo(nn.Module):
    def __init__(self):
        super(GradientLossVideo, self).__init__()
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', kernel)

    def gradient(self, x):
        """
        x: [B, C, H, W] — 逐帧单张图像
        输出: 拉普拉斯梯度图
        """
        B, C, H, W = x.shape
        kernel = self.laplacian_kernel.expand(C, 1, 3, 3).to(x.device)
        grad = F.conv2d(x, kernel, padding=1, groups=C)
        return grad

    def forward(self, fused: torch.Tensor, ref: torch.Tensor):
        """
        fused: [B, T, C, H, W]
        ref:   [B, T, C, H, W]
        """
        bs1 = fused.shape[0] // 16
        fused = fused.view(bs1, 16, 3, 256, 256)

        ref = ref.view(bs1, 16, 3, 256, 256)
        B, T, C, H, W = fused.shape
        loss = 0.0
        for t in range(T):
            grad_fused = self.gradient(fused[:, t])
            grad_ref = self.gradient(ref[:, t])
            loss += F.l1_loss(grad_fused, grad_ref)
        return loss / T

def load_balancing_loss(gate_probs):
    """
    gate_probs: Tensor [B*T, num_experts]
    """
    expert_usage = gate_probs.sum(dim=0)  # [num_experts]
    expert_frac = expert_usage / gate_probs.sum()
    loss = (expert_frac * torch.log(expert_frac + 1e-8)).sum()  # Entropy-based
    return -loss  # 越均匀越大，取负使之成为损失

def sparsity_loss(gate_probs, target_sparsity=0.1):
    """
    gate_probs: [B*T, num_experts]
    """
    avg_activation = gate_probs.mean(dim=1)  # 每个样本平均激活
    loss = torch.abs(avg_activation - target_sparsity).mean()
    return loss


def contrast_map(x):
    # x: [B, 3, H, W]
    B, C, H, W = x.shape
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)

    gx = F.conv2d(x, sobel_x.repeat(C, 1, 1, 1), padding=1, groups=C)
    gy = F.conv2d(x, sobel_y.repeat(C, 1, 1, 1), padding=1, groups=C)

    contrast = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)  # [B, 3, H, W]
    return contrast


def contrast_consistency_loss(fused_rgb, vis_rgb):
    """
    :param fused_rgb: [B, 3, H, W] - 融合图像
    :param vis_rgb:   [B, 3, H, W] - 原始可见光图像
    :return: 对比度损失（scalar）
    """
    contrast_fused = contrast_map(fused_rgb)
    contrast_vis = contrast_map(vis_rgb)

    return F.l1_loss(contrast_fused, contrast_vis)