import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0, alpha=0.25):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.full_like(logits, self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)

        pt = torch.sum(true_dist * prob, dim=1)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma

        loss = -torch.sum(true_dist * log_prob, dim=1)
        loss = focal_weight * loss

        return loss.mean()

class FocalLossWithLabelSmoothingHardMining(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0, alpha=0.25, top_k_percent=1.0):
        super(FocalLossWithLabelSmoothingHardMining, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.top_k_percent = top_k_percent  # 挖掘最难的 top_k 百分比样本

    def forward(self, logits, target):
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.full_like(logits, self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)

        pt = torch.sum(true_dist * prob, dim=1)  # shape: (B,)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma  # shape: (B,)

        loss = -torch.sum(true_dist * log_prob, dim=1)  # shape: (B,)
        loss = focal_weight * loss  # shape: (B,)

        # ✅ Top-K 挖掘困难样本
        if self.top_k_percent < 1.0:
            k = int(self.top_k_percent * loss.size(0))
            if k < 1:  # 防止取空
                k = 1
            topk_loss, _ = torch.topk(loss, k=k, largest=True)
            return topk_loss.mean()
        else:
            return loss.mean()

class FocalLossWithHardMiningAndConfidenceFilter(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0, alpha=0.25, mining_ratio=0.5, confidence_threshold=0.0):
        """
        :param smoothing: label smoothing epsilon
        :param gamma: focal loss gamma
        :param alpha: focal loss alpha
        :param mining_ratio: ratio of hardest samples to keep (0 ~ 1)
        :param confidence_threshold: skip samples with max prob below this threshold (optional)
        """
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.mining_ratio = mining_ratio
        self.confidence_threshold = confidence_threshold

    def forward(self, logits, target):
        num_classes = logits.size(1)
        with torch.no_grad():
            # 生成 label smoothing 分布
            true_dist = torch.full_like(logits, self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)

        # focal loss 权重项
        pt = torch.sum(true_dist * prob, dim=1)  # shape: (B,)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma

        # 原始损失
        loss = -torch.sum(true_dist * log_prob, dim=1)  # shape: (B,)
        loss = focal_weight * loss

        # step 1: 置信度过滤
        if self.confidence_threshold > 0.0:
            max_prob, _ = prob.max(dim=1)
            mask = max_prob >= self.confidence_threshold
            loss = loss[mask]
            if loss.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # step 2: Top-k 难样本挖掘
        if 0.0 < self.mining_ratio < 1.0:
            k = max(1, int(loss.size(0) * self.mining_ratio))
            topk_loss, _ = torch.topk(loss, k)
            return topk_loss.mean()
        else:
            return loss.mean()

def get_criterion(modelname='resnet18', opt=None):
    if modelname == "resnet18" or modelname == "abla_tem" or modelname == "abla_spat" or modelname == "abla_backbone" or modelname == "resnet18_based_0"or modelname == "resnet18__avgear"or modelname == "resnet18__weiduyasuo":
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLossWithLabelSmoothing(smoothing=0.1, gamma=2.0, alpha=0.25)
        criterion = FocalLossWithLabelSmoothingHardMining(smoothing=0.1, gamma=2.0, alpha=0.25,top_k_percent=0.7)
        # criterion = FocalLossWithHardMiningAndConfidenceFilter(smoothing=0.1, gamma=2.0, alpha=0.25, mining_ratio=0.5, confidence_threshold=0.3 )     # 过滤低于 0.3 的置信样本（可选）)(resnet18no need this loss)
    elif modelname == '8view':
        criterion = nn.CrossEntropyLoss()
    elif modelname ==  'resnet18_avgear':
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLossWithLabelSmoothing(smoothing=0.1, gamma=2.0, alpha=0.25)
        # criterion = FocalLossWithLabelSmoothingHardMining(smoothing=0.1, gamma=2.0, alpha=0.25,top_k_percent=0.7)
        criterion = FocalLossWithHardMiningAndConfidenceFilter(smoothing=0.1, gamma=2.0, alpha=0.25, mining_ratio=0.5, confidence_threshold=0.3 )     # 过滤低于 0.3 的置信样本（可选）)
    elif modelname == 'resnet18_weiduyasuo':
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLossWithLabelSmoothing(smoothing=0.1, gamma=2.0, alpha=0.25)
        criterion = FocalLossWithLabelSmoothingHardMining(smoothing=0.1, gamma=2.0, alpha=0.25,top_k_percent=0.7)
        # criterion = FocalLossWithHardMiningAndConfidenceFilter(smoothing=0.1, gamma=2.0, alpha=0.25, mining_ratio=0.5, confidence_threshold=0.3 )     # 过滤低于 0.3 的置信样本（可选）)

    else:
        criterion = None
    return criterion