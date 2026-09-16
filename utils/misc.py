import torch.nn as nn
import torch
import os

def weights_init_normal(m):
    """
    初始化卷积层和线性层的权重为标准正态分布，bias 为 0
    """
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def save_checkpoint(model, path, epoch=None):
    """
    保存模型检查点
    :param model: nn.Module 或 DataParallel
    :param path: 保存路径
    :param epoch: 当前 epoch（可选）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = model.state_dict()
    checkpoint = {'model_state_dict': state_dict}
    if epoch is not None:
        checkpoint['epoch'] = epoch
    torch.save(checkpoint, path)