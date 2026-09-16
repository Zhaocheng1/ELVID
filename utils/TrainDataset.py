import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


class TrainIRNpyDataset(Dataset):
    def __init__(self, json_path, transform=None, resize=(256,256), normalize=True):
        """
        加载红外数据（保持伪彩 RGB 输入，不转灰度）。
        """
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        self.keys = list(self.data_dict.keys())
        self.transform = transform
        self.resize = resize
        self.normalize = normalize

        # 三通道统一归一化：∈ [0,1] → [-1,1]
        # self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        # self.std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_dict[key]
        npy_path = item["data"]["ir"]
        label = item["label"]

        data = np.load(npy_path)  # [T, H, W, 3]
        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(f"Expected shape [T, H, W, 3], got {data.shape} from {npy_path}")

        T, H, W, C = data.shape
        frames = []

        for t in range(T):
            frame = data[t].astype(np.uint8)  # [H, W, 3]
            img = Image.fromarray(frame)  # 保持 RGB
            rgb_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3, H, W]
            frames.append(rgb_tensor)

        data_tensor = torch.stack(frames, dim=1)  # [3, T, H, W]

        # Resize 所有帧
        if self.resize:
            c, t, h, w = data_tensor.shape
            data_tensor = data_tensor.view(c * t, h, w)
            data_tensor = F.interpolate(data_tensor.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False)
            data_tensor = data_tensor.squeeze(0).view(c, t, *self.resize)

        # 标准化到 [-1, 1]
        if self.normalize:
            mean = self.mean.view(3, 1, 1, 1).to(data_tensor.device)
            std = self.std.view(3, 1, 1, 1).to(data_tensor.device)
            data_tensor = (data_tensor - mean) / std

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label

# class TrainVISNpyDataset(Dataset):
#     def __init__(self, json_path, transform=None, resize=(256, 256), normalize=True):
#         with open(json_path, 'r') as f:
#             self.data_dict = json.load(f)
#
#         self.keys = list(self.data_dict.keys())
#         self.transform = transform
#         self.resize = resize
#         self.normalize = normalize
#
#         # 适用于 RGB 图像的归一化参数
#         self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     def __len__(self):
#         return len(self.keys)
#
#     def __getitem__(self, idx):
#         key = self.keys[idx]
#         item = self.data_dict[key]
#         npy_path = item["data"]["vis"]
#         label = item["label"]
#
#         data = np.load(npy_path)  # shape: [T, H, W, 3]
#         if data.ndim != 4 or data.shape[-1] != 3:
#             raise ValueError(f"Expected shape [T, H, W, 3], got {data.shape} from {npy_path}")
#
#         T, H, W, C = data.shape
#         frames = []
#
#         for t in range(T):
#             frame = data[t].astype(np.uint8)  # [H, W, 3]
#             img = Image.fromarray(frame)  # 保持原始 RGB 色彩
#             rgb_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3, H, W], [0,1]
#             frames.append(rgb_tensor)
#
#         data_tensor = torch.stack(frames, dim=1)  # [3, T, H, W]
#
#         # Resize 所有帧
#         if self.resize:
#             c, t, h, w = data_tensor.shape
#             data_tensor = data_tensor.view(c * t, h, w)
#             data_tensor = F.interpolate(data_tensor.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False)
#             data_tensor = data_tensor.squeeze(0).view(c, t, *self.resize)
#
#         # 归一化 RGB 图像，范围 [0, 1] → [-1, 1]
#         if self.normalize:
#             mean = self.mean.view(3, 1, 1, 1).to(data_tensor.device)
#             std = self.std.view(3, 1, 1, 1).to(data_tensor.device)
#             data_tensor = (data_tensor - mean) / std
#
#         if self.transform:
#             data_tensor = self.transform(data_tensor)
#
#         return data_tensor, label
class TrainVISNpyDataset(Dataset):
    def __init__(self, json_path, transform=None, resize=(256,256), normalize=True):
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        self.keys = list(self.data_dict.keys())
        self.transform = transform
        self.resize = resize
        self.normalize = normalize

        # 适用于灰度伪RGB图像的归一化参数（即 R 通道复制）
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_dict[key]
        npy_path = item["data"]["vis"]
        label = item["label"]

        data = np.load(npy_path)  # shape: [T, H, W, 3]
        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(f"Expected shape [T, H, W, 3], got {data.shape} from {npy_path}")

        T, H, W, C = data.shape
        frames = []

        for t in range(T):
            frame = data[t].astype(np.uint8)  # [H, W, 3]
            img = Image.fromarray(frame)  # RGB 图像
            rgb_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3, H, W]

            r_channel = rgb_tensor[0:1, :, :]       # [1, H, W]
            r_repeated = r_channel.repeat(3, 1, 1)  # [3, H, W] 伪 RGB
            frames.append(r_repeated)

        data_tensor = torch.stack(frames, dim=1)  # [3, T, H, W]

        # Resize 所有帧
        if self.resize:
            c, t, h, w = data_tensor.shape
            data_tensor = data_tensor.view(c * t, h, w)
            data_tensor = F.interpolate(data_tensor.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False)
            data_tensor = data_tensor.squeeze(0).view(c, t, *self.resize)

        # 归一化（使用灰度图的均值和方差）
        if self.normalize:
            mean = self.mean.view(3, 1, 1, 1).to(data_tensor.device)
            std = self.std.view(3, 1, 1, 1).to(data_tensor.device)
            data_tensor = (data_tensor - mean) / std

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label


import torch
from torch.utils.data import DataLoader

train_dataset = TrainVISNpyDataset("/home/ubuntu/zhaocheng/infrared_visible_video_dataset/json_files/vis_dataset.json")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
for batch_idx, (Y, label) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print("  Data shape:", Y.shape)     # shape: [B, C=3, T, H, W]
    print("  Labels:", label)

    # 获取第一个视频的第一帧（灰度图重复3通道）
    first_video = Y[0]             # shape: [3, T, H, W]
    first_frame = first_video[:, 0, :, :]  # shape: [3, H, W]
    first_channel = first_frame[0,:,:]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(first_frame.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(first_frame.device)

    # 反归一化回 [0, 1]
    # first_frame = first_frame * 0.5 + 0.5
    first_frame = first_frame * std + mean
    first_frame = first_frame.clamp(0, 1)

    # 转为 PIL 图像
    img_np = to_pil_image(first_frame)

    # 可视化
    plt.imshow(first_channel.cpu(), cmap='gray')  # 用灰度图显示数值强度
    # plt.imshow(img_np, cmap='gray')  # 灰度图显示
    plt.title("First Frame of First Video (IR) - Denormalized")
    plt.axis('off')
    plt.show()
    break