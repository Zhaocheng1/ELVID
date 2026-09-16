

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
class IRNpyDataset_Test(Dataset):
    def __init__(self, json_path, transform=None, resize=(256, 256), normalize=True):
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        self.keys = list(self.data_dict.keys())
        self.transform = transform
        self.resize = resize
        self.normalize = normalize

        # 使用 ImageNet 三通道灰度伪RGB的标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item = self.data_dict[self.keys[idx]]
        npy_path = item["path"]
        label = item["label"]

        data = np.load(npy_path)  # shape: [T, H, W, C]
        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(f"Expected [T, H, W, 3], got {data.shape} from {npy_path}")

        t, h, w, c = data.shape
        frames = []

        for i in range(t):
            frame = data[i].astype(np.uint8)  # [H, W, 3]
            img = Image.fromarray(frame)      # 不再 .convert('L')
            tensor = TF.to_tensor(img)        # [3, H, W], ∈ [0, 1]
            frames.append(tensor)

        data_tensor = torch.stack(frames, dim=1)  # [3, T, H, W]
        print(data_tensor.shape)

        # Resize 所有帧
        if self.resize:
            c, t, h, w = data_tensor.shape
            data_tensor = data_tensor.view(c * t, h, w)
            data_tensor = F.interpolate(data_tensor.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False)
            data_tensor = data_tensor.squeeze(0).view(c, t, *self.resize)

        # 标准化
        if self.normalize:
            data_tensor = (data_tensor - self.mean.to(data_tensor.device)) / self.std.to(data_tensor.device)

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label


# class VISNpyDataset_Test(Dataset):
#     def __init__(self, json_path, transform=None, resize=(256, 256), normalize=True):
#         """
#         Args:
#             json_path (str): JSON路径
#             transform: 可选数据增强操作
#             resize: 输出图像尺寸 (H, W)
#             normalize: 是否执行标准化
#         """
#         with open(json_path, 'r') as f:
#             self.data_dict = json.load(f)
#
#         self.keys = list(self.data_dict.keys())
#         self.transform = transform
#         self.resize = resize
#         self.normalize = normalize
#
#         # 使用 ImageNet 标准 RGB 均值和方差
#         self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
#         self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
#
#     def __len__(self):
#         return len(self.keys)
#
#     def __getitem__(self, idx):
#         item = self.data_dict[self.keys[idx]]
#         npy_path = item["path"]
#         label = item["label"]
#
#         data = np.load(npy_path)  # shape: [T, H, W, C]
#         if data.ndim != 4 or data.shape[-1] != 3:
#             raise ValueError(f"Expected [T, H, W, 3], got {data.shape} from {npy_path}")
#
#         t, h, w, c = data.shape
#         frames = []
#
#         for i in range(t):
#             frame = data[i].astype(np.uint8)  # [H, W, 3]
#             img = Image.fromarray(frame)  # 保留RGB
#             tensor = TF.to_tensor(img)  # [3, H, W], ∈ [0,1]
#             frames.append(tensor)
#
#         data_tensor = torch.stack(frames, dim=1)  # → [3, T, H, W]
#
#         # Resize 所有帧
#         if self.resize is not None:
#             c, t, h, w = data_tensor.shape
#             data_tensor = data_tensor.view(c * t, h, w)
#             data_tensor = F.interpolate(data_tensor.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False)
#             data_tensor = data_tensor.squeeze(0).view(c, t, *self.resize)
#
#         # 标准化
#         if self.normalize:
#             data_tensor = (data_tensor - self.mean.to(data_tensor.device)) / self.std.to(data_tensor.device)
#
#         if self.transform:
#             data_tensor = self.transform(data_tensor)
#
#         return data_tensor, label
class VISNpyDataset_Test(Dataset):
    def __init__(self, json_path, transform=None, resize=(256, 256), normalize=True):
        """
        测试阶段可见光数据加载器，与训练数据保持一致的通道构造方式。
        """
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        self.keys = list(self.data_dict.keys())
        self.transform = transform
        self.resize = resize
        self.normalize = normalize

        # ImageNet 标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item = self.data_dict[self.keys[idx]]
        npy_path = item["path"]
        label = item["label"]

        data = np.load(npy_path)  # shape: [T, H, W, 3]
        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(f"Expected shape [T, H, W, 3], got {data.shape} from {npy_path}")

        t, h, w, c = data.shape
        frames = []
        g_frames = []
        b_frames = []

        for i in range(t):
            frame = data[i].astype(np.uint8)  # [H, W, 3]
            img = Image.fromarray(frame)  # RGB 图像
            rgb_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3, H, W]

            r_channel = rgb_tensor[0:1, :, :]        # [1, H, W]
            g_channel = rgb_tensor[1:2, :, :]
            b_channel = rgb_tensor[2:3, :, :]
            r_repeated = r_channel.repeat(3, 1, 1)   # [3, H, W]
            frames.append(r_repeated)
            g_repeated = g_channel.repeat(3, 1, 1)
            g_frames.append(g_repeated)
            b_repeated = b_channel.repeat(3, 1, 1)
            b_frames.append(b_repeated)

        data_tensor = torch.stack(frames, dim=1)  # [3, T, H, W]
        data_tensor_g = torch.stack(g_frames, dim=1)  # [3, T, H, W]
        data_tensor_b = torch.stack(b_frames, dim=1)  # [3, T, H, W]

        # Resize 所有帧
        if self.resize:
            c, t, h, w = data_tensor.shape
            data_tensor = data_tensor.view(c * t, h, w)
            data_tensor = F.interpolate(data_tensor.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False)
            data_tensor = data_tensor.squeeze(0).view(c, t, *self.resize)

            c1, t1, h1, w1 = data_tensor_g.shape
            data_tensor_g = data_tensor_g.view(c1 * t1, h1, w1)
            data_tensor_g = F.interpolate(data_tensor_g.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False)
            data_tensor_g = data_tensor_g.squeeze(0).view(c1, t1, *self.resize)

            c2, t2, h2, w2 = data_tensor_b.shape
            data_tensor_b = data_tensor_b.view(c2 * t2, h2, w2)
            data_tensor_b = F.interpolate(data_tensor_b.unsqueeze(0), size=self.resize, mode='bilinear',align_corners=False)
            data_tensor_b = data_tensor_b.squeeze(0).view(c2, t2, *self.resize)

        # 标准化
        if self.normalize:
            data_tensor = (data_tensor - self.mean.to(data_tensor.device)) / self.std.to(data_tensor.device)
            data_tensor_g = (data_tensor_g - self.mean.to(data_tensor_g.device)) / self.std.to(data_tensor_g.device)
            data_tensor_b = (data_tensor_b - self.mean.to(data_tensor_b.device)) / self.std.to(data_tensor_b.device)

        if self.transform:
            data_tensor = self.transform(data_tensor)
            data_tensor_g = self.transform(data_tensor_g)
            data_tensor_b = self.transform(data_tensor_b)

        return data_tensor,data_tensor_g,data_tensor_b, label

# json_path = "/home/ubuntu/zhaocheng/infrared_visible_video_dataset/m3svd_videos/json/test_ir_m3svd_only.json"
# dataset = IRNpyDataset_Test(json_path)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
#
# for batch_idx, (Y,label) in enumerate(dataloader):
# # for batch_idx, (Y, Y_g,Y_b,label) in enumerate(dataloader):
#     print(f"Batch {batch_idx}:")
#     print("  Data shape:", Y.shape)     # shape: [B, C=3, T, H, W]
#     print("  Labels:", label)
#
#     # 获取第一个视频的第一帧（灰度图重复3通道）
#     first_video = Y[3]             # shape: [3, T, H, W]
#     first_frame = first_video[:, 0, :, :]  # shape: [3, H, W]
#     first_channel = first_frame[0,:,:]
#
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(first_frame.device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(first_frame.device)
#
#     # 反归一化回 [0, 1]
#     # first_frame = first_frame * 0.5 + 0.5
#     first_frame = first_frame * std + mean
#     first_frame = first_frame.clamp(0, 1)
#
#     # 转为 PIL 图像
#     img_np = to_pil_image(first_frame)
#
#     # 可视化
#     # plt.imshow(first_channel.cpu(), cmap='gray')  # 用灰度图显示数值强度
#     plt.imshow(img_np, cmap='gray')  # 灰度图显示
#     plt.title("First Frame of First Video (IR) - Denormalized")
#     plt.axis('off')
#     plt.show()
#     break