import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utils.TrainDataset import TrainIRNpyDataset, TrainVISNpyDataset
from utils.TestDataset import VISNpyDataset_Test, IRNpyDataset_Test
from models.resnet18_based_fusion4_moe_multi_token_expert import ModifiedResNet18_fusion
from loss_functions.loss_fusion import ssim_loss_video,TemporalConsistencyLoss,GradientLossVideo, ContentLossVideo, consistency_loss_video,contrast_consistency_loss
from utils.logger import Logger
from utils.misc import weights_init_normal, save_checkpoint
import logging
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

def save_checkpoint(model, optimizer, epoch, opt, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建目录
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'opt': opt
    }, save_path)
def to_gray01(x):
    # x: [N,C,H,W] in [any range]
    if x.size(1) == 3:
        y = 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]
    else:
        y = x[:,0]
    # 归一到大致 [0,1]，不依赖真实范围（只为取分位数/边缘）
    y = (y - y.amin(dim=(1,2), keepdim=True)) / (y.amax(dim=(1,2), keepdim=True) - y.amin(dim=(1,2), keepdim=True) + 1e-6)
    return y  # [N,H,W]

def grad_mag(x):  # x: [N,1,H,W] 或 [N,C,H,W]
    # 简单前向差分 + padding
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    dx = torch.nn.functional.pad(dx, (0,0,1,0))
    dy = torch.nn.functional.pad(dy, (1,0,0,0))
    g = torch.sqrt(dx*dx + dy*dy + 1e-6)
    # 若是多通道，取均值到单通道
    if g.size(1) > 1:
        g = g.mean(dim=1, keepdim=True)
    return g  # [N,1,H,W]


class FusionTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.logger = Logger(opt.save_dir)  # 初始化 logger

        self.model = ModifiedResNet18_fusion(32).to(self.device)
        # self.model.apply(weights_init_normal)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)

        # 更合适的多阶段学习率衰减（例如在第 30 和 45 轮分别衰减）
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[20, 45],  # 更匹配 50 轮训练的计划
            gamma=0.1
        )

        # self.criterion_ssim1 = ssim_loss_video()
        self.criterion_temporal = TemporalConsistencyLoss()
        self.content = ContentLossVideo()
        self.grad = GradientLossVideo()


        # self.consistency_video = consistency_loss_video(alpha=0.7,beta=0.3)


        self.train_ir_dataset = TrainIRNpyDataset(opt.train_ir_json)
        self.train_vis_dataset = TrainVISNpyDataset(opt.train_vis_json)

        self.train_loader_ir = DataLoader(self.train_ir_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
        self.train_loader_vis = DataLoader(self.train_vis_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)

        log_path = os.path.join(opt.save_dir, 'log.txt')
        self.logger = Logger(fpath=log_path)
        self.logger.set_names(['epoch_loss'])

    def train(self):
        print("Start training...")
        for epoch in range(1, self.opt.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            self.model.gumbel_tau = max(0.5, 1.0 * (0.95 ** epoch))  # 每轮衰减 2%

            pbar = zip(tqdm(self.train_loader_ir), self.train_loader_vis)
            for (ir, _), (vis, _) in pbar:
                ir, vis = ir.to(self.device), vis.to(self.device)  # [B, T, C, H, W]
                self.optimizer.zero_grad()
                # print(ir.shape)
                ir = ir.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)
                ir = ir.reshape(-1, *ir.shape[-3:])  # (B*T,C,H,W)
                vis = vis.permute(0, 2, 1, 3, 4)
                vis = vis.reshape(-1, *vis.shape[-3:])
                # print("ir.shape:",ir.shape)
                # print("vis.shape", vis.shape)

                fusion, ir_fusion, vis_fusion ,lb_loss , sparse_loss_4 = self.model(ir, vis)
                # 1) IR 亮度图 & 掩膜
                ir_y = to_gray01(ir)  # [N,H,W]
                # 顶部 10% 作为“热显著”区域
                q = torch.quantile(ir_y.view(ir_y.size(0), -1), 0.90, dim=1, keepdim=True).view(-1, 1, 1)
                hot = (ir_y >= q).float()  # [N,H,W]
                # 边缘权重（归一）
                edge = grad_mag(ir_y.unsqueeze(1))  # [N,1,H,W]
                edge = edge / (edge.mean(dim=(2, 3), keepdim=True) + 1e-6)
                edge = edge.squeeze(1)  # [N,H,W]

                # 合成最终权重 w（可调系数）
                a_hot, b_edge = 2.0, 1.0  # 热区和边缘的放大系数
                w = (1.0 + a_hot * hot + b_edge * edge).detach()  # [N,H,W]
                w = w.unsqueeze(1)  # [N,1,H,W] 便于广播

                # === 新增：VIS 的细节保护权重 ===
                vis_y = to_gray01(vis)  # [N,H,W]
                edge_vis = grad_mag(vis_y.unsqueeze(1)).squeeze(1)  # [N,H,W]
                edge_vis = edge_vis / (edge_vis.mean(dim=(1, 2), keepdim=True) + 1e-6)
                c_edge_vis = 1.5  # 可见光边缘的强化系数（可在 1.0~2.5 间微调）
                w_vis = (1.0 + c_edge_vis * edge_vis).unsqueeze(1)  # [N,1,H,W]

                # 2) 用加权 IR 做“内容/梯度”损失（VIS 部分保持原来不加权）
                # 内容 L1：自己写一版加权，替代 self.content 的 IR 分量
                loss_content_ir = (w * (fusion - ir).abs()).mean()
                loss_content_vis = self.content(fusion, vis)  # 保留你原来的 VIS 内容项
                # loss_content_vis = (w_vis * (fusion - vis).abs()).mean()

                # 梯度 L1：自己写一版加权替代 IR 分量（VIS 保持原来）
                g_f = grad_mag(fusion)
                g_ir = grad_mag(ir)
                g_vi = grad_mag(vis)
                loss_grad_ir = (w * (g_f - g_ir).abs()).mean()
                # loss_grad_vis = (g_f - g_vi).abs().mean()
                loss_grad_vis = (w_vis * (g_f - g_vi).abs()).mean()

                # 权重（建议起步值，比你原来更保边缘）
                w_content_ir, w_content_vis = 1.5,1 #3, 3
                w_grad_ir, w_grad_vis = 8, 3# 6,6.5

                loss_content = w_content_ir * loss_content_ir + w_content_vis * loss_content_vis
                loss_grad = w_grad_ir * loss_grad_ir + w_grad_vis * loss_grad_vis
                # print("-------------------------------")
                # print(' loss_content_ir', loss_content_ir, ' loss_content_vis', loss_content_vis,' loss_grad_ir',  loss_grad_ir, ' loss_grad_vis',  loss_grad_vis,  )



                loss_temporal_ir = self.criterion_temporal(fusion, ir)
                loss_temporal_vis = self.criterion_temporal(fusion, vis)
                loss_temporal = loss_temporal_ir + loss_temporal_vis

                loss_ssim_ir = ssim_loss_video(fusion,  ir)
                loss_ssim_vis = ssim_loss_video(fusion, vis)
                loss_ssim = loss_ssim_ir + loss_ssim_vis
                # print("-------------------------------")
                # print(' loss_ssim',   loss_ssim, ' loss_ssim_ir',   loss_ssim_ir, '  loss_ssim_vis',  loss_ssim_vis, )

                loss_vis_contrast = contrast_consistency_loss(fusion,vis)
                # loss_ir_contrast = contrast_consistency_loss(fusion, ir)
                # print("-------------------------------")
                # print(' loss_vis_contrast', loss_vis_contrast )

                loss_ir_consistency = consistency_loss_video(ir_fusion, ir,alpha=0.7,beta=0.3)

                loss_vis_consistency = consistency_loss_video(vis_fusion, vis,alpha=0.7,beta=0.3)
                # print("-------------------------------")
                # print('  loss_ir_consistency ',  loss_ir_consistency , ' loss_vis_consistency',  loss_vis_consistency )

                # 建议把 temporal 从 20 降到 3~5 起步
                w_temporal = 20  #3
                w_ssim = 2.0   #2

                loss_total_fusion = loss_content + loss_grad + w_ssim * loss_ssim + w_temporal * loss_temporal \
                                    + loss_vis_contrast

                loss_fuzhu = 2 * loss_ir_consistency + 1 * loss_vis_consistency

                # 负载均衡/稀疏正则系数建议小一点，避免“抹平”
                w_lb, w_sparse = 1, 1
                loss = loss_total_fusion + loss_fuzhu + w_lb * lb_loss + w_sparse * sparse_loss_4
                # print("-------------------------------")
                # print('loss',loss,'loss_total_fusion',loss_total_fusion,'loss_fuzhu',loss_fuzhu )
                # print('loss',loss,'loss_total_fusion',loss_total_fusion,'loss_fuzhu',loss_fuzhu, 'lb_loss',lb_loss, 'sp_loss',sparse_loss_4)

                # print('loss_ir_consistency', loss_ir_consistency, 'loss_vis_consistency', loss_vis_consistency,'lb_loss',lb_loss, 'sp_loss',sp_loss)
                # print('loss_content',loss_content,'loss_ssim ', loss_ssim , 'loss_temporal',loss_temporal)

                loss.backward()

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} grad mean: {param.grad.abs().mean().item():.6f}")
                self.optimizer.step()

                epoch_loss += loss.item()

            self.lr_scheduler.step()
            avg_loss = epoch_loss / len(self.train_loader_ir)
            print(f"[Epoch {epoch}] Loss: {avg_loss:.6f}")
            self.logger.append([avg_loss])
            self.logger.log()
            with open(os.path.join(self.opt.save_dir, 'log.txt'), 'a') as f:
                f.write(f"Epoch {epoch}: loss = {avg_loss:.6f}\n")

            # if epoch % self.opt.save_interval == 0:
            if epoch % 1 == 0:
                save_path = os.path.join(self.opt.save_dir, f"epoch_fusion_main_1_moe_multi_token_expert1irc0906_{epoch}.pth")  # 可自定义路径和文件名
                save_checkpoint(self.model, self.optimizer, epoch, self.opt, save_path=save_path)

        print("Training finished.")


class FusionTester:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ModifiedResNet18_fusion(32).to(self.device)
        checkpoint_path = os.path.join(opt.save_dir, f"epoch_fusion_main_1_moe_multi_token_expert1irc0906_{opt.test_epoch}.pth")
        print('Loading checkpoint:', checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.ir_dataset = IRNpyDataset_Test(opt.val_ir_json, resize=(256, 256), normalize=True)
        self.vis_dataset = VISNpyDataset_Test(opt.val_vis_json, resize=(256, 256), normalize=True)
        self.ir_loader = DataLoader(self.ir_dataset, batch_size=1, shuffle=False, num_workers=2)
        self.vis_loader = DataLoader(self.vis_dataset, batch_size=1, shuffle=False, num_workers=2)

        # 与训练时一致的归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    @torch.no_grad()
    # def test(self):
    #     save_dir = os.path.join(self.opt.save_dir,
    #                             f'results_epoch_fusion_main_1_moe_multi_token_expert1c_{self.opt.test_epoch}')
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     for idx, ((ir_data, _), (vis_data, vis_data_g, vis_data_b, _)) in tqdm(
    #             enumerate(zip(self.ir_loader, self.vis_loader)),
    #             total=len(self.ir_loader), desc="Testing"):
    #         ir_data = ir_data.to(self.device)
    #         vis_data = vis_data.to(self.device)
    #         vis_data_g = vis_data_g.to(self.device)
    #         vis_data_b = vis_data_b.to(self.device)
    #
    #         ir_2d = ir_data.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
    #         vis_2d = vis_data.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
    #         fusion, fusion_ir, fusion_vis, lb_loss, sparse_loss_4 = self.model(ir_2d, vis_2d)
    #
    #         def save_tensor_img(tensor, name):
    #             img = (tensor * self.std + self.mean).clamp(0, 1)
    #             img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #             Image.fromarray(img_np).save(os.path.join(save_dir, f"{name}_{idx:03d}.png"))
    #
    #         save_tensor_img(fusion_vis[0], "fusion_vis")
    #         save_tensor_img(fusion_ir[0], "fusion_ir")
    #         save_tensor_img(fusion[0], "fusion")
    #         # save_tensor_img(ir_2d[0], "ir")
    #         # save_tensor_img(vis_2d[0], "vis")
    #
    #         # 获取原始路径
    #         ir_path = self.ir_loader.dataset.data_dict[self.ir_loader.dataset.keys[idx]]["path"]
    #         vis_path = self.vis_loader.dataset.data_dict[self.vis_loader.dataset.keys[idx]]["path"]
    #
    #         def save_raw_img(npy_path, name):
    #             raw = np.load(npy_path)[0]  # [H, W, C]
    #             Image.fromarray(raw.astype(np.uint8)).save(os.path.join(save_dir, f"{name}_raw_{idx:03d}.png"))
    #
    #         save_raw_img(ir_path, "ir")
    #         save_raw_img(vis_path, "vis")
    #
    #         # ---------- Step 0: 融合图反标准化 ----------
    #         fusion_denorm = (fusion[0] * self.std + self.mean).clamp(0, 1).cpu().numpy()  # [3,H,W], 0~1
    #         fusion_r = fusion_denorm.mean(axis=0)  # [H,W], 0~1
    #
    #         # ---------- Step 1: 读取 VIS 第0帧，构造初步彩色图 ----------
    #         vis_raw = np.load(vis_path)[0].astype(np.uint8)
    #         vis_raw_resz = cv2.resize(vis_raw, (256, 256), interpolation=cv2.INTER_LINEAR)
    #         vis_rgb = vis_raw_resz.astype(np.float32) / 255.0
    #
    #         pre_rgb = np.stack([fusion_r, vis_rgb[..., 1], vis_rgb[..., 2]], axis=2)  # [H,W,3], 0~1
    #
    #         # ---------- Step 2: YUV 组合 ----------
    #         # 从 VIS 提取 U/V
    #         vis_bgr = vis_raw_resz[..., ::-1]  # RGB->BGR
    #         vis_yuv = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YUV).astype(np.float32)
    #         U_vis, V_vis = vis_yuv[..., 1], vis_yuv[..., 2]  # 色度通道
    #
    #         # 从初步彩色图取 Y
    #         pre_bgr = (pre_rgb[..., ::-1] * 255.0).astype(np.uint8)
    #         pre_yuv = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2YUV).astype(np.float32)
    #         Y_pre = pre_yuv[..., 0] / 255.0  # 0~1
    #
    #         # ---------- Step 3: 增强亮度 ----------
    #         # γ 矫正 (<1 提亮)
    #         gamma = 0.75
    #         Y_new = np.power(np.clip(Y_pre, 0, 1), gamma)
    #
    #         # 可选：直方图均衡，进一步提亮对比度
    #         # Y_new_eq = cv2.equalizeHist((Y_new*255).astype(np.uint8)) / 255.0
    #         # Y_new = 0.7*Y_new + 0.3*Y_new_eq
    #
    #         Y_new = (Y_new * 255.0).astype(np.float32)
    #
    #         # ---------- Step 4: 拼接 YUV 并回 RGB ----------
    #         yuv_out = np.stack([Y_new, U_vis, V_vis], axis=2).astype(np.uint8)
    #         out_bgr = cv2.cvtColor(yuv_out, cv2.COLOR_YUV2BGR)
    #         out_rgb = out_bgr[..., ::-1]
    #
    #         Image.fromarray(out_rgb).save(os.path.join(save_dir, f"fused_rgb_{idx:03d}.png"))
    #
    #     print("✅ Test completed.")
    #-----------------------------------------------------final test
    # def test(self):
    #     import cv2  # 确保已安装：pip install opencv-python
    #
    #     save_dir = os.path.join(
    #         self.opt.save_dir,
    #         f'/home/ubuntu/zhaocheng/infrared_visible_video_dataset/best_model_main8/results_epoch_fusion_main_1_moe_multi_token_expert1c10906_{self.opt.test_epoch}'
    #     )
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     for idx, ((ir_data, _), (vis_data, vis_data_g, vis_data_b, _)) in tqdm(
    #             enumerate(zip(self.ir_loader, self.vis_loader)),
    #             total=len(self.ir_loader), desc="Testing"
    #     ):
    #         ir_data = ir_data.to(self.device)
    #         vis_data = vis_data.to(self.device)
    #
    #         # 模型前向
    #         ir_2d = ir_data.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
    #         vis_2d = vis_data.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
    #         fusion, fusion_ir, fusion_vis, lb_loss, sparse_loss_4 = self.model(ir_2d, vis_2d)
    #
    #         # 保存可视化用（可留可删）
    #         def save_tensor_img(tensor, name):
    #             img = (tensor * self.std + self.mean).clamp(0, 1)
    #             img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #             Image.fromarray(img_np).save(os.path.join(save_dir, f"{name}_{idx:03d}.png"))
    #
    #         save_tensor_img(fusion_vis[0], "fusion_vis")
    #         save_tensor_img(fusion_ir[0], "fusion_ir")
    #         save_tensor_img(fusion[0], "fusion")
    #
    #         # 原始路径（第 idx 个样本）
    #         ir_path = self.ir_loader.dataset.data_dict[self.ir_loader.dataset.keys[idx]]["path"]
    #         vis_path = self.vis_loader.dataset.data_dict[self.vis_loader.dataset.keys[idx]]["path"]
    #
    #         # 保存原始第 0 帧（可留可删）
    #         def save_raw_img(npy_path, name):
    #             raw = np.load(npy_path)[0]  # [H, W, C]
    #             Image.fromarray(raw.astype(np.uint8)).save(
    #                 os.path.join(save_dir, f"{name}_raw_{idx:03d}.png")
    #             )
    #
    #         save_raw_img(ir_path, "ir")
    #         save_raw_img(vis_path, "vis")
    #
    #         # ---------------------------
    #         # YUV 方案：用融合图替换 VIS 的 Y (亮度)
    #         # ---------------------------
    #
    #         # 1) 融合亮度：先反标准化到 0~1，再取均值作为亮度
    #         fusion_denorm = (fusion[0] * self.std + self.mean).clamp(0, 1)  # [3,H,W]
    #         fusion_y = fusion_denorm.mean(dim=0).cpu().numpy()  # [H,W], 0~1
    #
    #         # 2) 取原始 VIS 第 0 帧，缩放到模型输出分辨率（若你不是 256×256，请改成对应尺寸）
    #         vis_raw = np.load(vis_path)[0].astype(np.uint8)  # [H0,W0,3], RGB, 0~255
    #         vis_raw_resz = cv2.resize(vis_raw, (256, 256), interpolation=cv2.INTER_LINEAR)
    #
    #         # 3) RGB→YUV（OpenCV 用 BGR）
    #         vis_bgr = vis_raw_resz[..., ::-1]  # RGB->BGR
    #         vis_yuv = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YUV).astype(np.float32)  # [H,W,3], 0~255
    #
    #         # 4) 亮度合成：让融合图主导（alpha 越大，融合图越主导）
    #         alpha_y = 0.5  # 0.9~1.0 之间调；1.0 表示完全用融合 Y
    #         Y_vis = vis_yuv[..., 0] / 255.0
    #         Y_new = alpha_y * fusion_y + (1.0 - alpha_y) * Y_vis
    #
    #         # 可选：轻微 γ 调整提升暗部细节（<1 提亮）
    #         gamma = 0.75
    #         Y_new = np.clip(np.power(np.clip(Y_new, 0, 1), gamma), 0, 1)
    #
    #         # 5) 替换 Y
    #         vis_yuv[..., 0] = (Y_new * 255.0).astype(np.float32)
    #
    #         # 6) 可选：提高饱和度（放大 U/V 偏离 128 的幅度），让颜色更明显但不盖住亮度
    #         sat_gain = 1.15 # 1.0 不变；1.1~1.3 稍微增强；太大会偏色
    #         vis_yuv[..., 1] = np.clip(128.0 + (vis_yuv[..., 1] - 128.0) * sat_gain, 0, 255)
    #         vis_yuv[..., 2] = np.clip(128.0 + (vis_yuv[..., 2] - 128.0) * sat_gain, 0, 255)
    #
    #         # 7) 回到 RGB 并保存
    #         fused_bgr = cv2.cvtColor(vis_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
    #         fused_rgb = fused_bgr[..., ::-1]
    #         Image.fromarray(fused_rgb).save(os.path.join(save_dir, f"fused_rgb_{idx:03d}.png"))
    #
    #     print("✅ Test completed.")

    #-----------------------------------------fineal every frame
    def test(self):
        import cv2

        save_dir = os.path.join(
            self.opt.save_dir,
            f'/home/ubuntu/zhaocheng/infrared_visible_video_dataset/best_model_main8/results_epoch_fusion_main_1_moe_multi_token_expert1c10906_finealvis_{self.opt.test_epoch}'
        )
        os.makedirs(save_dir, exist_ok=True)

        for idx, ((ir_data, _), (vis_data, vis_data_g, vis_data_b, _)) in tqdm(
                enumerate(zip(self.ir_loader, self.vis_loader)),
                total=len(self.ir_loader), desc="Testing"
        ):
            ir_data = ir_data.to(self.device)  # [B,T,C,H,W]
            vis_data = vis_data.to(self.device)  # [B,T,C,H,W]


            # 模型前向
            ir_data = ir_data.permute(0, 2, 1, 3, 4)
            vis_data = vis_data.permute(0, 2, 1, 3, 4)
            B, T, C, H, W = vis_data.shape

            # 展平为 [B*T,C,H,W]
            ir_2d = ir_data.reshape(-1, C, H, W)
            vis_2d = vis_data.reshape(-1, C, H, W)
            fusion, fusion_ir, fusion_vis, lb_loss, sparse_loss_4 = self.model(ir_2d, vis_2d)

            # 恢复为 [B,T,C,H,W]
            print("ir_data:", ir_data.shape)
            print("ir_2d:", ir_2d.shape)
            print("fusion before view:", fusion.shape)
            print("fusion after view:", fusion.view(B, T, C, H, W).shape)

            fusion = fusion.view(B, T, C, H, W)
            fusion_ir = fusion_ir.view(B, T, C, H, W)
            fusion_vis = fusion_vis.view(B, T, C, H, W)

            # 遍历 batch 中的每个视频
            for b in range(B):
                sample_id = idx * B + b
                sample_dir = os.path.join(save_dir, f"sample_{sample_id:03d}")
                os.makedirs(sample_dir, exist_ok=True)

                # 原始路径
                ir_path = self.ir_loader.dataset.data_dict[self.ir_loader.dataset.keys[sample_id]]["path"]
                vis_path = self.vis_loader.dataset.data_dict[self.vis_loader.dataset.keys[sample_id]]["path"]

                ir_raw = np.load(ir_path).astype(np.uint8)  # [T,H,W,3]
                vis_raw = np.load(vis_path).astype(np.uint8)  # [T,H,W,3]
                for t in range(T):
                    # ----------------------
                    # 保存网络输出帧
                    # ----------------------
                    print(T)
                    print(fusion.shape, fusion[b, t].shape)

                    def save_tensor_img(tensor, name, sample_dir, t):
                        # tensor: [3,H,W]
                        if tensor.dim() == 4 and tensor.shape[0] != 3:
                            # 如果进来的是 [T,H,W]，取第一帧或 permute
                            tensor = tensor.permute(1, 0, 2)  # [3,H,W]

                        img = (tensor * self.std.view(3, 1, 1).to(tensor.device) +
                               self.mean.view(3, 1, 1).to(tensor.device)).clamp(0, 1)
                        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_np).save(os.path.join(sample_dir, f"{name}_{t:03d}.png"))

                    # save_tensor_img(fusion[b, t], "fusion", sample_dir, t)  # [3,H,W]
                    # save_tensor_img(fusion_ir[b, t], "fusion_ir", sample_dir, t)  # [3,H,W]
                    # save_tensor_img(fusion_vis[b, t], "fusion_vis", sample_dir, t)  # [3,H,W]

                    # ----------------------
                    # 保存原始帧
                    # ----------------------
                    # Image.fromarray(ir_raw[t]).save(os.path.join(sample_dir, f"ir_raw_{t:03d}.png"))
                    # Image.fromarray(vis_raw[t]).save(os.path.join(sample_dir, f"vis_raw_{t:03d}.png"))

                    # ----------------------
                    # 融合到 YUV 的结果
                    # ----------------------
                    fusion_denorm = (fusion_vis[b, t] * self.std + self.mean).clamp(0, 1)
                    fusion_y = fusion_denorm.mean(dim=0).cpu().numpy()

                    vis_raw_resz = cv2.resize(vis_raw[t], (H, W), interpolation=cv2.INTER_LINEAR)
                    vis_bgr = vis_raw_resz[..., ::-1]
                    vis_yuv = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YUV).astype(np.float32)

                    alpha_y = 0.7
                    Y_vis = vis_yuv[..., 0] / 255.0
                    Y_new = alpha_y * fusion_y + (1.0 - alpha_y) * Y_vis

                    gamma = 0.85
                    Y_new = np.clip(np.power(np.clip(Y_new, 0, 1), gamma), 0, 1)

                    vis_yuv[..., 0] = (Y_new * 255.0).astype(np.float32)

                    sat_gain = 1.15
                    vis_yuv[..., 1] = np.clip(128.0 + (vis_yuv[..., 1] - 128.0) * sat_gain, 0, 255)
                    vis_yuv[..., 2] = np.clip(128.0 + (vis_yuv[..., 2] - 128.0) * sat_gain, 0, 255)

                    fused_bgr = cv2.cvtColor(vis_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
                    fused_rgb = fused_bgr[..., ::-1]
                    Image.fromarray(fused_rgb).save(os.path.join(sample_dir, f"fused_rgb_{t:03d}.png"))

                print("✅ Test completed (all frames saved).")

        print("✅ Test completed (all frames saved).")

    # def test(self):
    #     save_dir = os.path.join(self.opt.save_dir, f'results_epoch_fusion_main_1_moe_multi_token_expert1_{self.opt.test_epoch}')
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     for idx, ((ir_data, _), (vis_data, _)) in tqdm(enumerate(zip(self.ir_loader, self.vis_loader)), total=len(self.ir_loader), desc="Testing"):
    #         ir_data = ir_data.to(self.device)
    #         vis_data = vis_data.to(self.device)
    #
    #         ir_2d = ir_data.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
    #         vis_2d = vis_data.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
    #         fusion, fusion_ir, fusion_vis,lb_loss , sparse_loss_4 = self.model(ir_2d, vis_2d)
    #
    #         def save_tensor_img(tensor, name):
    #             img = (tensor * self.std + self.mean).clamp(0, 1)
    #             img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #             Image.fromarray(img_np).save(os.path.join(save_dir, f"{name}_{idx:03d}.png"))
    #
    #         save_tensor_img(fusion_vis[0], "fusion_vis")
    #         save_tensor_img(fusion_ir[0], "fusion_ir")
    #         save_tensor_img(fusion[0], "fusion")
    #         # save_tensor_img(ir_2d[0], "ir")
    #         # save_tensor_img(vis_2d[0], "vis")
    #
    #         def save_raw_img(npy_path, name):
    #             raw = np.load(npy_path)[0]  # [H, W, C]
    #             Image.fromarray(raw.astype(np.uint8)).save(os.path.join(save_dir, f"{name}_raw_{idx:03d}.png"))
    #
    #         ir_path = self.ir_loader.dataset.data_dict[self.ir_loader.dataset.keys[idx]]["path"]
    #         vis_path = self.vis_loader.dataset.data_dict[self.vis_loader.dataset.keys[idx]]["path"]
    #         save_raw_img(ir_path, "ir")
    #         save_raw_img(vis_path, "vis")
    #
    #     print("✅ Test completed.")
