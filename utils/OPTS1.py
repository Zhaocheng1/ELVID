import argparse
import os

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ---------------------- 基本配置 ----------------------
        self.parser.add_argument('--task', default='fusion', help='任务名称')
        self.parser.add_argument('--save_dir', default='/home/ubuntu/zhaocheng/infrared_visible_video_dataset/best_model', help='模型保存目录')
        self.parser.add_argument('--gpus', default='0', help='使用的GPU编号，多个GPU用逗号分隔')

        # ---------------------- 数据路径 ----------------------
        self.parser.add_argument('--train_ir_json',type=str,default='/home/ubuntu/zhaocheng/infrared_visible_video_dataset/json_files/ir_dataset.json',   help='红外训练数据 JSON 路径')
        self.parser.add_argument('--train_vis_json',type=str,default='/home/ubuntu/zhaocheng/infrared_visible_video_dataset/json_files/vis_dataset.json',  help='可见光训练数据 JSON 路径（可选）')
        self.parser.add_argument('--val_ir_json',type=str,default='/home/ubuntu/zhaocheng/infrared_visible_video_dataset/json_files/test_ir_only.json',   help='红外验证数据 JSON 路径')
        self.parser.add_argument('--val_vis_json',type=str, default='/home/ubuntu/zhaocheng/infrared_visible_video_dataset/json_files/test_vis_only.json', help='可见光验证数据 JSON 路径')

        # ---------------------- 训练参数 ----------------------
        self.parser.add_argument('--batch_size', type=int, default=2, help='训练 batch 大小')
        self.parser.add_argument('--epochs', type=int, default=60, help='训练总轮数')
        self.parser.add_argument('--test_epoch', type=int, default=10, help='测试')
        self.parser.add_argument('--save_interval', type=int, default=5, help='训练')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
        self.parser.add_argument('--lr_step', type=int, nargs='+', default=[20,45], help='学习率衰减 epoch')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=20, help='Layer-wise learning rate decay epoch')
        self.parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
        self.parser.add_argument('--resume', type=str, default='', help='恢复训练的模型路径')

        # ---------------------- 损失函数权重 ----------------------
        # self.parser.add_argument('--lambda_recon', type=float, default=10.0, help='重建损失权重')
        self.parser.add_argument('--lambda_ssim', type=float, default=1.0, help='结构相似性损失权重')
        self.parser.add_argument('--lambda_grad', type=float, default=1.0, help='边缘梯度损失权重')
        self.parser.add_argument('--lambda_temporal', type=float, default=1.0, help='时序一致性损失权重')

        # ---------------------- 模型参数 ----------------------
        self.parser.add_argument('--model_type', type=str, default='ModifiedResNet18', help='模型类型，如 fusion_net')
        self.parser.add_argument('--input_size', type=int, nargs=2, default=[448, 448], help='输入图像大小')
        self.parser.add_argument('--input_channels1', type=int, default=1, help='红外图像通道数')
        self.parser.add_argument('--input_channels2', type=int, default=1, help='VIS图像通道数')
        self.parser.add_argument('--output_channels', type=int, default=1, help='输出图像通道数')
        self.parser.add_argument('--num_frames', type=int, default=16, help='每段视频包含的帧数')

        # ---------------------- 其他 ----------------------
        self.parser.add_argument('--seed', type=int, default=42, help='随机种子')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # GPU 设置
        opt.gpus_str = opt.gpus
        opt.gpus = [int(g) for g in opt.gpus.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

        # 创建保存目录
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        return opt