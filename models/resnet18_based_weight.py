import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
import torch.nn.functional as F
from models.Expert_MoE3 import CrossLayerExpertFusion
from models.confidence_weighted_fusion import ConfidenceWeightedFusion,ConfidenceWeightedFusion1,CrossWeightedFusion,CrossWeightedFusion1
# ========== 多尺度局部注意力辅助函数 ==========
def get_local_mask(T, window, device):
    mask = torch.full((T, T), float('-inf'), device=device)
    for i in range(T):
        for j in range(max(0, i - window), min(T, i + window + 1)):
            mask[i, j] = 0.0
    return mask

# ========== 简化版多尺度局部注意力 + 熵置信控制模块 ==========
class MultiScaleLocalAttentionWithEntropyGate(nn.Module):
    def __init__(self, channels, num_heads=4, windows=[1, 2, 4], dropout=0.1):
        super().__init__()
        self.local_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout)
            for _ in windows
        ])
        self.windows = windows
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_pool = x.view(B, T, C, -1).mean(-1)  # (B, T, C)
        x_pool = x_pool.permute(1, 0, 2)  # (T, B, C) for MultiheadAttention

        # 多尺度局部注意力
        local_contexts = []
        local_weights = []
        for attn, win in zip(self.local_attns, self.windows):
            mask = get_local_mask(T, win, x.device)
            local_out, attn_w = attn(x_pool, x_pool, x_pool, attn_mask=mask) #使用相同的x_pool作为query/key/value输入
            local_contexts.append(local_out) # (B, T, C)组成的列表
            local_weights.append(attn_w)

        local_context = torch.stack(local_contexts, dim=0).mean(0)  # 平均融合 (B, T, C)
        local_context = local_context.permute(1, 0, 2)
        attn_probs = torch.stack(local_weights, dim=0).mean(0)     # 平均注意力 (B, T, T)

        fused = self.norm(local_context)
        fused = self.proj(fused).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W) # (B, T, C, H, W)

        # ===== 熵引导帧置信度计算 =====
        with torch.no_grad():
            entropy = - (attn_probs * torch.log(attn_probs + 1e-8)).sum(-1)  # (B, T)
            entropy_min = entropy.min(dim=1, keepdim=True)[0] #min方法返回一个元组，第一个元素是计算得到的最小值，第二个元素是这些最小值的索引
            entropy_max = entropy.max(dim=1, keepdim=True)[0]
            entropy_norm = (entropy - entropy_min) / (entropy_max - entropy_min + 1e-6)
            confidence = 1 - entropy_norm  # (B, T)

        confidence = confidence.view(B, T, 1, 1, 1)
        x_out = (x + fused) * confidence  # 帧置信控制

        return x_out, confidence  # (B, T, C, H, W), (B, T, 1, 1, 1)

class GatedTimeAwareSEBlock(nn.Module):
    def __init__(self, channels, max_len=500, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.gate_fc = nn.Linear(channels, 1)  # 帧级“保留分支”
        self.time_embed = nn.Parameter(torch.randn(1, max_len, channels))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_pool = x.view(B, T, C, -1).mean(-1)  # (B, T, C)
        time_info = self.time_embed[:, :T, :]
        x_enhanced = x_pool + time_info  # (B, T, C)shijianxinxiqianru

        # 通道注意力
        se = F.relu(self.fc1(x_enhanced))
        se = torch.sigmoid(self.fc2(se))  # (B, T, C)

        # 帧门控：是否保留该帧（0~1）
        gate = torch.sigmoid(self.gate_fc(x_enhanced))  # (B, T, 1)
        # 融合权重
        out = x * se.view(B, T, C, 1, 1)
        out = out * gate.view(B, T, 1, 1, 1)  # Gated attention
        return out, gate  # 返回帧置信度（可视化/筛选用）

class FusionTemporalBlock(nn.Module):
    def __init__(self, channels, max_len=500, reduction=16, heads=4, windows=[1, 2, 4],dropout_rate=0.2):
        super().__init__()
        self.tse = GatedTimeAwareSEBlock(channels, max_len=max_len, reduction=reduction)
        self.attn = MultiScaleLocalAttentionWithEntropyGate(channels, num_heads=heads, windows=windows)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x_tem1, gate_tem = self.tse(x)
        x_tem2, confidence_tem = self.attn(x)   # 帧间多尺度注意力 + 置信度
        return x_tem1 + x_tem2, confidence_tem ,gate_tem ,x_tem1,x_tem2    # 残差融合

####################################spatial###################################
# 多尺度空间注意力模块
class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, 1, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, 1, 5, padding=2)
        self.conv7 = nn.Conv2d(channels, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_3 = self.conv3(x)
        attn_5 = self.conv5(x)
        attn_7 = self.conv7(x)
        attn = self.sigmoid((attn_3 + attn_5 + attn_7) / 3)
        return attn  # (B, 1, H, W)

# 熵计算函数
def compute_attention_entropy(attn_map):
    B, _, H, W = attn_map.shape
    flat = attn_map.view(B, -1) #(B,H*W)
    norm = flat / (flat.sum(dim=1, keepdim=True) + 1e-6) #(B,H*W)
    entropy = - (norm * torch.log(norm + 1e-8)).sum(dim=1)  # (B,)
    return entropy  # (B,)

# 修改版的 EchocardiographySpatialAttentionBlock 模块
class EchocardiographySpatialAttentionBlock(nn.Module):
    def __init__(self, channels, fusion_mode="adaptive"):
        """
        fusion_mode: 'conf_weighted', 'gated', or 'adaptive'
        """
        super().__init__()
        self.spatial_attn = MultiScaleSpatialAttention(channels)
        self.fusion_mode = fusion_mode
        if fusion_mode == "adaptive":
            self.fusion_gate = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        返回:
            x_out: (B, T, C, H, W)
            confidence: (B, T, 1, 1, 1)
            attention_maps: (B, T, 1, H, W)
        """
        B, T, C, H, W = x.shape
        x_out = torch.zeros_like(x)
        attention_maps = []
        entropy_list = []

        # 计算每一帧的注意力图与熵
        for t in range(T):
            x_t = x[:, t]  # (B, C, H, W)
            attn = self.spatial_attn(x_t)  # (B, 1, H, W)

            if t > 0 and t < T - 1:
                attn_prev = self.spatial_attn(x[:, t - 1])
                attn_next = self.spatial_attn(x[:, t + 1])
                attn = (attn + attn_prev + attn_next) / 3

            entropy = compute_attention_entropy(attn)
            entropy_list.append(entropy.unsqueeze(1))  # (B, 1)
            attention_maps.append(attn.unsqueeze(1))   # (B, 1, 1, H, W)

        # 拼接
        entropy_all = torch.cat(entropy_list, dim=1)  # (B, T)
        attn_maps_all = torch.cat(attention_maps, dim=1)  # (B, T, 1, H, W)
        conf = 1 - ((entropy_all - entropy_all.min(1, keepdim=True)[0]) /
                    (entropy_all.max(1, keepdim=True)[0] - entropy_all.min(1, keepdim=True)[0] + 1e-6)) # (B, T)
        conf = conf.view(B, T, 1, 1, 1)  # (B, T, 1, 1, 1)

        # 根据融合方式进行加权
        for t in range(T):
            x_t = x[:, t] # (B, C, H, W)
            attn = attn_maps_all[:, t] # (B, 1, H, W)
            c = conf[:, t] # (B, 1, 1, 1)

            if self.fusion_mode == "conf_weighted":
                x_out[:, t] = x_t * attn * c#先用 attn 进行空间注意力加权，然后再用 conf 加上全局的帧级置信度加权。如果当前帧置信度低（如 c ≈ 0），则整个帧的信息对最终输出影响减小。
            elif self.fusion_mode == "gated":
                attn_gate = attn * c + (1 - c)#根据置信度 c 对 attn 进行门控变换，计算出的 attn_gate 是一种平衡权重。当 c ≈ 1 时，attn_gate ≈ attn,当 c ≈ 0 时，attn_gate ≈ 1，也就是不使用 attention，直接保留原始特征
                x_out[:, t] = x_t * attn_gate
            elif self.fusion_mode == "adaptive":
                # 学习得到一个融合因子 alpha ∈ [0,1]
                alpha = self.fusion_gate(c.view(B, 1))  # (B, 1)
                alpha = alpha.view(B, 1, 1, 1)
                x_out[:, t] = x_t * (alpha * attn + (1 - alpha))#学习一个 α ∈ [0,1] 的因子，来控制 attn 与 identity 的线性组合权重。如果 α ≈ 1 → 完全信任注意力机制,如果 α ≈ 0 → 放弃注意力，保留原始特征.让网络自动调节注意力的使用程度，更具适应性和灵活性。
            else:
                x_out[:, t] = x_t * attn  # fall back

        return x_out, conf, attn_maps_all

def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_dim),
        act_fn,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, act_fn=nn.ReLU(inplace=True)):
        super(UNetDecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_fn

    def forward(self, x, skip):
        skip = skip.view(-1, *skip.shape[2:])
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # skip 是 encoder 对应层输出
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x

class UNetDecoderBlock1(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, act_fn=nn.ReLU(inplace=True)):
        super(UNetDecoderBlock1, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_fn

    def forward(self, x, skip):
        skip = skip.view(-1, *skip.shape[2:])
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # skip 是 encoder 对应层输出
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


############################多视图logits融合###############################
class MultiViewInteraction(nn.Module):
    def __init__(self, num_classes, num_views=8, hidden_dim=64):
        super().__init__()
        self.num_views = num_views

        # 交叉注意力机制
        self.query = nn.Linear(num_classes, hidden_dim)
        self.key = nn.Linear(num_classes, hidden_dim)
        self.value = nn.Linear(num_classes, hidden_dim)

        # 自适应权重生成
        self.weight_net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 特征增强层
        self.enhance = nn.Sequential(
            nn.Linear(num_classes + hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, num_classes)
        )

    def forward(self, logits_list):
        """
        Args: 输入是一个列表
            logits_list: list of 8 tensors with shape (bs, num_classes)
        Returns:
            fused_logits: tensor of shape (bs, num_classes)
        """
        # 1. 堆叠所有logits
        stacked = torch.stack(logits_list, dim=1)  # (bs, 8, c)

        # 2. 交叉注意力交互
        Q = self.query(stacked)  # (bs, 8, h)
        K = self.key(stacked)    # (bs, 8, h)
        V = self.value(stacked)  # (bs, 8, h)

        attn = torch.matmul(Q, K.transpose(1,2)) / torch.sqrt(torch.tensor(Q.size(-1)))
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, V)  # (bs, 8, h)

        # 3. 自适应加权融合
        weights = torch.cat([self.weight_net(view) for view in logits_list], dim=1)
        weights = F.softmax(weights, dim=1).unsqueeze(-1)  # (bs, 8, 1)

        # 4. 多阶段融合
        weighted_sum = (stacked * weights).sum(dim=1)  # (bs, c)
        cross_attn_sum = context.mean(dim=1)  # (bs, h)

        # 5. 特征增强
        enhanced = self.enhance(torch.cat([weighted_sum, cross_attn_sum], dim=1))
        return enhanced
###############################backbone###################################
class ModifiedResNet18_weight(nn.Module):
    def __init__(self,ngf):
        super().__init__()
        # 加载预训练模型并提取所需层
        original = resnet34(pretrained=True)
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        self.out_dim = ngf

        # 提取到conv2_x的输出
        self.conv1 = original.conv1
        self.bn1 = original.bn1
        self.relu = original.relu
        self.maxpool = original.maxpool
        self.layer1 = original.layer1  # 对应conv2_x
        self.layer2 = original.layer2
        self.layer3 = original.layer3
        self.layer4 = original.layer4
        self.avgpool = original.avgpool
        # 新建全连接层（不加载预训练权重）
        self.fc = nn.Linear(512, 2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.5)

        self.conv0 =  nn.Conv2d(128, 64, kernel_size=1)
        self.tem_branch2 = FusionTemporalBlock(64)
        self.spat_branch2 = EchocardiographySpatialAttentionBlock(64)
        self.tem_branch3 = FusionTemporalBlock(128)
        self.spat_branch3 = EchocardiographySpatialAttentionBlock(128)
        self.tem_branch4 = FusionTemporalBlock(256)
        self.spat_branch4 = EchocardiographySpatialAttentionBlock(256)
        self.tem_branch5 = FusionTemporalBlock(512)
        self.spat_branch5 = EchocardiographySpatialAttentionBlock(512)

        self.fusion_weight1 = CrossWeightedFusion(64,64)
        self.fusion_weight2 = CrossWeightedFusion1(64,128)
        self.fusion_weight3 = CrossWeightedFusion1(128,256)
        self.fusion_weight4 = CrossWeightedFusion1(256,512)

        # self.fusion_weight1 = ConfidenceWeightedFusion(64, 64)
        # self.fusion_weight2 = ConfidenceWeightedFusion1(64, 128)
        # self.fusion_weight3 = ConfidenceWeightedFusion1(128, 256)
        # self.fusion_weight4 = ConfidenceWeightedFusion1(256, 512)

        # self.fusion_expertset1 = CrossLayerExpertFusion(64,64,4)
        # self.fusion_expertset2 = CrossLayerExpertFusion(64,128,4)
        # self.fusion_expertset3 = CrossLayerExpertFusion(128,256,4)
        # self.fusion_expertset4 = CrossLayerExpertFusion(256,512,4)
        # # 冻结所有参数（可选）
        # for param in self.parameters():
        #     param.requires_grad_(False)
        # 加载预训练权重（自动处理）
        self._load_pretrained_weights(original)

        # self.deconv_0_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn)
        # self.deconv_1_1 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn)
        # self.deconv_2_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn)
        # self.deconv_3_1 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn)
        # self.deconv_4_1 = conv_decod_block(self.out_dim * 1, 16, act_fn)
        # self.out1 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1, stride=1), nn.Tanh())  # self.final_out_dim
        self.deconv_0_1 = UNetDecoderBlock(512, 256, 256)
        self.deconv_1_1 = UNetDecoderBlock(256, 128, 128)
        self.deconv_2_1 = UNetDecoderBlock(128, 64, 64)
        self.deconv_3_1_ = UNetDecoderBlock1(64, 128, 32)
        self.deconv_4_1 = UNetDecoderBlock(32, 16, 16)
        self.out1 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),nn.Tanh(),nn.Conv2d(16, 3, kernel_size=1, stride=1)) #  self.final_out_dim

    def _load_pretrained_weights(self, pretrained_model):
        model_dict = self.state_dict()
        pretrained_dict = pretrained_model.state_dict()

        # 过滤策略：
        # 1. 排除fc层（新定义）
        # 2. 排除layer2之后的权重（因为输入维度被custom_layer改变）
        filtered_dict = {k: v for k, v in pretrained_dict.items()
                         if k in model_dict
                         and not k.startswith('fc')}

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)  # 允许部分加载 strict=False可省略

    def forward(self, x_ir,y_vis):
        # 标准前向传播流程
        #32,3,224,224
 # -----------------------------------layer0---------------------------------------------------------------------------------
        x_ir = self.conv1(x_ir)#32,64,112,112
        x_ir = self.bn1(x_ir)#32,64,112,112
        x_ir = self.relu(x_ir)#32,64,112,112
        x_ir1 = x_ir
        x_ir = self.maxpool(x_ir)#32,64,56,56d

        y_vis = self.conv1(y_vis)  # 32,64,112,112
        y_vis = self.bn1(y_vis)  # 32,64,112,112
        y_vis = self.relu(y_vis)  # 32,64,112,112
        y_vis1 = y_vis
        y_vis = self.maxpool(y_vis)  # 32,64,56,56

        x_ir1 =  x_ir1.view(-1, 16, * x_ir1.shape[1:])
        y_vis1 = y_vis1.view(-1, 16, *y_vis1.shape[1:])

        fusion_layer0 = torch.cat([x_ir1,   y_vis1], dim=2) #[32,64,256,128]
        fusion_layer0 = fusion_layer0.view(-1, fusion_layer0.shape[2], fusion_layer0.shape[3], fusion_layer0.shape[4])
        fusion_layer0 = self.conv0(fusion_layer0)
        fusion_layer0 = fusion_layer0.view(-1,16, *fusion_layer0.shape[1:])


        #-----------------------------------layer1---------------------------------------------------------------------------------
        ir_feature2 = self.layer1(x_ir) #(bs*16,64,56,56)
        ir_feature2_ = ir_feature2.view(-1,16, *ir_feature2.shape[1:]) #(bs,16,64,56,56)(2,16,64,56,56)
        x_tem1, confidence_tem_ir1, _, _, _ = self.tem_branch2(ir_feature2_)
        x_spat1, conf_spat_ir1, attn_maps_all_spat1 = self.spat_branch2(ir_feature2_)
        ir_features2__ = x_tem1 + x_spat1 + ir_feature2_ #要不要加feature2？(bs,16,64,56,56)

        vis_feature2 = self.layer1(y_vis)  # (bs*16,64,56,56)
        vis_feature2_ = ir_feature2.view(-1, 16, *vis_feature2.shape[1:])  # (bs,16,64,56,56)(2,16,64,56,56)
        y_tem2, confidence_tem_vis1, _, _, _ = self.tem_branch2(vis_feature2_)
        y_spat2, conf_spat_vis1, attn_maps_all_spat2 = self.spat_branch2(vis_feature2_)
        vis_features2__ = y_tem2+ y_spat2 + vis_feature2_  # 要不要加feature2？(bs,16,64,56,56)

        # fusion_layer1_ = self.fusion_expertset1(fusion_layer0, ir_features2__, vis_features2__)
        # fusion_layer1 = fusion_layer1_["fused_output"]
        # lb_loss_1 = fusion_layer1_["load_balancing_loss"]
        # sparse_loss_1 = fusion_layer1_["sparsity_loss"]
        conf_ir1 = confidence_tem_ir1 + conf_spat_ir1
        conf_vis1 = confidence_tem_vis1 + conf_spat_vis1
        # print(conf_ir1 .shape)
        # print(conf_vis1.shape)
        fusion_layer1 = self.fusion_weight1( vis_feat=vis_features2__, ir_feat=ir_features2__,prev_fused_feat=fusion_layer0)


        # -----------------------------------layer2---------------------------------------------------------------------------------
        ir_feature3 = self.layer2(ir_features2__.view(-1, *ir_features2__.shape[-3:])) #输入(bs*16,64,56,56) 输出(bs*16,128,28,28)
        ir_feature3_ = ir_feature3.view(-1, 16, *ir_feature3.shape[1:]) #(bs,16,128,28,28)
        x_tem2, confidence_tem_ir2, _, _, _ = self.tem_branch3(ir_feature3_)
        x_spat2, conf_spat_ir2, attn_maps_all_spat2 = self.spat_branch3(ir_feature3_)
        ir_features3__ = x_tem2 + x_spat2 + ir_feature3_ #(bs,16,128,28,28)

        vis_feature3 = self.layer2( vis_features2__.view(-1, *vis_features2__.shape[-3:]))  # 输入(bs*16,64,56,56) 输出(bs*16,128,28,28)
        vis_feature3_ = vis_feature3.view(-1, 16, *vis_feature3.shape[1:])  # (bs,16,128,28,28)
        y_tem2, confidence_tem_vis2, _, _, _ = self.tem_branch3(vis_feature3_)
        y_spat2, conf_spat_vis2, attn_maps_all_spat2 = self.spat_branch3(vis_feature3_)
        vis_features3__ = y_tem2 + y_spat2 + vis_feature3_  # (bs,16,128,28,28)


        # fusion_layer2_ = self.fusion_expertset2(fusion_layer1, ir_features3__, vis_features3__)
        # fusion_layer2= fusion_layer2_["fused_output"]
        # lb_loss_2 = fusion_layer2_["load_balancing_loss"]
        # sparse_loss_2 = fusion_layer2_["sparsity_loss"]
        conf_ir2 = confidence_tem_ir2 + conf_spat_ir2
        conf_vis2 = confidence_tem_vis2 + conf_spat_vis2
        fusion_layer2 = self.fusion_weight2(vis_feat=vis_features3__, ir_feat=ir_features3__, prev_fused_feat=fusion_layer1)



 # -----------------------------------layer3---------------------------------------------------------------------------------
        ir_feature4 = self.layer3(ir_features3__.view(-1, *ir_features3__.shape[-3:])) #输入(bs*16,128,28,28) 输出(bs*16,256,14,14)
        ir_feature4_ = ir_feature4.view(-1, 16, *ir_feature4.shape[1:]) #(bs,16,256,14,14)
        x_tem3, confidence_tem_ir3, _, _, _ = self.tem_branch4(ir_feature4_)
        x_spat3, conf_spat_ir3, attn_maps_all_spat3 = self.spat_branch4(ir_feature4_)
        ir_features4__ = x_tem3 + x_spat3 + ir_feature4_ #(bs,16,256,14,14)

        vis_feature4 = self.layer3(vis_features3__.view(-1, *vis_features3__.shape[-3:]))  # 输入(bs*16,128,28,28) 输出(bs*16,256,14,14)
        vis_feature4_ = vis_feature4.view(-1, 16, *vis_feature4.shape[1:])  # (bs,16,256,14,14)
        y_tem3, confidence_tem_vis3, _, _, _ = self.tem_branch4(vis_feature4_)
        y_spat3, conf_spat_vis3, attn_maps_all_spat3 = self.spat_branch4(vis_feature4_)
        vis_features4__ = y_tem3 + y_spat3 + vis_feature4_  # (bs,16,256,14,14)

        # fusion_layer3_ = self.fusion_expertset3(fusion_layer2, ir_features4__, vis_features4__)
        # fusion_layer3 = fusion_layer3_["fused_output"]
        # lb_loss_3 = fusion_layer3_["load_balancing_loss"]
        # sparse_loss_3 = fusion_layer3_["sparsity_loss"]
        conf_ir3 = confidence_tem_ir3 + conf_spat_ir3
        conf_vis3 = confidence_tem_vis3 + conf_spat_vis3
        fusion_layer3 = self.fusion_weight3(vis_feat=vis_features4__, ir_feat=ir_features4__, prev_fused_feat=fusion_layer2)



# -----------------------------------layer4---------------------------------------------------------------------------------
        ir_feature5 = self.layer4(ir_features4__.view(-1, *ir_features4__.shape[-3:])) #输入(bs*16,256,14,14) 输出(bs*16,512,7,7)
        ir_feature5_ = ir_feature5.view(-1, 16, *ir_feature5.shape[1:]) #(bs,16,512,7,7)
        x_tem4, confidence_tem_ir4, _, _, _ = self.tem_branch5(ir_feature5_)
        x_spat4, conf_spat_ir4, attn_maps_all_spat = self.spat_branch5(ir_feature5_)
        ir_features5__ = x_tem4 + x_spat4 + ir_feature5_ #(bs,16,512,7,7)

        vis_feature5 = self.layer4(vis_features4__.view(-1, *vis_features4__.shape[-3:]))  # 输入(bs*16,256,14,14) 输出(bs*16,512,7,7)
        vis_feature5_ = vis_feature5.view(-1, 16, *vis_feature5.shape[1:])  # (bs,16,512,7,7)
        y_tem4, confidence_tem_vis4, _, _, _ = self.tem_branch5(vis_feature5_)
        y_spat4, conf_spat_vis4, attn_maps_all_spat1 = self.spat_branch5(vis_feature5_)
        vis_features5__ = y_tem4 + y_spat4 + vis_feature5_  # (bs,16,512,7,7)

        # ir_features5__ = self.dropout2(ir_features5__.view(-1, *ir_features5__.shape[-3:]))  # Dropout 2d
        # ir_features5__ = ir_features5__.view(-1, 16, 512, 16, 16)
        # vis_features5__ = self.dropout2(vis_features5__.view(-1, *vis_features5__.shape[-3:]))  # Dropout 2d
        # vis_features5__ = vis_features5__.view(-1, 16, 512, 16, 16)

        # fusion_layer4_ = self.fusion_expertset4(fusion_layer3, ir_features5__, vis_features5__)
        # fusion_layer4 = fusion_layer4_["fused_output"]
        # lb_loss_4 = fusion_layer4_["load_balancing_loss"]
        # sparse_loss_4 = fusion_layer4_["sparsity_loss"]
        conf_ir4 = confidence_tem_ir4 + conf_spat_ir4
        conf_vis4 = confidence_tem_vis4 + conf_spat_vis4
        fusion_layer4 = self.fusion_weight4(vis_feat=vis_features5__, ir_feat=ir_features5__,  prev_fused_feat=fusion_layer3)

        # -----------------------------------fusion_decoder---------------------------------------------------------------------------------
        fusion_layer4 = fusion_layer4.view(-1, fusion_layer4.shape[2], fusion_layer4.shape[3], fusion_layer4.shape[4])
        # deconv_0_1 = self.deconv_0_1(fusion_layer4_)
        # deconv_1_1 = self.deconv_1_1(deconv_0_1)
        # deconv_2_1 = self.deconv_2_1(deconv_1_1)
        # deconv_3_1 = self.deconv_3_1(deconv_2_1)
        # deconv_4_1 = self.deconv_4_1(deconv_3_1)
        # output1 = self.out1(deconv_4_1)
        deconv_0_1 = self.deconv_0_1(fusion_layer4, fusion_layer3)
        deconv_1_1 = self.deconv_1_1(deconv_0_1, fusion_layer2)
        deconv_2_1 = self.deconv_2_1(deconv_1_1, fusion_layer1)
        deconv_3_1 = self.deconv_3_1_(deconv_2_1, fusion_layer0)
        output1 = self.out1(deconv_3_1)
# -----------------------------------infrared_decoder---------------------------------------------------------------------------------
        ir_features5__ = ir_features5__.view(-1, ir_features5__.shape[2], ir_features5__.shape[3], ir_features5__.shape[4])
        # ir_deconv_0_1 = self.deconv_0_1(ir_features5__)
        # ir_deconv_1_1 = self.deconv_1_1(ir_deconv_0_1)
        # ir_deconv_2_1 = self.deconv_2_1(ir_deconv_1_1)
        # ir_deconv_3_1 = self.deconv_3_1(ir_deconv_2_1)
        # ir_deconv_4_1 = self.deconv_4_1(ir_deconv_3_1)
        # ir_output1 = self.out1(ir_deconv_4_1)
        ir_deconv_0_1 = self.deconv_0_1(ir_features5__, ir_features4__)
        ir_deconv_1_1 = self.deconv_1_1(ir_deconv_0_1, ir_features3__)
        ir_deconv_2_1 = self.deconv_2_1(ir_deconv_1_1, ir_features2__)
        ir_deconv_3_1 = self.deconv_3_1_(ir_deconv_2_1, x_ir1)
        ir_output1 = self.out1(ir_deconv_3_1)
# -----------------------------------visible decoder---------------------------------------------------------------------------------
        vis_features5__ = vis_features5__.view(-1, vis_features5__.shape[2], vis_features5__.shape[3], vis_features5__.shape[4])
        # vis_deconv_0_1 = self.deconv_0_1(vis_features5__ )
        # vis_deconv_1_1 = self.deconv_1_1(vis_deconv_0_1)
        # vis_deconv_2_1 = self.deconv_2_1(vis_deconv_1_1)
        # vis_deconv_3_1 = self.deconv_3_1(vis_deconv_2_1)
        # vis_deconv_4_1 = self.deconv_4_1(vis_deconv_3_1)
        # vis_output1 = self.out1(vis_deconv_4_1)

        vis_deconv_0_1 = self.deconv_0_1(vis_features5__, vis_features4__)
        vis_deconv_1_1 = self.deconv_1_1(vis_deconv_0_1, vis_features3__)
        vis_deconv_2_1 = self.deconv_2_1(vis_deconv_1_1, vis_features2__)
        vis_deconv_3_1 = self.deconv_3_1_(vis_deconv_2_1, y_vis1)
        vis_output1 = self.out1(vis_deconv_3_1)

        # total_lb_loss = (0.1 * lb_loss_1 + 0.2 * lb_loss_2 + 0.3 * lb_loss_3 + 0.4 * lb_loss_4)
        # total_sp_loss = (0.1 * sparse_loss_1 + 0.2 * sparse_loss_2 + 0.3 * sparse_loss_3 + 0.4 * sparse_loss_4)


        # print(f"lb_loss_1: {lb_loss_1}, lb_loss_2: {lb_loss_2}, lb_loss_3: {lb_loss_3}, lb_loss_4: {lb_loss_4}")

        # return output1,  ir_output1 , vis_output1, total_lb_loss ,  total_sp_loss,

        return output1, ir_output1, vis_output1,


# class MultiViewResNetInteraction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 创建8个独立的ResNet实例
#         self.resnet_blocks = nn.ModuleList([ModifiedResNet18() for _ in range(8)])
#         # 创建视图交互模块
#         self.view_interaction = ViewAttentionInteraction()
#
#     def forward(self, x_list):
#         """
#         Args:
#             x_list: 包含8个张量的列表，每个张量形状为(bs, 16, 224, 224, 3)
#         Returns:
#             经过视图交互后的最终输出
#         """
#         # # 验证输入列表长度
#         # assert len(x_list) == 8, "输入必须包含8个视角的张量"
#
#         # 并行处理所有视角
#         view_outputs = []
#         for idx, (x, resnet) in enumerate(zip(x_list, self.resnet_blocks)):
#             # 调整维度顺序 (bs, 16, H, W, C) -> (bs*16, C, H, W)
#             x = x.view(-1, *x.shape[-3:]).permute(0, 3, 1, 2).contiguous()
#             # 通过对应的ResNet处理
#             out = resnet(x)
#             view_outputs.append(out)
#
#         # 执行视图间交互
#         final_output = self.view_interaction(view_outputs)
#         return final_output

class MultiViewResNetInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建8个独立的ResNet实例
        self.resnet_blocks = nn.ModuleList([ModifiedResNet18() for _ in range(8)])
        # 创建视图交互模块
        self.view_interaction = MultiViewInteraction(2)

    def forward(self, x_list):
        """
        Args:
            x_list: 包含8个张量的列表，每个张量形状为(bs, 16, 224, 224, 3)
        Returns:
            经过视图交互后的最终输出
        """
        # # 验证输入列表长度
        # assert len(x_list) == 8, "输入必须包含8个视角的张量"

        # 并行处理所有视角
        view_outputs = []
        for idx, (x, resnet) in enumerate(zip(x_list, self.resnet_blocks)):
            # 调整维度顺序 (bs, 16, H, W, C) -> (bs*16, C, H, W)
            x = x.view(-1, *x.shape[-3:]).permute(0, 3, 1, 2).contiguous()
            # 通过对应的ResNet处理
            out = resnet(x)[0] #(bs,c)
            view_outputs.append(out)

        # 执行视图间交互
        final_output = self.view_interaction(view_outputs)
        return final_output

# # 使用示例
if __name__ == "__main__":
   # 初始化模型
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    from torchvision.models import resnet18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备编号:", torch.cuda.current_device())
    print("当前使用的 GPU:", torch.cuda.get_device_name(device))
    model = ModifiedResNet18_weight(32).to(device)
    #模拟输入 (bs=2, 16帧)
    dummy_input1 = torch.randn(2, 16, 256, 256, 3, )
    dummy_input2 = torch.randn(2, 16, 256, 256, 3, )

    # x = [torch.randn(2, 16, 224, 224, 3).to(device) for _ in range(8)]
    # 数据预处理
    x = dummy_input1.view(-1, *dummy_input1.shape[2:]).permute(0, 3, 1, 2).contiguous()  # (32,3, 224, 224)
    y = dummy_input1.view(-1, *dummy_input1.shape[2:]).permute(0, 3, 1, 2).contiguous()  # (32,3, 224, 224)
    print(x.shape)
    # x = x.permute(0, 3, 1, 2).contiguous() #AttributeError: 'list' object has no attribute 'to'
    # print(x.shape)
    x = x.to(device)# (32, 3, 224, 224)
    y = y.to(device)  # (32, 3, 224, 224)
    print("即将喂给模型的 x.shape =", x.shape)
    print("x.device =", x.device)
    print("model.device =", next(model.parameters()).device)
    assert x.shape[1] == 3, f"通道数必须是3，当前{x.shape[1]}"
    out,_,_ = model(x,y)
    print(out.shape)
#     #
#     # # 特征提取
#     # with torch.no_grad():
#     #     features,conf = model(x)  # (32, 64, 56, 56)
#
#     # features,conf = model(x)
#     #
#     # print(features.shape)
#     # print(conf.shape)
#     # 恢复帧维度
#     # features = features.view(-1,16, *features.shape[1:])  # (2, 16, 64, 56, 56)
#     #
#     # print("输出特征维度:", features.shape)
#     # 输出: torch.Size([2, 16, 64, 56, 56])
#
    # # 使用示例
    # batch_size = 32
    # num_classes = 2
    # num_views = 8
    #
    # # 模拟8个不同视角的logits
    # logits_list = [torch.randn(batch_size, num_classes) for _ in range(num_views)]
    #
    # model = MultiViewInteraction(num_classes=num_classes)
    # output = model(logits_list)
    # print(output.shape)  # torch.Size([32, 2])