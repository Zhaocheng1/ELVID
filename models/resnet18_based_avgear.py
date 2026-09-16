import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

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

#########################多视图中间特征注意力融合##########################################
# class ViewAttentionInteraction(nn.Module):
#     def __init__(self, in_channels, num_views=8):
#         super().__init__()
#         self.num_views = num_views
#
#         # 视角注意力权重生成（保持最后维度连续性）
#         self.view_att = nn.Sequential(
#             nn.Linear(in_channels, num_views),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, features):
#         bs, t, c, h, w = features[0].shape
#         device = features[0].device
#
#         # 合并维度时确保连续性
#         merged_features = [f.view(bs*t, c, h, w).contiguous() for f in features]
#
#         # 计算全局特征描述符
#         gap_features = torch.stack([
#             F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1).contiguous()  # (bs*t, c)
#             for f in merged_features
#         ])  # (num_views, bs*t, c)
#
#         # 计算视角注意力权重（添加连续性保证）
#         view_scores = gap_features.mean(dim=-1).permute(1,0).contiguous()  # (bs*t, num_views)
#         view_weights = self.view_att(view_scores)  # (bs*t, num_views)
#
#         # 加权融合（保持内存连续性）
#         fused_feature = torch.zeros_like(merged_features[0])
#         for i in range(self.num_views):
#             # 确保权重张量连续
#             weight = view_weights[:, i].contiguous().view(bs*t, 1, 1, 1)
#             fused_feature.add_(merged_features[i] * weight)
#
#         # 恢复维度时保持连续性
#         return fused_feature.view(bs, t, c, h, w).contiguous()

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
class ModifiedResNet18_1(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练模型并提取所需层
        original = resnet18(pretrained=True)

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

        self.tem_branch2 = FusionTemporalBlock(64)
        self.spat_branch2 = EchocardiographySpatialAttentionBlock(64)
        self.tem_branch3 = FusionTemporalBlock(128)
        self.spat_branch3 = EchocardiographySpatialAttentionBlock(128)
        self.tem_branch4 = FusionTemporalBlock(256)
        self.spat_branch4 = EchocardiographySpatialAttentionBlock(256)
        self.tem_branch5 = FusionTemporalBlock(512)
        self.spat_branch5 = EchocardiographySpatialAttentionBlock(512)
        # # 冻结所有参数（可选）
        # for param in self.parameters():
        #     param.requires_grad_(False)
        # 加载预训练权重（自动处理）
        self._load_pretrained_weights(original)

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

    def forward(self, x):
        # 标准前向传播流程
        #32,3,224,224
        x = self.conv1(x)#32,64,112,112
        x = self.bn1(x)#32,64,112,112
        x = self.relu(x)#32,64,112,112
        x = self.maxpool(x)#32,64,56,56

        feature2 = self.layer1(x) #(bs*16,64,56,56)
        feature2_ = feature2.view(-1,16, *feature2.shape[1:]) #(bs,16,64,56,56)(2,16,64,56,56)
        x_tem, confidence_tem, _, _, _ = self.tem_branch2(feature2_)
        x_spat, conf_spat, attn_maps_all_spat = self.spat_branch2(feature2_)
        features2__ = x_tem + x_spat + feature2_ #要不要加feature2？(bs,16,64,56,56)

        feature3 = self.layer2(features2__.view(-1, *features2__.shape[-3:])) #输入(bs*16,64,56,56) 输出(bs*16,128,28,28)
        feature3_ = feature3.view(-1, 16, *feature3.shape[1:]) #(bs,16,128,28,28)
        x_tem, confidence_tem, _, _, _ = self.tem_branch3(feature3_)
        x_spat, conf_spat, attn_maps_all_spat = self.spat_branch3(feature3_)
        features3__ = x_tem + x_spat + feature3_ #(bs,16,128,28,28)

        features3__ = self.dropout1(features3__.view(-1, *features3__.shape[-3:]))
        features3__ = features3__.view(-1, 16, 128, 28, 28)

        feature4 = self.layer3(features3__.view(-1, *features3__.shape[-3:])) #输入(bs*16,128,28,28) 输出(bs*16,256,14,14)
        feature4_ = feature4.view(-1, 16, *feature4.shape[1:]) #(bs,16,256,14,14)
        x_tem, confidence_tem, _, _, _ = self.tem_branch4(feature4_)
        x_spat, conf_spat, attn_maps_all_spat = self.spat_branch4(feature4_)
        features4__ = x_tem + x_spat + feature4_ #(bs,16,256,14,14)

        feature5 = self.layer4(features4__.view(-1, *features4__.shape[-3:])) #输入(bs*16,256,14,14) 输出(bs*16,512,7,7)
        feature5_ = feature5.view(-1, 16, *feature5.shape[1:]) #(bs,16,512,7,7)
        x_tem, confidence_tem, _, _, _ = self.tem_branch5(feature5_)
        x_spat, conf_spat, attn_maps_all_spat = self.spat_branch5(feature5_)
        features5__ = x_tem + x_spat + feature5_ #(bs,16,512,7,7)

        features5__ = self.dropout2(features5__.view(-1, *features5__.shape[-3:]))  # Dropout 2d
        features5__ = features5__.view(-1, 16, 512, 7, 7)

        features6 = self.avgpool(features5__.view(-1, *features5__.shape[-3:])) #输入(bs*16,512,7,7) 输出(bs*16,512,1,1)
        features6 = self.dropout2(features6)#32,512,1,1
        features7 = self.fc(features6.squeeze(-1).squeeze(-1)).view(-1,16,2) #(bs,16,2)#2,16,2
#----------------------------------------------------------------------------------------------
        # # 压缩置信度维度并归一化
        # # confidence_squeezed = confidence_tem.squeeze(dim=(2, 3, 4))  # (bs, 16)
        # confidence_squeezed = confidence_tem.squeeze(2).squeeze(2).squeeze(2)#2,16#获取每帧置信度,使用 softmax 得到每一帧的重要性权重，越重要的帧，权重大。
        # weights = torch.softmax(confidence_squeezed, dim=1)  # 沿帧维度归一化#2,16
        #
        # # 维度扩展并加权求和
        # output = (features7 * weights.unsqueeze(-1)).sum(dim=1)  # (bs, num_classes)#2,2对每帧预测结果做加权求和,表示：最终视频的预测结果，是 所有帧预测结果的加权平均。
#---------------------------------------------------------------------------------------------------
        # confidence_squeezed = confidence_tem.squeeze(2).squeeze(2).squeeze(2)  # 2,16#获取每帧置信度,使用 softmax 得到每一帧的重要性权重，越重要的帧，权重大。
        # #添加一个“top-k 帧筛选策略”,
        # B, T = confidence_squeezed.shape
        # topk = 4  # 可调参数
        #
        # # 获取 top-k 帧的索引和权重
        # topk_values, topk_indices = torch.topk(confidence_squeezed, topk, dim=1)  # shape: (B, k)
        #
        # # 归一化权重
        # topk_weights = torch.softmax(topk_values, dim=1)  # shape: (B, k)
        #
        # # 从 features7 中选择 top-k 帧的 logits
        # # features7: (B, T, num_classes)“从每个样本的16帧中提取 top-k 帧的 logits（形状为 (B, k, C)）
        # topk_features = torch.gather(
        #     features7, dim=1,
        #     index=topk_indices.unsqueeze(-1).expand(-1, -1, features7.shape[-1])
        # )  # shape: (B, k, num_classes)
        #
        # # 加权求和得到最终输出
        # output = (topk_features * topk_weights.unsqueeze(-1)).sum(dim=1)  # (B, num_classes)
 # ---------------------------------------------------------------------------------------------------
        confidence_squeezed = confidence_tem.squeeze(2).squeeze(2).squeeze(2)  # 2,16#获取每帧置信度,使用 softmax 得到每一帧的重要性权重，越重要的帧，权重大。
        B, T = confidence_squeezed.shape
        min_k = 8  # 最少保留帧数
        top_k_for_thresh = 8  # 用前 top-5 置信度均值作为动态阈值

        device = confidence_squeezed.device
        output = torch.zeros(B, features7.shape[-1], device=device)  # (B, num_classes)

        for b in range(B):
            conf_b = confidence_squeezed[b]  # (16,)
            # print(conf_b)
            feat_b = features7[b]  # (16, num_classes)

            # 计算动态阈值（前 top-k 平均）
            topk_values, _ = torch.topk(conf_b, top_k_for_thresh)
            dynamic_threshold = topk_values.mean()

            # 找出高置信度帧
            high_conf_indices = (conf_b > dynamic_threshold ).nonzero(as_tuple=False).squeeze(-1)

            if high_conf_indices.numel() < min_k:
                # 回退：保留置信度前 min_k 帧
                _, topk_indices = torch.topk(conf_b, min_k)
                selected_indices = topk_indices
            else:
                selected_indices = high_conf_indices

            selected_feats = feat_b[selected_indices]  # (k, num_classes)
            selected_confs = conf_b[selected_indices]  # (k,)

            weights = torch.softmax(selected_confs, dim=0)  # (k,)
            output[b] = (selected_feats * weights.unsqueeze(-1)).sum(dim=0)  # (num_classes,)
        return output, confidence_tem #conf torch.Size([bs, 16, 1, 1, 1])


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModifiedResNet18().to(device)

    # model = MultiViewResNetInteraction().to(device)
    # resnet18 = resnet18(pretrained=True).to(device)
    # # 检查预训练权重加载情况
    # print("layer3权重是否匹配:",
    #       torch.allclose(model.layer3[0].conv1.weight.data,
    #                      resnet18.layer3[0].conv1.weight.data))  # 应为True

    # print("自定义层是否可训练:",
    #       model.fc.weight.requires_grad)  # 应为True
    # print("自定义层是否可训练:",
    #       model.tem_branch2.weight.requires_grad)  # 应为True AttributeError: 'FusionTemporalBlock' object has no attribute 'weight'
    # print("自定义层是否可训练:",
    #       model.spat_branch3.weight.requires_grad)  # 应为True

    #模拟输入 (bs=2, 16帧)
    dummy_input = torch.randn(2, 16, 224, 224,3, )

    # x = [torch.randn(2, 16, 224, 224, 3).to(device) for _ in range(8)]
    # 数据预处理
    x = dummy_input.view(-1, *dummy_input.shape[2:]).permute(0, 3, 1, 2).contiguous()  # (32,3, 224, 224)
    print(x.shape)
    # x = x.permute(0, 3, 1, 2).contiguous() #AttributeError: 'list' object has no attribute 'to'
    # print(x.shape)
    x = x.to(device)# (32, 3, 224, 224)
    print("即将喂给模型的 x.shape =", x.shape)
    print("x.device =", x.device)
    print("model.device =", next(model.parameters()).device)
    assert x.shape[1] == 3, f"通道数必须是3，当前{x.shape[1]}"
    out,_ = model(x)
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