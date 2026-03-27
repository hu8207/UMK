# %%

from cgitb import reset

from turtle import forward, shape

from numpy import percentile

import torch.nn as nn

import torch.nn.functional as F

from models.NF import MAF

# import tensorflow as tf

import torch

import numpy as np

import scipy

from torch.nn.utils.parametrizations import weight_norm



# class GNN(nn.Module):

#     def __init__(self, input_size, hidden_size, num_layers=2):

#         super(GNN, self).__init__()

#         self.num_layers = num_layers

#         # 输入层

#         self.lin_n = nn.Linear(input_size, hidden_size)

#         self.lin_r = nn.Linear(input_size, hidden_size, bias=False)

#         self.lin_2 = nn.Linear(hidden_size, hidden_size)

#         # 隐藏层

#         self.hidden_layers = nn.ModuleList([

#             nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)

#         ])

#         # 输出层

#         self.lin_out = nn.Linear(hidden_size, hidden_size)

#         # 激活函数

#         self.activation = nn.GELU()

#

#     def forward(self, h, A):

#         # 初始变换

#         h_n = self.lin_n(torch.einsum('nkld,nkj->njld', h, A))

#         h_r = self.lin_r(h[:, :, :-1])

#         h_n[:, :, 1:] += h_r

#

#         # 输出层

#         # h = self.lin_out(h_n)

#         h = self.lin_2(F.relu(h_n))

#         return h



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm





class TemporalConvBlock(nn.Module):



    def __init__(self, hidden_size, dilation=1):

        super().__init__()

        kernel_size = 3

        padding = (kernel_size - 1) * dilation // 2  # 对称填充



        self.conv = weight_norm(

            nn.Conv2d(

                hidden_size,

                hidden_size,

                kernel_size=(1, kernel_size),  # 时间维度卷积

                padding=(0, padding),

                dilation=(1, dilation),

                padding_mode='circular'

            )

        )

        self.norm = nn.LayerNorm(hidden_size)

        self.activation = nn.GELU()



    def forward(self, x):

        # 输入形状: [batch, nodes, time, features]

        identity = x



        # 时间卷积（保持维度）

        x = x.permute(0, 3, 1, 2)  # [batch, feat, nodes, time]

        x = self.conv(x)

        x = x.permute(0, 2, 3, 1)  # [batch, nodes, time, feat]



        # 归一化与残差

        x = self.norm(x + identity)

        return self.activation(x)





# ==========================================

# Algorithm 3: Multi-scale Feature Fusion Layer

# ==========================================



class MultiScaleConv1D(nn.Module):

    """

    并行多尺度1D卷积 (Algorithm 3, Step 2)

    使用不同kernel_size (3, 5, 7) 提取多尺度特征

    """

    def __init__(self, hidden_size, kernel_sizes=[3, 5, 7]):

        super().__init__()

        self.kernel_sizes = kernel_sizes

        self.num_scales = len(kernel_sizes)

        

        # 为每个尺度创建独立的卷积层

        self.convs = nn.ModuleList()

        for k in kernel_sizes:

            padding = k // 2  # 保持长度不变

            self.convs.append(

                nn.Conv1d(

                    in_channels=hidden_size,

                    out_channels=hidden_size,

                    kernel_size=k,

                    padding=padding,

                    padding_mode='circular'

                )

            )

    

    def forward(self, x):

        # x: [batch, nodes, time, features] -> 需要调整维度

        # 转换为: [batch*nodes, features, time] 用于Conv1d

        B, N, T, H = x.shape

        x = x.reshape(B * N, T, H).permute(0, 2, 1)  # [B*N, H, T]

        

        # 并行计算多尺度特征

        features = []

        for conv in self.convs:

            feat = conv(x)  # [B*N, H, T]

            feat = feat.permute(0, 2, 1)  # [B*N, T, H]

            feat = feat.reshape(B, N, T, H)  # [B, N, T, H]

            features.append(feat)

        

        return features  # List of [B, N, T, H] with len=3





class FeatureFusionLayer(nn.Module):

    """

    特征融合层 (Algorithm 3, Steps 3-5)

    Concat -> Conv2D -> GELU -> LayerNorm

    """

    def __init__(self, hidden_size, num_scales=3):

        super().__init__()

        self.hidden_size = hidden_size

        self.num_scales = num_scales

        

        # Step 4: Conv2D降维 (将concat后的3*H维度降回H)

        self.conv2d = nn.Conv2d(

            in_channels=hidden_size * num_scales,

            out_channels=hidden_size,

            kernel_size=1  # 1x1卷积用于特征融合

        )

        

        # Step 5: LayerNorm + GELU

        self.norm = nn.LayerNorm(hidden_size)

        self.activation = nn.GELU()

        

    def forward(self, features):

        # features: List of [B, N, T, H], len=num_scales

        # Step 3: Concat -> [B, N, T, num_scales*H]

        x = torch.cat(features, dim=-1)  # [B, N, T, 3*H]

        

        # 转换为Conv2D格式: [B, 3*H, N, T]

        x = x.permute(0, 3, 1, 2)  # [B, 3*H, N, T]

        

        # Step 4: Conv2D降维 -> [B, H, N, T]

        x = self.conv2d(x)

        

        # 转换回: [B, N, T, H]

        x = x.permute(0, 2, 3, 1)  # [B, N, T, H]

        

        # Step 5: GELU + LayerNorm

        x = self.activation(x)

        x = self.norm(x)

        

        return x  # [B, N, T, H]





class Router(nn.Module):

    """

    Router网络 (Algorithm 3, Step 8)

    学习专家权重: R^{B*L} 

    """

    def __init__(self, hidden_size, num_experts, reduction=4):

        super().__init__()

        self.hidden_size = hidden_size

        self.num_experts = num_experts

        

        # 使用MLP生成路由权重

        self.router = nn.Sequential(

            nn.Linear(hidden_size, hidden_size // reduction),

            nn.ReLU(),

            nn.Linear(hidden_size // reduction, num_experts)

        )

        self.softmax = nn.Softmax(dim=-1)

    

    def forward(self, x):

        # x: [B, N, T, H]

        B, N, T, H = x.shape

        

        # 全局平均池化获取全局特征

        global_feat = x.mean(dim=[1, 2])  # [B, H]

        

        # 生成路由权重

        logits = self.router(global_feat)  # [B, num_experts]

        weights = self.softmax(logits)  # [B, num_experts], sum=1

        

        return weights  # [B, num_experts]





class Expert(nn.Module):

    """

    单个专家网络 (Algorithm 3, Step 9)

    Linear -> GELU -> LayerNorm

    """

    def __init__(self, hidden_size):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(hidden_size, hidden_size),

            nn.GELU(),

            nn.LayerNorm(hidden_size)

        )

    

    def forward(self, x):

        # x: [B, N, T, H]

        B, N, T, H = x.shape

        x_flat = x.reshape(B * N * T, H)  # [B*N*T, H]

        out = self.net(x_flat)  # [B*N*T, H]

        return out.reshape(B, N, T, H)  # [B, N, T, H]





class ExpertHybridMechanism(nn.Module):

    """

    专家混合机制 (Algorithm 3, Steps 6-11)

    Reshape -> Router -> Experts -> Weighted Sum -> Softmax

    """

    def __init__(self, hidden_size, num_experts=4):

        super().__init__()

        self.hidden_size = hidden_size

        self.num_experts = num_experts

        

        # Router生成专家权重

        self.router = Router(hidden_size, num_experts)

        

        # 多个并行专家

        self.experts = nn.ModuleList([

            Expert(hidden_size) for _ in range(num_experts)

        ])

        

        # Softmax归一化 (Algorithm 3, Step 11)

        self.softmax = nn.Softmax(dim=-1)

    

    def forward(self, x):

        # x: [B, N, T, H] (F_concat_expert after reshape)

        B, N, T, H = x.shape

        

        # Step 8: Router生成权重 [B, num_experts]

        router_weights = self.router(x)  # [B, num_experts]

        

        # Step 9: 每个专家处理 (并行计算)

        expert_outputs = []

        for expert in self.experts:

            expert_out = expert(x)  # [B, N, T, H]

            expert_outputs.append(expert_out)

        

        # 堆叠专家输出: [B, N, T, H, num_experts]

        expert_stack = torch.stack(expert_outputs, dim=-1)

        

        # 加权求和: [B, num_experts] -> [B, 1, 1, 1, num_experts]

        weights = router_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        

        # Step 10: F_expert_gelu = sum(W_router[l] * F_expert_l)

        weighted_sum = (expert_stack * weights).sum(dim=-1)  # [B, N, T, H]

        

        # LayerNorm

        weighted_sum = nn.functional.layer_norm(weighted_sum, weighted_sum.shape[-1:])

        

        # Step 11: Softmax -> [B, N, T, H]

        # 注意: Algorithm 3中的softmax可能是对某个维度，这里按论文实现

        output = self.softmax(weighted_sum)

        

        return output, router_weights  # 返回输出和路由权重(用于分析)





class MultiScaleFeatureFusionWithMoE(nn.Module):

    """

    完整的多尺度特征融合层 (Algorithm 3, Steps 1-11)

    整合Multi-scale Conv1D + Feature Fusion + Router + Expert Hybrid

    """

    def __init__(self, hidden_size, kernel_sizes=[3, 5, 7], num_experts=4):

        super().__init__()

        self.hidden_size = hidden_size

        

        # Step 1: Linear (已经在GCN之前完成，这里主要是多尺度处理)

        

        # Step 2: Multi-scale Conv1D

        self.multi_scale_conv = MultiScaleConv1D(hidden_size, kernel_sizes)

        

        # Steps 3-5: Feature Fusion (Concat + Conv2D + GELU + LayerNorm)

        self.feature_fusion = FeatureFusionLayer(hidden_size, num_scales=len(kernel_sizes))

        

        # Steps 6-11: Expert Hybrid Mechanism

        self.expert_hybrid = ExpertHybridMechanism(hidden_size, num_experts)

        

    def forward(self, h_gcn):

        # h_gcn: [B, N, T, H] 来自GCN的输出

        

        # Step 2: 多尺度卷积

        multi_scale_features = self.multi_scale_conv(h_gcn)  # List of [B, N, T, H]

        

        # Steps 3-5: 特征融合

        fused_features = self.feature_fusion(multi_scale_features)  # [B, N, T, H]

        

        # Step 6: Reshape (在算法中标记为 F_concat_expert)

        # 这里fused_features已经是 [B, N, T, H]，可以作为Expert输入

        

        # Steps 7-11: 专家混合机制

        output, router_weights = self.expert_hybrid(fused_features)  # [B, N, T, H]

        

        # 残差连接 (与原始gcn_mlp保持一致)

        output = output + h_gcn

        

        return output, router_weights





# ==========================================

# End of Algorithm 3 Implementation

# ==========================================





class GNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):

        super(GNN, self).__init__()

        self.num_layers = num_layers



        # 空间传播层

        self.spatial_convs = nn.ModuleList([

            # nn.Linear(input_size if i == 0 else hidden_size, hidden_size)

            nn.Linear(hidden_size, hidden_size)

            for i in range(num_layers)

        ])



        # 时间卷积层

        self.temporal_convs = nn.ModuleList([

            TemporalConvBlock(hidden_size, dilation=2 ** i)

            for i in range(num_layers)

        ])



        # 残差参数

        self.alpha = nn.Parameter(torch.ones(num_layers))

        self.norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(0.1)



    def forward(self, h, A):

        # 输入形状: [batch, nodes, time, features]

        batch, nodes, time, _ = h.shape



        for i, (s_conv, t_conv) in enumerate(zip(self.spatial_convs, self.temporal_convs)):

            # 空间聚合

            h_spatial = torch.einsum('nkld,nkj->njld', h, A)  # 空间传播

            h_spatial = s_conv(h_spatial)  # 特征变换



            # 时间卷积

            h_temporal = t_conv(h)



            # 残差融合

            h = self.norm(h + self.alpha[i] * h_spatial + h_temporal)

            h = self.dropout(h)



        return h





class STGNN(GNN):

    """时空统一接口（兼容原始调用）"""



    def __init__(self, input_size, hidden_size, num_layers=2):

        super().__init__(input_size, hidden_size, num_layers)

        # 输出适配层

        self.fc_out = nn.Linear(hidden_size, hidden_size)



    def forward(self, h, A):

        h = super().forward(h, A)

        return self.fc_out(h)





import math

import torch.nn as nn

import matplotlib.pyplot as plt





def interpolate(tensor, index, target_size, mode='nearest', dim=0):

    print(tensor.shape)

    source_length = tensor.shape[dim]

    if source_length > target_size:

        raise AttributeError('no need to interpolate')

    if dim == -1:

        new_tensor = torch.zeros((*tensor.shape[:-1], target_size), dtype=tensor.dtype, device=tensor.device)

    if dim == 0:

        new_tensor = torch.zeros((target_size, *tensor.shape[1:],), dtype=tensor.dtype, device=tensor.device)

    scale = target_size // source_length

    reset = target_size % source_length

    # if mode == 'nearest':

    new_index = index

    new_tensor[new_index, :] = tensor

    new_tensor[:new_index[0], :] = tensor[0, :].unsqueeze(0)

    for i in range(source_length - 1):

        new_tensor[new_index[i]:new_index[i + 1], :] = tensor[i, :].unsqueeze(0)

    new_tensor[new_index[i + 1]:, :] = tensor[i + 1, :].unsqueeze(0)

    return new_tensor





def plot_attention(data, i, X_label=None, Y_label=None):

    '''

      Plot the attention model heatmap

      Args:

        data: attn_matrix with shape [ty, tx], cutted before 'PAD'

        X_label: list of size tx, encoder tags

        Y_label: list of size ty, decoder tags

    '''

    fig, ax = plt.subplots(figsize=(20, 8))  # set figure size

    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)

    fig.colorbar(heatmap)

    # Set axis labels

    if X_label != None and Y_label != None:

        X_label = [x_label for x_label in X_label]

        Y_label = [y_label for y_label in Y_label]



        xticks = range(0, len(X_label))

        ax.set_xticks(xticks, minor=False)  # major ticks

        ax.set_xticklabels(X_label, minor=False, rotation=45)  # labels should be 'unicode'



        yticks = range(0, len(Y_label))

        ax.set_yticks(yticks, minor=False)

        ax.set_yticklabels(Y_label[::-1], minor=False)  # labels should be 'unicode'



        ax.grid(True)

        plt.show()

        plt.savefig('graph/attention{:04d}.jpg'.format(i))





class ScaleDotProductAttention(nn.Module):

    """

    compute scale dot product attention



    Query : given sentence that we focused on (decoder)

    Key : every sentence to check relationship with Qeury(encoder)

    Value : every sentence same with Key (encoder)

    """



    def __init__(self, c):

        super(ScaleDotProductAttention, self).__init__()

        self.w_q = nn.Linear(c, c)

        self.w_k = nn.Linear(c, c)

        self.w_v = nn.Linear(c, c)

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.2)



    def forward(self, x, mask=None, e=1e-12):

        # input is 4 dimension tensor

        # [batch_size, head, length, d_tensor]

        shape = x.shape

        x_shape = x.reshape((shape[0], shape[1], -1))

        batch_size, length, c = x_shape.size()

        q = self.w_q(x_shape)

        k = self.w_k(x_shape)

        k_t = k.view(batch_size, c, length)  # transpose

        score = (q @ k_t) / math.sqrt(c)  # scaled dot product



        # 2. apply masking (opt)

        if mask is not None:

            score = score.masked_fill(mask == 0, -1e9)



        # 3. pass them softmax to make [0, 1] range

        score = self.dropout(self.softmax(score))



        return score, k





class MTGFLOWZL(nn.Module):



    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",

                 batch_norm=True):

        super(MTGFLOWZL, self).__init__()

        # 1D卷积层

        # self.cnn = nn.Sequential(

        # nn.Conv1d(in_channels=1, out_channels = input_size, kernel_size = 3, padding = 1),

        # nn.ReLU(),

        # # nn.MaxPool1d(kernel_size=2)

        # )

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)

        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)



        if model == "MAF":

            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')

            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,

                          batch_norm=batch_norm, activation='tanh')



        self.gcn_mlp = MultiScaleFeatureFusionWithMoE(

            hidden_size=hidden_size,

            kernel_sizes=[3, 5, 7],

            num_experts=4

        )



        self.ann = nn.Sequential(

            nn.Linear(n_sensor * window_size * 32, 100),

            nn.ReLU(),

            nn.BatchNorm1d(100),

        )

        self.attention = ScaleDotProductAttention(window_size * input_size)

        self.proj_main = nn.Sequential(

            nn.Linear(hidden_size, 128),

            nn.ReLU(),

            nn.LayerNorm(128)

        )



        # 对比分支投影头（轻量化设计）

        self.proj_cont = nn.Sequential(

            nn.Linear(hidden_size, 128),

            nn.ReLU(),

            nn.LayerNorm(128)

        )



        # 动量更新编码器（稳定对比学习）

        self.proj_cont_momentum = nn.Sequential(

            nn.Linear(hidden_size, 128),

            nn.ReLU(),

            nn.LayerNorm(128)

        )

        self._init_momentum_encoder()



    def _init_momentum_encoder(self):

        """初始化动量编码器参数"""

        for param, param_m in zip(self.proj_cont.parameters(),

                                  self.proj_cont_momentum.parameters()):

            param_m.data.copy_(param.data)

            param_m.requires_grad = False



    def update_momentum_encoder(self, m=0.999):

        """动量更新规则"""

        with torch.no_grad():

            for param, param_m in zip(self.proj_cont.parameters(),

                                      self.proj_cont_momentum.parameters()):

                param_m.data = param_m.data * m + param.data * (1. - m)



    # def forward(self, x, ):

    def forward(self, x):

        # 解包所有5个返回值

        ann_out, log_prob, h_gcn, z_main, z_cont = self.test(x)



        # 计算对比损失

        contrastive_loss = F.cosine_similarity(z_main, z_cont.detach()).mean()

        total_loss = log_prob.mean() - 0.1 * contrastive_loss  # 加权求和



        # 返回结果（保持与原始调用兼容）

        return ann_out, total_loss, h_gcn, log_prob



        # hid, log_prob, h_gcn = self.test(x, )

        # return hid, log_prob.mean(), h_gcn, log_prob



    def test(self, x, ):

        # x: N  K  L  D

        # import pdb; pdb.set_trace()

        full_shape = x.shape

        graph, _ = self.attention(x)

        self.graph = graph



        # reshape: N*K, L, D

        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # # reshape: N*K, D, L

        # x = x.permute(0, 2, 1)

        # x = self.cnn(x)

        # #reshape: N*K, L, D

        # x = x.permute(0, 2, 1)

        # rnn处理

        h, _ = self.rnn(x)



        # resahpe: N, K, L, H

        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h_gcn = self.gcn(h, graph)

        # import pdb; pdb.set_trace()



        # 使用新的多尺度特征融合层 (Algorithm 3)

        shape_hgcn = h_gcn.shape

        h_gcn, router_weights = self.gcn_mlp(h_gcn)  # 返回融合后的特征和路由权重

        h_gcn_for_ann = h_gcn.reshape(full_shape[0], -1)



        # reshappe N*K*L,H

        h_gcn_reshape = h_gcn.reshape((-1, h_gcn.shape[3]))

        x = x.reshape((-1, full_shape[3]))



        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h_gcn_reshape).reshape([full_shape[0], -1])

        log_prob = log_prob.mean(dim=1)



        # 主分支投影 (用于MAF)

        z_main = self.proj_main(h_gcn_reshape.detach())  # [B*K*L, 128]



        # 对比分支投影

        with torch.no_grad():  # 停止对比分支梯度传播

            z_cont = self.proj_cont_momentum(h_gcn_reshape)  # 动量编码



        return self.ann(h_gcn_for_ann), log_prob, h_gcn, z_main, z_cont

        # return self.ann(h_gcn_for_ann), log_prob, h_gcn



    def get_graph(self):

        return self.graph



    def locate(self, x, ):



        # x: N X K X L X D

        full_shape = x.shape



        graph, _ = self.attention(x)

        # reshape: N*K, L, D

        self.graph = graph



        # reshape: N*K, L, D

        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # reshape: N*K, D, L

        x = x.permute(0, 2, 1)

        x = self.cnn(x)

        # reshape: N*K, L, D

        x = x.permute(0, 2, 1)

        # rnn处理

        h, _ = self.rnn(x)



        # resahpe: N, K, L, H

        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, graph)



        # reshappe N*K*L,H

        h = h.reshape((-1, h.shape[3]))

        x = x.reshape((-1, full_shape[3]))

        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)

        import pdb

        pdb.set_trace()



        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])



        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))



    def _DistanceSquared(self, x, y):

        m, n = x.size(0), y.size(0)

        # Normalize inputs to prevent large values

        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)

        y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)

        xx = torch.pow(x_norm, 2).sum(1, keepdim=True).expand(m, n)

        yy = torch.pow(y_norm, 2).sum(1, keepdim=True).expand(n, m).t()

        dist = xx + yy

        dist = torch.addmm(dist, mat1=x_norm, mat2=y_norm.t(), beta=1, alpha=-2)

        dist = dist.clamp(min=1e-6, max=1e6)



        return dist



    # Cache for gamma values to avoid repeated computation

    _gamma_cache = {}



    def _CalGamma(self, v):

        if v not in self._gamma_cache:

            a = scipy.special.gamma((v + 1) / 2)

            b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)

            self._gamma_cache[v] = a / b

        return self._gamma_cache[v]



    def _TwowaydivergenceLoss(self, P_, Q_, select=None):

        EPS = 1e-7

        # Clamp values to prevent numerical instability

        P_ = torch.clamp(P_, EPS, 1 - EPS)

        Q_ = torch.clamp(Q_, EPS, 1 - EPS)

        losssum1 = P_ * torch.log(Q_ + EPS)

        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)

        losssum = -1 * (losssum1 + losssum2)

        # Clamp final loss to prevent explosion

        return torch.clamp(losssum.mean(), max=100.0)



    # Pre-computed constant

    _two_pi = 2 * 3.14159265359



    def _Similarity(self, dist, gamma, v=100, h=1, pow=2):

        dist_rho = dist.clamp(min=0, max=1e6)  # Prevent extreme distances

        Pij = gamma * self._two_pi * gamma * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))

        # Clamp similarity to valid probability range

        return torch.clamp(Pij, min=1e-7, max=1.0 - 1e-7)



    # 模型集成：训练多个模型并进行集成，以提高预测的稳定性和准确性。

    def ensemble_predict(self, input_data):

        predictions = [model(input_data) for model in self]

        return torch.mean(torch.stack(predictions), dim=0)



    # Cache for eye masks

    _eye_cache = {}



    def LossManifold(

            self,

            input_data,

            latent_data,

            v_input,

            v_latent,

            w=1,

            metric="euclidean",

            label=None,

    ):

        batch_size = input_data.shape[0]

        half_batch = batch_size // 2



        data_1 = input_data[:half_batch]

        dis_P = self._DistanceSquared(data_1, data_1)



        latent_data_1 = latent_data[:half_batch]

        latent_data_2 = latent_data[half_batch:]



        # Cache gamma values

        gamma_input = self._CalGamma(v_input)

        gamma_latent = self._CalGamma(v_latent)



        P_2 = self._Similarity(dist=dis_P, gamma=gamma_input, v=v_input)

        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)

        Q_2 = self._Similarity(dist=dis_Q_2, gamma=gamma_latent, v=v_latent)



        # Cache eye mask per size

        mask_size = P_2.shape[0]

        cache_key = (mask_size, input_data.device)

        if cache_key not in self._eye_cache:

            self._eye_cache[cache_key] = torch.eye(mask_size, device=input_data.device)

        eye_mask = self._eye_cache[cache_key]



        # Use diagonal and off-diagonal directly

        diag_mask = eye_mask.bool()

        loss_ce_posi = self._TwowaydivergenceLoss(P_=P_2[diag_mask], Q_=Q_2[diag_mask])

        loss_ce_nega = self._TwowaydivergenceLoss(P_=P_2[~diag_mask], Q_=Q_2[~diag_mask])



        w1, w2 = 1 / (1 + w), w / (1 + w)

        return w2 * loss_ce_nega, w1 * loss_ce_posi / batch_size





class test(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",

                 batch_norm=True):

        super(test, self).__init__()



        if model == "MAF":

            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, batch_norm=batch_norm,

                          activation='tanh', mode='zero')

        self.attention = ScaleDotProductAttention(window_size * input_size)



    def forward(self, x, ):

        return self.test(x, ).mean()



    def test(self, x):

        x = x.unsqueeze(2).unsqueeze(3)

        full_shape = x.shape

        x = x.reshape((full_shape[0] * full_shape[1], full_shape[2], full_shape[3]))

        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2]).reshape(

            [full_shape[0], full_shape[1], -1])  # *full_shape[1]*full_shape[2]

        log_prob = log_prob.mean(dim=1)

        return log_prob



    def locate(self, x, ):

        # x: N X K X L X D

        x = x.unsqueeze(2).unsqueeze(3)

        full_shape = x.shape



        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))



        # reshappe N*K*L,H

        x = x.reshape((-1, full_shape[3]))

        a = self.nf.log_prob(x, full_shape[1], full_shape[2])  # *full_shape[1]*full_shape[2]

        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])



        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))

