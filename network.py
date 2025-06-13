from random import random
import torch
import torch.nn as nn
import random
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

manualseed = 64
random.seed(manualseed)
np.random.seed(manualseed)
torch.manual_seed(manualseed)
torch.cuda.manual_seed(manualseed)
cudnn.deterministic = True


class GraphAttentionLayer(nn.Module):
    """图注意力层实现"""

    def __init__(self, input_dim, output_dim, dropout=0.3, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, text, er_features):
        # text: [batch_size, input_dim]
        # er_features: [batch_size, 5, input_dim]

        batch_size = text.size(0)
        num_neighbors = er_features.size(1)  # 5

        # 线性变换
        text_feat = torch.mm(text, self.W)  # [batch_size, output_dim]
        er_feats = torch.matmul(er_features, self.W)  # [batch_size, 5, output_dim]

        # 扩展text特征用于计算注意力
        text_expanded = text_feat.unsqueeze(1).repeat(1, num_neighbors, 1)  # [batch_size, 5, output_dim]

        # 拼接特征计算注意力系数
        concat_features = torch.cat([text_expanded, er_feats], dim=2)  # [batch_size, 5, 2*output_dim]
        e = self.leakyrelu(torch.matmul(concat_features, self.a).squeeze(2))  # [batch_size, 5]

        # 注意力权重归一化
        attention = F.softmax(e, dim=1)  # [batch_size, 5]
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 加权聚合邻居特征
        aggregated = torch.bmm(attention.unsqueeze(1), er_feats).squeeze(1)  # [batch_size, output_dim]

        # 中心节点与聚合特征结合
        zed = F.elu(text_feat + aggregated)  # [batch_size, output_dim]

        return zed


class UnimodalDetection(nn.Module):
        def __init__(self, shared_dim=256, prime_dim = 64):
            super(UnimodalDetection, self).__init__()
            
            self.text_uni = nn.Sequential(
                nn.Linear(1280, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

            self.image_uni = nn.Sequential(
                nn.Linear(1512, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

        def forward(self, text_encoding, image_encoding):
            text_prime = self.text_uni(text_encoding)
            image_prime = self.image_uni(image_encoding)
            return text_prime, image_prime

class CrossModule(nn.Module):
    def __init__(
            self,
            corre_out_dim=64):
        super(CrossModule, self).__init__()
        self.corre_dim = 1024
        self.c_specific = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        correlation = torch.cat((text, image),1)
        
        correlation_out = self.c_specific(correlation.float())
        return correlation_out


# class MultiModal(nn.Module):
#     def __init__(
#             self,
#             feature_dim = 64,
#             h_dim = 64
#             ):
#         super(MultiModal, self).__init__()
#         self.weights = nn.Parameter(torch.rand(13, 1))
#         #SENET
#         self.senet = nn.Sequential(
#                 nn.Linear(3, 3),
#                 nn.GELU(),
#                 nn.Linear(3, 3),
#         )
#         self.sigmoid = nn.Sigmoid()
#
#         self.w = nn.Parameter(torch.rand(1))
#         self.b = nn.Parameter(torch.rand(1))
#
#         self.avepooling =  nn.AvgPool1d(64, stride=1)
#         self.maxpooling =  nn.MaxPool1d(64, stride=1)
#
#         self.resnet101 = torchvision.models.resnet101(pretrained=True).cuda()
#
#         self.uni_repre = UnimodalDetection()
#         self.cross_module = CrossModule()
#         self.classifier_corre = nn.Sequential(
#             nn.Linear(feature_dim, h_dim),
#             nn.BatchNorm1d(h_dim),
#             nn.ReLU(),
#             nn.Linear(h_dim, h_dim),
#             nn.BatchNorm1d(h_dim),
#             nn.ReLU(),
#             nn.Linear(h_dim, 2)
#         )
#     def forward(self, input_ids, all_hidden_states, image_raw, text, image):
#         # Process image
#         image_raw = self.resnet101(image_raw)
#
#         # Process text
#         ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, input_ids.shape[0], 1, 768)
#         atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
#         atten = F.softmax(atten.view(-1), dim=0)
#         text_raw = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
#
#         # Unimodal processing
#         text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw, image], 1))
#
#         # Cross-modal processing
#         correlation = self.cross_module(text, image)
#
#         # Calculate similarity weights
#         sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
#         sim = sim * self.w + self.b
#         mweight = sim.unsqueeze(1)
#
#         # Apply correlation weights
#         correlation = correlation * mweight
#
#         # Combine features
#         final_feature = torch.cat([text_prime.unsqueeze(1), image_prime.unsqueeze(1), correlation.unsqueeze(1)], 1)
#
#         # Pooling and transformation
#         s1 = self.avepooling(final_feature)
#         s2 = self.maxpooling(final_feature)
#         s1 = s1.view(s1.size(0), -1)
#         s2 = s2.view(s2.size(0), -1)
#         s1 = self.senet(s1)
#         s2 = self.senet(s2)
#         s = self.sigmoid(s1 + s2)
#         s = s.view(s.size(0), s.size(1), 1)
#
#         # Apply pooling weights
#         final_feature = s * final_feature
#
#         # Classification
#         pre_label = self.classifier_corre(final_feature[:, 0, :] + final_feature[:, 1, :] + final_feature[:, 2, :])
#
#         return pre_label


class MultiModal(nn.Module):
    def __init__(
            self,
            feature_dim=64,
            h_dim=64,
            text_dim=1280  # 新增：text特征维度
    ):
        super(MultiModal, self).__init__()
        self.weights = nn.Parameter(torch.rand(13, 1))

        # 图注意力机制
        self.graph_attn = GraphAttentionLayer(
            input_dim=text_dim,  # 与text特征维度一致
            output_dim=text_dim  # 保持维度不变
        )

        # 新增: ER特征处理模块
        self.er_proj = nn.Sequential(
            nn.Linear(text_dim * 5, text_dim * 5),  # 保持原始大小
            nn.BatchNorm1d(text_dim * 5),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # SENET (修改为4通道融合，包含ZED)
        self.senet = nn.Sequential(
            nn.Linear(4, 4),  # 从3通道改为4通道
            nn.GELU(),
            nn.Linear(4, 4),
        )
        self.sigmoid = nn.Sigmoid()

        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

        self.avepooling = nn.AvgPool1d(64, stride=1)
        self.maxpooling = nn.MaxPool1d(64, stride=1)

        self.resnet101 = torchvision.models.resnet101(pretrained=True).cuda()

        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, input_ids, all_hidden_states, image_raw, text, image, er):
        # Process image
        image_raw = self.resnet101(image_raw)

        # Process text
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, input_ids.shape[0], 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        text_raw = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])

        # 处理ER特征
        # er_projected = self.er_proj(er)  # [batch_size, text_dim*5]
        er_projected = torch.full_like(text_raw)
        er_features = torch.stack((er_projected,er_projected,er_projected,er_projected,er_projected),dim=1)

        # 重塑ER为5个text_dim大小的特征
        # batch_size = er_projected.size(0)
        # er_features = er_projected.view(batch_size, 5, -1)  # [batch_size, 5, text_dim]

        # 使用图注意力机制聚合
        zed = self.graph_attn(text_raw, er_features)  # [batch_size, text_dim]

        # Unimodal processing
        text_prime, image_prime = self.uni_repre(
            torch.cat([text_raw, text], 1),
            torch.cat([image_raw, image], 1)
        )

        # Cross-modal processing
        correlation = self.cross_module(text, image)

        # Calculate similarity weights
        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) *
                        torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)

        # Apply correlation weights
        correlation = correlation * mweight


        shared_image_feature, shared_image_feature_1 = 0, 0
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_prime
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tmp_image_feature + self.positional_image)
            # yan:gate_image_feature[:, i].unsqueeze(1).unsqueeze(1) shape为 (batch_size, 1, 1)
            shared_image_feature += (tmp_image_feature * gate_image_feature[:, i].unsqueeze(1).unsqueeze(1))

        # 应该是cls token 的值
        shared_image_feature = shared_image_feature[:, 0]
        # shared_image _feature_1 = shared_image_feature_1[:, 0]

        ## TEXT AND MM EXPERTS
        shared_text_feature, shared_text_feature_1 = 0, 0
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            tmp_text_feature = text_prime
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](tmp_text_feature + self.positional_text)  # text_feature: 64, 170, 768
            shared_text_feature += (tmp_text_feature * gate_text_feature[:, i].unsqueeze(1).unsqueeze(1))
            # shared_text_feature_1 += (tmp_text_feature * gate_text_feature_1[:, i].unsqueeze(1).unsqueeze(1))
        shared_text_feature = shared_text_feature[:, 0]
        # shared_text_feature_1 = shared_text_feature_1[:, 0]


        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)
        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)


        aux_atn_score = 1 - torch.sigmoid(
            aux_output).clone().detach()  # torch.abs((torch.sigmoid(aux_output).clone().detach()-0.5)*2)
        is_mu = self.mapping_IS_MLP_mu(torch.sigmoid(image_only_output).clone().detach())
        t_mu = self.mapping_T_MLP_mu(torch.sigmoid(text_only_output).clone().detach())
        vgg_mu = self.mapping_IP_MLP_mu(torch.sigmoid(vgg_only_output).clone().detach())
        cc_mu = self.mapping_CC_MLP_mu(aux_atn_score.clone().detach())  # 1-aux_atn_score
        is_sigma = self.mapping_IS_MLP_sigma(torch.sigmoid(image_only_output).clone().detach())
        t_sigma = self.mapping_T_MLP_sigma(torch.sigmoid(text_only_output).clone().detach())
        vgg_sigma = self.mapping_IP_MLP_sigma(torch.sigmoid(vgg_only_output).clone().detach())
        cc_sigma = self.mapping_CC_MLP_sigma(aux_atn_score.clone().detach())  # 1-aux_atn_score

        shared_image_feature = self.adaIN(shared_image_feature, is_mu,
                                          is_sigma)  # shared_image_feature * (image_atn_score)
        shared_text_feature = self.adaIN(shared_text_feature, t_mu, t_sigma)  # shared_text_feature * (text_atn_score)

        # yan: 广播机制  irr_score shape = shared_mm_feature shape
        concat_feature_main_biased = torch.stack((shared_image_feature,
                                                  shared_text_feature,
                                                 zed
                                                  ), dim=1)


        fusion_tempfeat_main_task, _ = self.final_attention(concat_feature_main_biased)
        gate_main_task = self.fusion_SE_network_main_task(fusion_tempfeat_main_task)

        final_feature_main_task = 0
        for i in range(self.num_expert):
            fusing_expert = self.final_fusing_experts[i]
            tmp_fusion_feature = concat_feature_main_biased
            for j in range(self.depth):
                tmp_fusion_feature = fusing_expert[j](tmp_fusion_feature + self.positional_modal_representation)
            tmp_fusion_feature = tmp_fusion_feature[:, 0]
            final_feature_main_task += (tmp_fusion_feature * gate_main_task[:, i].unsqueeze(1))

        final_feature_main_task_lite = self.mix_trim(final_feature_main_task)
        mix_output = self.mix_classifier(final_feature_main_task_lite)

        return (mix_output, image_only_output, text_only_output)


        # # Combine features (新增ZED特征)
        # final_feature = torch.cat([
        #     text_prime.unsqueeze(1),
        #     image_prime.unsqueeze(1),
        #     correlation.unsqueeze(1),
        #     zed.unsqueeze(1)  # 新增ZED特征
        # ], 1)
        #
        # # Pooling and transformation
        # s1 = self.avepooling(final_feature)
        # s2 = self.maxpooling(final_feature)
        # s1 = s1.view(s1.size(0), -1)
        # s2 = s2.view(s2.size(0), -1)
        # s1 = self.senet(s1)
        # s2 = self.senet(s2)
        # s = self.sigmoid(s1 + s2)
        # s = s.view(s.size(0), s.size(1), 1)
        #
        # # Apply pooling weights
        # final_feature = s * final_feature
        #
        # # Classification
        # pre_label = self.classifier_corre(
        #     final_feature[:, 0, :] +
        #     final_feature[:, 1, :] +
        #     final_feature[:, 2, :] +
        #     final_feature[:, 3, :]  # 新增ZED特征的贡献
        # )
        #
        # return pre_label