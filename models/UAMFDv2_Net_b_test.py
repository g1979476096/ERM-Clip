from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
import copy
import pickle as pickle
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import datetime
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer
# import clip
from transformers import pipeline
# from googletrans import Translator
# from logger import Logger
import models_mae
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
from timm.models.vision_transformer import Block


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(1))/(x.shape[1])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

    def forward(self, x, mu, sigma):
        # yan: x [bs, 768]  mu [bs, 1]  sigma [bs, 1]
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim=dim

    def forward(self,x):
        x1,x2 = x.chunk(2,dim=self.dim)
        return x1*x2


class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
                            torch.nn.Linear(input_shape, input_shape),
                            nn.SiLU(),
                            #SimpleGate(dim=2),
                            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        # scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # scores = self.attention_layer(inputs)
        # outputs = scores*inputs
        return outputs, scores


class UAMFD_Net(nn.Module):
    def __init__(self,
                 batch_size=64, dataset='weibo', text_token_len=197, image_token_len=197, is_use_bce=True,
                 thresh=0.5
                 ):
        # NOTE: NOW WE ONLY SUPPORT BASE MODEL!
        self.thresh = thresh
        self.batch_size = batch_size
        self.text_token_len, self.image_token_len = text_token_len, image_token_len
        model_size = 'base'
        self.model_size = model_size
        self.dataset = dataset
        self.LOW_BATCH_SIZE_AND_LR = ['Twitter', 'politi']
        print("we are using adaIN")

        self.unified_dim, self.text_dim = 768, 768
        self.is_use_bce = is_use_bce
        out_dim = 1 if self.is_use_bce else 2
        self.num_expert = 2  # 2
        self.depth = 1  # 2
        super(UAMFD_Net, self).__init__()
        # IMAGE: MAE
        # if self.dataset in self.LOW_BATCH_SIZE_AND_LR:
        #     self.image_model = Block(dim=self.unified_dim, num_heads=8)
        # else:
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        checkpoint = torch.load('/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/models/mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        # checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)



        # yan:
        self.inter_image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        checkpoint = torch.load('/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/models/mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.inter_image_model.load_state_dict(checkpoint['model'], strict=False)

        self.interText_trans = nn.Sequential(nn.Linear(300, self.unified_dim),
                                                         nn.SiLU(),
                                                         nn.Linear(self.unified_dim, self.unified_dim),
                                                         )

        # for param in self.image_model.parameters():
        #     param.requires_grad = False

        # image_model_finetune = []
        # self.finetune_depth = 1
        # for j in range(self.finetune_depth):
        #     image_model_finetune.append(Block(dim=self.unified_dim, num_heads=8))  # note: need to output model[:,0]
        # self.image_model_finetune = nn.ModuleList(image_model_finetune)

        # TEXT: BERT OR PRETRAINED FROM WWW
        english_lists = ['gossip', 'Twitter', 'politi']
        model_name = 'bert-base-chinese' if self.dataset not in english_lists else 'bert-base-uncased'
        print("BERT: using {}".format(model_name))
        # if self.dataset in self.LOW_BATCH_SIZE_AND_LR:
        #     self.text_model = Block(dim=self.unified_dim, num_heads=8)
        # else:

        self.text_model = BertModel.from_pretrained(model_name)

        # for param in self.text_model.parameters():
        #     param.requires_grad = False
        # text_model_finetune = []
        # for j in range(self.finetune_depth):
        #     text_model_finetune.append(Block(dim=self.unified_dim, num_heads=8))  # note: need to output model[:,0]
        # self.text_model_finetune = nn.ModuleList(text_model_finetune)

        self.text_attention = TokenAttention(self.unified_dim)

        # IMAGE: RESNET-50
        # self.vgg_net = torchvision.models.resnet50(pretrained=False)
        # self.vgg_net.fc = nn.Sequential(
        #     nn.Linear(2048, self.unified_dim)
        # )
        # from CNN_architectures.pytorch_resnet import ResNet50
        # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()

        # yan:取消vgg模块
        # from CNN_architectures.pytorch_inceptionet import GoogLeNet
        # self.vgg_net = GoogLeNet(num_classes=self.unified_dim, use_SRM=True).cuda()

        # self.vgg_net = self.vgg_net.cuda()
        self.image_attention = TokenAttention(self.unified_dim)

        self.mm_attention = TokenAttention(self.unified_dim)

        self.interImg_attention = TokenAttention(self.unified_dim)
        self.interText_attention = TokenAttention(self.unified_dim)
        self.intermm_attention = TokenAttention(self.unified_dim)
        # GATE, EXPERTS
        # feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64} # 64*5 note there are 5 kernels and 5 experts!
        image_expert_list, text_expert_list, mm_expert_list, interImg_expert_list,interText_expert_list,intermm_expert_list = [], [], [], [], [], []

        for i in range(self.num_expert):
            image_expert = []
            for j in range(self.depth):
                image_expert.append(Block(dim=self.unified_dim, num_heads=8))  # note: need to output model[:,0]

            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)

        for i in range(self.num_expert):
            text_expert = []
            mm_expert = []
            for j in range(self.depth):
                text_expert.append(Block(dim=self.unified_dim, num_heads=8))
                mm_expert.append(Block(dim=self.unified_dim, num_heads=8))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
            mm_expert = nn.ModuleList(mm_expert)
            mm_expert_list.append(mm_expert)

        for i in range(self.num_expert):
            interText_expert = []
            interImg_expert = []
            intermm_expert = []
            for j in range(self.depth):
                interText_expert.append(Block(dim=self.unified_dim, num_heads=8))
                interImg_expert.append(Block(dim=self.unified_dim, num_heads=8))
                intermm_expert.append(Block(dim=self.unified_dim, num_heads=8))

            interText_expert = nn.ModuleList(interText_expert)
            interText_expert_list.append(interText_expert)
            interImg_expert = nn.ModuleList(interImg_expert)
            interImg_expert_list.append(interImg_expert)
            intermm_expert = nn.ModuleList(intermm_expert)
            intermm_expert_list.append(intermm_expert)

        self.image_experts = nn.ModuleList(image_expert_list)
        self.text_experts = nn.ModuleList(text_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)
        self.interImg_experts = nn.ModuleList(interImg_expert_list)
        self.interText_experts = nn.ModuleList(interText_expert_list)
        self.intermm_experts = nn.ModuleList(intermm_expert_list)
        # self.out_unified_dim = 320
        # yan: SiLU(x) = x * sigmod(x)
        self.image_gate_mae = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                            nn.SiLU(),
                                            # SimpleGate(),
                                            # nn.BatchNorm1d(int(self.unified_dim/2)),
                                             nn.Linear(self.unified_dim, self.num_expert),
                                            # nn.Dropout(0.1),
                                            # nn.Softmax(dim=1)
                                            )
        # self.image_gate_vgg = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                 nn.LayerNorm(self.unified_dim),
        #                                 SimpleGate(), # nn.SiLU(),
        #                                 nn.Dropout(0.2),
        #                                 nn.Linear(self.unified_dim, 1),
        #                                 nn.Softmax(dim=1)
        #                                 )

        self.text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       # SimpleGate(),
                                       # nn.BatchNorm1d(int(self.unified_dim/2)),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       # nn.Dropout(0.1),
                                       # nn.Softmax(dim=1)
                                       )
        self.mm_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.SiLU(),
                                     # SimpleGate(),
                                     # nn.BatchNorm1d(int(self.unified_dim/2)),
                                     nn.Linear(self.unified_dim, self.num_expert),
                                     # nn.Dropout(0.1),
                                     # nn.Softmax(dim=1)
                                     )
        self.mm_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.SiLU(),
                                     # SimpleGate(),
                                     # nn.BatchNorm1d(int(self.unified_dim/2)),
                                     nn.Linear(self.unified_dim, self.num_expert),
                                     # nn.Dropout(0.1),
                                     # nn.Softmax(dim=1)
                                     )

        self.image_gate_mae_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                              nn.SiLU(),
                                              # SimpleGate(),
                                              # nn.BatchNorm1d(int(self.unified_dim/2)),
                                              nn.Linear(self.unified_dim, self.num_expert),
                                              # nn.Dropout(0.1),
                                              # nn.Softmax(dim=1)
                                              )

        self.text_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                         nn.SiLU(),
                                         # SimpleGate(),
                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                         nn.Linear(self.unified_dim, self.num_expert),
                                         # nn.Dropout(0.1),
                                         # nn.Softmax(dim=1)
                                         )
        self.interText_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                         nn.SiLU(),
                                         # SimpleGate(),
                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                         nn.Linear(self.unified_dim, self.num_expert),
                                         # nn.Dropout(0.1),
                                         # nn.Softmax(dim=1)
                                         )
        self.interImg_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                         nn.SiLU(),
                                         # SimpleGate(),
                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                         nn.Linear(self.unified_dim, self.num_expert),
                                         # nn.Dropout(0.1),
                                         # nn.Softmax(dim=1)
                                         )
        self.intermm_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                         nn.SiLU(),
                                         # SimpleGate(),
                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                         nn.Linear(self.unified_dim, self.num_expert),
                                         # nn.Dropout(0.1),
                                         # nn.Softmax(dim=1)
                                         )
        ## MAIN TASK GATES
        self.final_attention = TokenAttention(self.unified_dim)

        # self.mm_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                              SimpleGate(),
        #                                              nn.BatchNorm1d(128),
        #                                              # nn.Dropout(0.2),
        #                                              )
        # self.text_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                SimpleGate(),
        #                                                nn.BatchNorm1d(128),
        #                                                # nn.Dropout(0.2),
        #                                                )
        # self.image_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                 SimpleGate(),
        #                                                 nn.BatchNorm1d(128),
        #                                                 # nn.Dropout(0.2),
        #                                                 )
        # self.vgg_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                               SimpleGate(),
        #                                               nn.BatchNorm1d(128),
        #                                               # nn.Dropout(0.2),
        #                                               )
        self.fusion_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                                         nn.SiLU(),
                                                         # SimpleGate(),
                                                         # nn.BatchNorm1d(int(self.unified_dim/2)),
                                                         nn.Linear(self.unified_dim, self.num_expert),
                                                         # nn.Softmax(dim=1)
                                                         )
        ## AUXILIARY TASK GATES
        # self.mm_SE_network_aux_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                              nn.LayerNorm(256),
        #                                              SimpleGate(), # nn.SiLU(),
        #                                              nn.Dropout(0.2),
        #                                              )
        # self.text_SE_network_aux_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                nn.LayerNorm(256),
        #                                                SimpleGate(), # nn.SiLU(),
        #                                                nn.Dropout(0.2),
        #                                                )
        # self.image_SE_network_aux_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                 nn.LayerNorm(256),
        #                                                 SimpleGate(), # nn.SiLU(),
        #                                                 nn.Dropout(0.2),
        #                                                 )
        # self.fusion_SE_network_aux_task = nn.Sequential(nn.Linear(3 * 256, 256),
        #                                                  nn.LayerNorm(256),
        #                                                  SimpleGate(), # nn.SiLU(),
        #                                                  nn.Dropout(0.2),
        #                                                  nn.Linear(256, self.num_expert),
        #                                                  nn.Softmax(dim=1)
        #                                                  )

        # self.irre_dim = self.unified_dim
        # self.irre_feature = None
        # self.irr_MLP = nn.Linear(self.unified_dim, self.unified_dim)
        # yan: 论文中 ex 设置为可训练参数
        # self.irrelevant_tensor = nn.Parameter(torch.ones((1,self.unified_dim)),requires_grad=True)

        # self.irrelevant_token = nn.Parameter(torch.randn(self.unified_dim),requires_grad=True) #torch.zeros_like(shared_mm_feature).cuda()
        ## POSITIONAL ENCODING FOR MM FEATURE

        # yan：取消 论文中 ex 设置为可训练参数，使用正交进行解缠
        self.disentangle_irrelevant = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                                         nn.SiLU(),
                                                         nn.Linear(self.unified_dim, self.unified_dim),
                                                         )
        self.disentangle_relevant = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                                    nn.SiLU(),
                                                    nn.Linear(self.unified_dim, self.unified_dim),
                                                    )

        # yan


        # self.positional_imagefeature = pos[:, 0, :].squeeze().cuda()
        # self.positional_textfeature = pos[:, 1, :].squeeze().cuda()
        # self.positional_mmfeature = pos[:, 2, :].squeeze().cuda()
        # self.irrelevant_tensor = pos[:,3,:].squeeze().cuda()
        # print(pos)
        # print(pos[:, 25, :])

        # self.mm_SE_network_classifier = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                              nn.LayerNorm(self.unified_dim),
        #                              SimpleGate(), # nn.SiLU(),
        #                              nn.Dropout(0.2),
        #                              nn.Linear(self.unified_dim, 2)
        #                                    )

        # CLASSIFICATION HEAD
        self.mix_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.text_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.image_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        # self.vgg_trim = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        #     # SimpleGate(),
        #     # nn.BatchNorm1d(64),
        #     # nn.Dropout(0.2),
        # )
        # self.vgg_alone_classifier = nn.Sequential(
        #     nn.Linear(64, out_dim),
        # )

        self.aux_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        #### mapping MLPs
        self.mapping_IS_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IS_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim,1),
        )
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        # self.mapping_IP_MLP_mu = nn.Sequential(
        #     nn.Linear(1, self.unified_dim),
        #     nn.SiLU(),
        #     # nn.BatchNorm1d(self.unified_dim),
        #     nn.Linear(self.unified_dim, 1),
        # )
        # self.mapping_IP_MLP_sigma = nn.Sequential(
        #     nn.Linear(1, self.unified_dim),
        #     nn.SiLU(),
        #     # nn.BatchNorm1d(self.unified_dim),
        #     nn.Linear(self.unified_dim, 1),
        # )
        self.mapping_CC_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()

        final_fusing_expert = []
        for i in range(self.num_expert):
            fusing_expert = []
            for j in range(self.depth):
                fusing_expert.append(Block(dim=self.unified_dim, num_heads=8))
            fusing_expert = nn.ModuleList(fusing_expert)
            final_fusing_expert.append(fusing_expert)

        self.final_fusing_experts = nn.ModuleList(final_fusing_expert)

        self.mm_score = None

        # self.final_fusing_experts = Block(dim=self.unified_dim, num_heads=8)

    def get_pretrain_features(self, input_ids, attention_mask, token_type_ids, image, no_ambiguity, category=None,
                              calc_ambiguity=False):
        image_feature = self.image_model.forward_ying(image)
        text_feature = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]

        return image_feature, text_feature

    def forward(self, input_ids, attention_mask, token_type_ids, image, no_ambiguity,\
                        image_aug=None, category=None,\
                        calc_ambiguity=False, image_feature=None, text_feature=None,\
                        return_features=False,interpretability_img_s=None, interpretability_text_s=None):

        # 测试
        # input_ids, image, attention_mask, token_type_ids,
        #                 image_aug=None, category=None,no_ambiguity=True,
        #                 calc_ambiguity=False, image_feature=None, text_feature=None,
        #                 return_features=True
        # 测试

        # print(input_ids.shape) # (24,197)
        # print(attention_mask.shape) # (24,197)
        # print(token_type_ids.shape) # (24,197)
        # print(interpretability_img.shape, interpretability_text.shape, interpretability_text, 'yan'*20)

        batch_size = image.shape[0]
        if image_aug is None:
            image_aug = image

        ## POSITIONAL ENCODING FOR MM FEATURE
        # yan:https://pypi.org/project/positional-encodings/
        p_1d_mm = PositionalEncoding1D(self.unified_dim)
        x_mm = torch.rand(batch_size, self.image_token_len + self.text_token_len, self.unified_dim)
        self.positional_mm = p_1d_mm(x_mm).cuda()
        p_1d_image = PositionalEncoding1D(self.unified_dim)
        x_image = torch.rand(batch_size, self.image_token_len, self.unified_dim)
        self.positional_image = p_1d_image(x_image).cuda()
        p_1d_text = PositionalEncoding1D(self.unified_dim)
        x_text = torch.rand(batch_size, self.text_token_len, self.unified_dim)
        self.positional_text = p_1d_text(x_text).cuda()
        ## POSITIONAL ENCODING FOR MULTIMODAL
        ## 5 IS BECAUSE THE MODALS ARE IMAGE TEXT MM IRRELEVANT AND VGG
        p_1d = PositionalEncoding1D(self.unified_dim)
        # x = torch.rand(batch_size, 5, self.unified_dim)
        x = torch.rand(batch_size, 5, self.unified_dim)
        self.positional_modal_representation = p_1d(x).cuda()

        # BASE FEATURE AND ATTENTION
        # if category is None:
        #     category = torch.zeros((batch_size))

        # IMAGE MAE:  OUTPUT IS (BATCH, 197, 768)
        ## FILTER OUT INVALID MODAL INFORMATION
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if image_feature is None:
            image_feature = self.image_model.forward_ying(image_aug)

        # yan:
        if interpretability_img_s is not None:
            interpretability_img = self.inter_image_model.forward_ying(interpretability_img_s)

        if interpretability_text_s is not None:
            interpretability_text = self.interText_trans(interpretability_text_s)

        # print(interpretability_img.shape,interpretability_text.shape,'wang'*20)
        # for j in range(self.finetune_depth):
        #     image_feature = self.image_model_finetune[j](image_feature + self.positional_image)

        # yan:rm vgg
        # vgg_feature = self.vgg_net(image)  # 64, 768


        # TEXT:  INPUT IS (BATCH, WORDLEN, 768)
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        # if self.dataset in self.LOW_BATCH_SIZE_AND_LR:
        #     text_feature = self.text_model(input_ids)
        # else:
        if text_feature is None:
            # yan: BertModel embed_dim 默认768
            text_feature = self.text_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)[0]

        # yan:============
        # https://zhuanlan.zhihu.com/p/350837279 timm.model vit Block
        # batch_size 8
        # print('='*20)
        # print(image_feature.shape, vgg_feature.shape, text_feature.shape)
        # torch.Size([8, 197, 768]) torch.Size([8, 768]) torch.Size([8, 197, 768])
        # print('=' * 20)
        # ======yan=======

        # for j in range(self.finetune_depth):
        #     text_feature = self.text_model_finetune[j](text_feature + self.positional_text)

        # print("text_feature size {}".format(text_feature.shape)) # 64,170,768
        # print("image_feature size {}".format(image_feature.shape)) # 64,197,1024
        text_atn_feature, _ = self.text_attention(text_feature)

        # IMAGE ATTENTION: OUTPUT IS (BATCH, 768)
        image_atn_feature, _ = self.image_attention(image_feature)

        mm_atn_feature, _ = self.mm_attention(torch.cat((image_feature, text_feature), dim=1))

        # yan:
        interText_atn_feature, _ = self.interText_attention(interpretability_text)
        interImg_atn_feature, _ = self.interImg_attention(interpretability_img)
        intermm_atn_feature, _ = self.intermm_attention(torch.cat((interpretability_img, interpretability_text), dim=1))

        # print("text_atn_feature size {}".format(text_atn_feature.shape)) # 64, 768
        # print("image_atn_feature size {}".format(image_atn_feature.shape))
        # GATE
        # gate_image_feature_mae = self.image_gate_mae(image_atn_feature)
        # gate_image_feature_vgg = self.image_gate_vgg(vgg_feature)
        # gate_image_feature = self.soft_max(torch.cat((gate_image_feature_mae,gate_image_feature_vgg),dim=1))

        # yan:image_atn_feature  (batch_size, 768)
        # 从最初的 (8,197,768)经过TokenAttention变成(8,768),为了计算MMOE中的gate gate(batch_size, num_expert)
        gate_image_feature = self.image_gate_mae(image_atn_feature)
        gate_text_feature = self.text_gate(text_atn_feature)  # 64 320

        gate_mm_feature = self.mm_gate(mm_atn_feature)
        gate_mm_feature_1 = self.mm_gate_1(mm_atn_feature)

        # yan:
        gate_interImg_feature = self.interImg_gate(interImg_atn_feature)
        gate_interText_feature = self.interText_gate(interText_atn_feature)  # 64 320
        gate_intermm_feature = self.intermm_gate(intermm_atn_feature)


        # gate_image_feature_1 = self.image_gate_mae_1(image_atn_feature)
        # gate_text_feature_1 = self.text_gate_1(text_atn_feature)  # 64 320

        # IMAGE EXPERTS
        # NOTE: IMAGE/TEXT/MM EXPERTS WILL BE MLPS IF WE USE WWW LOADER
        shared_image_feature, shared_image_feature_1 = 0, 0
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tmp_image_feature + self.positional_image)
            # yan:gate_image_feature[:, i].unsqueeze(1).unsqueeze(1) shape为 (batch_size, 1, 1)
            shared_image_feature += (tmp_image_feature * gate_image_feature[:, i].unsqueeze(1).unsqueeze(1))
            # yan:
            # tmp_image_feature * gate_image_feature[:, i].unsqueeze(1).unsqueeze(1)
            # 等于[197, 768] * [1, 1] 即每一个batch 768维向量都乘一个对应的gate值 如-0.0440，来进行门控
            # shared_image_feature 最终是所有专家经过门控后的值的和

            # print('=' * 20)
            # print(tmp_image_feature.shape, gate_image_feature.shape, \
            #       gate_image_feature[:, i].unsqueeze(1).unsqueeze(1).shape, gate_image_feature[:, i])
            # torch.Size([8, 197, 768]) torch.Size([8, 2]) torch.Size([8, 1, 1])
            # tensor([-0.0440, -0.0440, -0.0440, -0.0440, -0.0440, -0.0440, -0.0440, -0.0440],
            # device='cuda:0', grad_fn=<SelectBackward0>)
            # print('=' * 20)
            # ========yan=====
            # shared_image_feature_1 += (tmp_image_feature * gate_image_feature_1[:, i].unsqueeze(1).unsqueeze(1))
        # yan: 取每个batch 二维的第一行  (batch_size, 768)
        # 应该是cls token 的值
        shared_image_feature = shared_image_feature[:, 0]
        # shared_image _feature_1 = shared_image_feature_1[:, 0]

        ## TEXT AND MM EXPERTS
        shared_text_feature, shared_text_feature_1 = 0, 0
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            tmp_text_feature = text_feature
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](tmp_text_feature + self.positional_text)  # text_feature: 64, 170, 768
            shared_text_feature += (tmp_text_feature * gate_text_feature[:, i].unsqueeze(1).unsqueeze(1))
            # shared_text_feature_1 += (tmp_text_feature * gate_text_feature_1[:, i].unsqueeze(1).unsqueeze(1))
        shared_text_feature = shared_text_feature[:, 0]
        # shared_text_feature_1 = shared_text_feature_1[:, 0]

        mm_feature = torch.cat((image_feature, text_feature), dim=1)
        # mm_feature = torch.cat((shared_image_feature_1, shared_text_feature_1), dim=1)
        shared_mm_feature, shared_mm_feature_CC = 0, 0
        for i in range(self.num_expert):
            mm_expert = self.mm_experts[i]
            tmp_mm_feature = mm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tmp_mm_feature + self.positional_mm)
            shared_mm_feature += (tmp_mm_feature * gate_mm_feature[:, i].unsqueeze(1).unsqueeze(1))
            shared_mm_feature_CC += (tmp_mm_feature * gate_mm_feature_1[:, i].unsqueeze(1).unsqueeze(1))
        shared_mm_feature = shared_mm_feature[:, 0]
        shared_mm_feature_CC = shared_mm_feature_CC[:, 0]

        # yan:
        # print(shared_mm_feature.shape,shared_mm_feature_CC.shape)
        # torch.Size([8, 768])
        # torch.Size([8, 768])

        shared_interImg_feature = 0
        for i in range(self.num_expert):
            image_expert = self.interImg_experts[i]
            tmp_image_feature = interpretability_img
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tmp_image_feature + self.positional_image)
            # yan:gate_image_feature[:, i].unsqueeze(1).unsqueeze(1) shape为 (batch_size, 1, 1)
            shared_interImg_feature += (tmp_image_feature * gate_interImg_feature[:, i].unsqueeze(1).unsqueeze(1))

        shared_interImg_feature = shared_interImg_feature[:, 0]


        shared_interText_feature = 0
        for i in range(self.num_expert):
            text_expert = self.interText_experts[i]
            tmp_text_feature = interpretability_text
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](tmp_text_feature + self.positional_text)  # text_feature: 64, 170, 768
            shared_interText_feature += (tmp_text_feature * gate_interText_feature[:, i].unsqueeze(1).unsqueeze(1))

        shared_interText_feature = shared_interText_feature[:, 0]

        # cat的是shared_interText_feature 还是interpretability_text?
        intermm_feature = torch.cat((interpretability_img, interpretability_text), dim=1)
        shared_intermm_feature = 0
        for i in range(self.num_expert):
            mm_expert = self.intermm_experts[i]
            tmp_mm_feature = intermm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tmp_mm_feature + self.positional_mm)
            shared_intermm_feature += (tmp_mm_feature * gate_intermm_feature[:, i].unsqueeze(1).unsqueeze(1))

        shared_intermm_feature = shared_intermm_feature[:, 0]





        dis_irr = self.disentangle_irrelevant(shared_mm_feature)
        dis_rr = self.disentangle_relevant(shared_mm_feature)
        # 正交
        dis_rr_t = torch.transpose(dis_rr, 0, 1)
        mul_irr = torch.mm(dis_irr, dis_rr_t)
        # 获取对角线元素
        diag = torch.diag(mul_irr)
        diag_abs = torch.abs(diag)
        mutual_info_loss = torch.sum(diag_abs)
        # print(torch.sum(diag)) tensor(49.6681, device='cuda:0', grad_fn=<SumBackward0>)

        # print('11')

        # print(dis_irr.shape,dis_rr.shape,'irrs')torch.Size([8, 768]) torch.Size([8, 768]) irrs



        # image_mask = torch.ones_like(shared_image_feature)
        # mm_mask = torch.ones_like(shared_mm_feature)
        # text_mask = torch.ones_like(shared_text_feature)
        # ## MASKING OF MULTIMODALS TO FORBIDDEN INVALID INFORMATION
        # ## CATEGORY: MULTI 0 IMAGE 1 TEXT 2
        # for idx, value in enumerate(category):
        #     if value == 2:
        #         torch.zero_(image_mask[idx])
        #         torch.zero_(mm_mask[idx])
        #     elif value == 1:
        #         torch.zero_(text_mask[idx])
        # shared_image_feature = shared_image_feature * image_mask
        # shared_text_feature = shared_text_feature * text_mask
        # shared_mm_feature = shared_mm_feature * mm_mask
        # vgg_feature = vgg_feature * image_mask

        ## SCORES FOR THE FOUR MODALS
        ## NOTE: MMSCORE->0 IF IMAGE AND TEXT ARE FROM ONE NEWS
        ## AND THEREFORE SHOULD BE REVERTED AS 1-MMSCORE LATER
        ###### NOTE: HUGE MODIFICATION HAS TAKEN PLACE IN V2 ########
        ## GATES FOR AUXILIARY TASK
        # mm_tempfeat_aux_task = self.mm_SE_network_aux_task(shared_mm_feature)
        # image_tempfeat_aux_task = self.image_SE_network_aux_task(shared_image_feature)
        # text_tempfeat_aux_task = self.text_SE_network_aux_task(shared_text_feature)
        # fusion_tempfeat_aux_task = torch.cat((mm_tempfeat_aux_task, image_tempfeat_aux_task, text_tempfeat_aux_task), dim=1)
        # gate_aux_task = self.fusion_SE_network_aux_task(fusion_tempfeat_aux_task)

        ## ADD POSITIONAL ENCODING ON THE MULTIMODAL FEATURES
        # shared_image_feature = shared_image_feature + self.positional_imagefeature
        # shared_text_feature = shared_text_feature + self.positional_textfeature
        # shared_mm_feature = shared_mm_feature + self.positional_mmfeature
        # irrelevant_token = self.irrelevant_token.unsqueeze(0).repeat(batch_size,1)

        # concat_feature = torch.stack((shared_image_feature, shared_text_feature,
        #                               shared_mm_feature,irrelevant_token), dim=1)
        # final_feature_aux_task = 0
        # for i in range(self.num_expert):
        #     fusing_expert = self.final_fusing_experts[i]
        #     tmp_fusion_feature = concat_feature
        #     for j in range(self.depth):
        #         tmp_fusion_feature = fusing_expert[j](tmp_fusion_feature+self.positional_modal_representation)
        #     tmp_fusion_feature = tmp_fusion_feature[:,0]
        #     final_feature_aux_task += (tmp_fusion_feature * gate_aux_task[:, i].unsqueeze(1))
        shared_mm_feature_lite = self.aux_trim(shared_mm_feature_CC)
        aux_output = self.aux_classifier(shared_mm_feature_lite)  # final_feature_aux_task
        # if self.is_use_bce:
        #     aux_output = torch.sigmoid(aux_output)

        # yan: 此处计算 论文中 Consistency Learning  0 is real   1 is fake
        if calc_ambiguity:
            return aux_output, aux_output, aux_output

        ## UNIMODAL BRANCHES, NOT USED ANY MORE
        # aux_output = aux_output.clone().detach()
        # vgg_feature_lite = self.vgg_trim(vgg_feature)
        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)

        # vgg_only_output = self.vgg_alone_classifier(vgg_feature_lite)
        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)
        # if self.is_use_bce:
        #     image_only_output = torch.sigmoid(image_only_output)
        #     text_only_output = torch.sigmoid(text_only_output)

        ## WEIGHTED MULTIMODAL FEATURES, REMEMBER TO DETACH AUX_OUTPUT
        # soft_scores = torch.softmax(torch.cat((aux_output,image_only_output,text_only_output,vgg_only_output),dim=1),dim=1)
        ## IF IMAGE-TEXT MATCHES, aux_output WOULD BE 0, OTHERWISE 1.
        # yan: 0 is real   1 is fake
        aux_atn_score = 1 - torch.sigmoid(aux_output).clone().detach()  # torch.abs((torch.sigmoid(aux_output).clone().detach()-0.5)*2)
        # yan: 如果原tensor的requires_grad=True
        # clone()操作后的tensor requires_grad=True
        # detach()操作后的tensor requires_grad=False
        is_mu = self.mapping_IS_MLP_mu(torch.sigmoid(image_only_output).clone().detach())
        t_mu = self.mapping_T_MLP_mu(torch.sigmoid(text_only_output).clone().detach())
        # vgg_mu = self.mapping_IP_MLP_mu(torch.sigmoid(vgg_only_output).clone().detach())
        cc_mu = self.mapping_CC_MLP_mu(aux_atn_score.clone().detach())  # 1-aux_atn_score
        is_sigma = self.mapping_IS_MLP_sigma(torch.sigmoid(image_only_output).clone().detach())
        t_sigma = self.mapping_T_MLP_sigma(torch.sigmoid(text_only_output).clone().detach())
        # vgg_sigma = self.mapping_IP_MLP_sigma(torch.sigmoid(vgg_only_output).clone().detach())
        cc_sigma = self.mapping_CC_MLP_sigma(aux_atn_score.clone().detach())  # 1-aux_atn_score
        # yan:
        # print('=上='*20)
        # print(is_mu, is_mu.shape, is_sigma, is_sigma.shape)
        # torch.Size([8, 1])
        # print('=下=' * 20)
        #=====yan===
        # image_atn_score = self.mapping(image_only_output)  # torch.abs((torch.sigmoid(image_only_output).clone().detach() - 0.5) * 2)
        # text_atn_score = self.mapping(text_only_output)  # torch.abs((torch.sigmoid(text_only_output).clone().detach() - 0.5) * 2)
        # vgg_atn_score = self.mapping(vgg_only_output)  # torch.abs((torch.sigmoid(vgg_only_output).clone().detach() - 0.5) * 2)

        shared_image_feature = self.adaIN(shared_image_feature,is_mu,is_sigma) #shared_image_feature * (image_atn_score)
        shared_text_feature = self.adaIN(shared_text_feature,t_mu,t_sigma) #shared_text_feature * (text_atn_score)
        shared_mm_feature = shared_mm_feature #shared_mm_feature #* (aux_atn_score)
        # vgg_feature = self.adaIN(vgg_feature,vgg_mu,vgg_sigma) #vgg_feature * (vgg_atn_score)
        # irr_score = self.irr_MLP(torch.ones((batch_size,self.unified_dim)).cuda())
        # yan: 广播机制  irr_score shape = shared_mm_feature shape
        # irr_score = torch.ones_like(shared_mm_feature)*self.irrelevant_tensor #torch.ones_like(shared_mm_feature).cuda()
        irrelevant_token = self.adaIN(dis_irr,cc_mu,cc_sigma)
        # print(irrelevant_token.shape, shared_intermm_feature.shape,'irr'*20)
        # yan: bsz 一半置零 为有监督数据   一般不置零为半监督数据
        # bsz 为 8
        shared_intermm_feature = shared_intermm_feature * 0.5
        shared_intermm_feature[0:2] = 0

        #解馋 消融实验
        # irrelevant_token = 0
        # shared_intermm_feature = shared_intermm_feature * 0
        #
        concat_feature_main_biased = torch.stack((shared_image_feature,
                                                  shared_text_feature,
                                                  shared_mm_feature,
                                                  # vgg_feature,
                                                  irrelevant_token,
                                                  shared_intermm_feature
                                                  ), dim=1)

        ## GATES FOR MAIN TASK
        # mm_tempfeat_main_task = self.mm_SE_network_main_task(shared_mm_feature)
        # image_tempfeat_main_task = self.image_SE_network_main_task(shared_image_feature)
        # vgg_tempfeat_main_task = self.vgg_SE_network_main_task(vgg_feature)
        # text_tempfeat_main_task = self.text_SE_network_main_task(shared_text_feature)
        # fusion_tempfeat_main_task = torch.cat(
        #     (mm_tempfeat_main_task, image_tempfeat_main_task, text_tempfeat_main_task, vgg_tempfeat_main_task), dim=1)

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
        # if self.is_use_bce:
        #     mix_output = torch.sigmoid(mix_output)
        if return_features:
            # 测试
            # return mix_output
            #
            #  vgg_only_output,  vgg_feature_lite,
            # torch.mean(self.irrelevant_tensor), \
            return mix_output, image_only_output, text_only_output, aux_output, \
                   mutual_info_loss,\
                   (final_feature_main_task_lite, shared_image_feature_lite, shared_text_feature_lite, shared_mm_feature_lite)

        #  vgg_only_output,
        return mix_output, image_only_output, text_only_output, aux_output, \
               mutual_info_loss
               # torch.mean(self.irrelevant_tensor)

    def mapping(self, score):
        ## score is within 0-1
        diff_with_thresh = torch.abs(score-self.thresh)
        interval = torch.where(score-self.thresh>0, 1-self.thresh, self.thresh)
        return diff_with_thresh/interval


if __name__ == '__main__':
    # yan: 计算模型参数总量和模型计算量
    from thop import profile
    model = UAMFD_Net()
    device = torch.device("cpu")
    input1 = torch.randn(1, 197, 768)
    input2 = torch.randn(1, 197, 768)
    # input3 = torch.randn(1, 197, 768)
    flops,params = profile(model,inputs=(input1,input2))

    # stat(self.localizer.to(torch.device('cuda:0')), (3, 512, 512))
