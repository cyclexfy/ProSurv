import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import MultiheadAttention
from models.model_utils import *
from nystrom_attention import NystromAttention


class ProSurv(nn.Module):
    def __init__(self, 
                 omic_input_dim, 
                 fusion='concat', 
                 n_classes=4,
                 model_size_path: str='small', 
                 model_size_geno: str='small', 
                 mil_model_type='TransMIL',
                 geno_mlp_type='SNN',
                 memory_size=32,
                 dropout=0.1):
        
        super(ProSurv, self).__init__()
        self.fusion = fusion
        self.geno_input_dim = omic_input_dim
        self.n_classes = n_classes
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_geno = {'small': [1024, 256], 'big': [1024, 1024, 1024, 256]}

        ### Define Prototype Bank
        self.memory_size = memory_size
        self.memory_dim = 256

        self.path_prototype_bank = nn.Parameter(torch.empty(self.n_classes, self.memory_size, self.memory_dim))
        self.geno_prototype_bank = nn.Parameter(torch.empty(self.n_classes, self.memory_size, self.memory_dim))
        torch.nn.init.xavier_uniform_(self.path_prototype_bank, gain=1.0)
        torch.nn.init.xavier_uniform_(self.geno_prototype_bank, gain=1.0)

        ### pathlogy FC
        size = self.size_dict_path[model_size_path]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.LayerNorm(normalized_shape = size[1]))
        fc.append(nn.Dropout(dropout))
        self.path_proj = nn.Sequential(*fc)

        self.path_attn_net = pathMIL(model_type=mil_model_type, input_dim=size[1], dropout=dropout)
        
        ### Genomic SNN
        hidden = self.size_dict_geno[model_size_geno]
        if geno_mlp_type == 'SNN':
            geno_snn = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                geno_snn.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
            self.geno_snn = nn.Sequential(*geno_snn)
        else:
            self.geno_snn = nn.Sequential(
                nn.Linear(omic_input_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout))

        ### Multihead Attention
        self.path_intra_read_attn = MultiheadAttention(q_dim = self.size_dict_geno[model_size_geno][-1], k_dim = self.memory_dim, 
                                        v_dim = self.memory_dim, embed_dim = size[1], out_dim = size[1], 
                                        n_head = 4, dropout=dropout, temperature=0.5)

        self.geno_intra_read_attn = MultiheadAttention(q_dim = size[1], k_dim = self.memory_dim, 
                                        v_dim = self.memory_dim, embed_dim = size[1], out_dim = size[1], 
                                        n_head = 4, dropout=dropout, temperature=0.5)
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(size[1]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU()])
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        
    def forward(self, **kwargs):
        # input data
        x_path = kwargs['x_path']
        x_geno = kwargs['x_omic']
        label = kwargs['label']
        censor = kwargs['censor']
        is_training = kwargs['training']
        input_modality = kwargs['input_modality']

        if x_path!=None:
            batch_size = x_path.shape[0]
        elif x_geno!=None:
            batch_size = x_geno.shape[0]
        else:
            raise NotImplementedError

        # pathlogy projection
        if x_path!=None:
            h_path = self.path_proj(x_path) #[B, n_patchs, D]
            # pathlogy attention net
            h_path = self.path_attn_net(h_path) #[B, D]

        # Genomic SNN
        if x_geno!=None:
            h_geno = self.geno_snn(x_geno).squeeze(1) #[B, D]

        if is_training:
            # 先归一化，再计算相似度
            # similarity
            path_sim_loss = 0.
            geno_sim_loss = 0.

            if input_modality in ['path', 'path_and_geno']:
                path_prototype_norm = F.normalize(self.path_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim)) #[n_classes*size, D]
                h_path_norm = F.normalize(h_path) #[B, D]
                path_similarity = torch.matmul(h_path_norm, torch.transpose(path_prototype_norm, 0, 1)).reshape(
                    -1, self.n_classes, self.memory_size) #[B, n_classes, size]
                
                path_sim_loss = get_sim_loss(path_similarity, label, censor)

            if input_modality in ['geno', 'path_and_geno']:
                geno_prototype_norm = F.normalize(self.geno_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim)) #[n_classes*size, D]
                h_geno_norm = F.normalize(h_geno) #[B, D]
                geno_similarity = torch.matmul(h_geno_norm, torch.transpose(geno_prototype_norm, 0, 1)).reshape(
                    -1, self.n_classes, self.memory_size) #[B, n_classes, size]
                geno_sim_loss = get_sim_loss(geno_similarity, label, censor)
            
            sim_loss = path_sim_loss + geno_sim_loss

        # intra-modal read attention
        if input_modality in ['geno', 'path_and_geno']:
            path_prototype_bank_flat = self.path_prototype_bank.reshape(
                self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
            h_path_read = self.path_intra_read_attn(h_geno.unsqueeze(1), path_prototype_bank_flat, path_prototype_bank_flat).squeeze(1) # [B, D]

        if input_modality in ['path', 'path_and_geno']:
            geno_prototype_bank_flat = self.geno_prototype_bank.reshape(
                self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
            h_geno_read = self.geno_intra_read_attn(h_path.unsqueeze(1), geno_prototype_bank_flat, geno_prototype_bank_flat).squeeze(1) # [B, D]

        if input_modality == 'path':
            h_path_read = h_path
            h_geno = h_geno_read
        elif input_modality == 'geno':
            h_geno_read = h_geno
            h_path = h_path_read
        elif input_modality == 'path_and_geno':
            pass
        else:
            raise NotImplementedError(f'input_modality: {input_modality} not suported')
                
        h_path_avg = (h_path + h_path_read) /2
        h_geno_avg = (h_geno + h_geno_read) /2

        if self.training:
            path_loss_align = 0.
            geno_loss_align = 0.
            
            if input_modality == 'path_and_geno':
                path_loss_align = get_align_loss(h_path_read, h_path)
                geno_loss_align = get_align_loss(h_geno_read, h_geno)

            loss_align = path_loss_align + geno_loss_align

        ### Fusion Layer
        if self.fusion == 'bilinear':
            h = self.mm(h_path_avg, h_geno_avg).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path_avg, h_geno_avg], dim=-1))
        else:
            h = self.mm(h_path)
                
        ### Survival Layer
        logits = self.classifier(h)
        
        if is_training:
            return logits, sim_loss, loss_align
        else:
            if kwargs['return_feature']:
                return logits, h_path, h_geno_read, h_geno, h_geno_read
            else:
                return logits


class Memory_without_reconstruction(nn.Module):
    def __init__(self, 
                 omic_input_dim, 
                 fusion='concat', 
                 n_classes=4,
                 model_size_path: str='small', 
                 model_size_geno: str='small', 
                 mil_model_type='TransMIL',
                 memory_size=16,
                 dropout=0.1):
        
        super(Memory_without_reconstruction, self).__init__()
        self.fusion = fusion
        self.geno_input_dim = omic_input_dim
        self.n_classes = n_classes
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_geno = {'small': [1024, 256], 'big': [1024, 1024, 1024, 256]}

        ### pathlogy FC
        size = self.size_dict_path[model_size_path]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.LayerNorm(normalized_shape = size[1]))
        fc.append(nn.Dropout(dropout))
        self.path_proj = nn.Sequential(*fc)

        self.path_attn_net = pathMIL(model_type=mil_model_type, input_dim=size[1], dropout=dropout)
        
        ### Genomic SNN
        hidden = self.size_dict_geno[model_size_geno]
        geno_snn = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            geno_snn.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
        self.geno_snn = nn.Sequential(*geno_snn)
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(size[1]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU()])
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        
    def forward(self, **kwargs):
        # input data
        x_path = kwargs['x_path']
        x_geno = kwargs['x_omic']
        label = kwargs['label']
        censor = kwargs['censor']
        is_training = kwargs['training']
        input_modality = kwargs['input_modality']

        # pathlogy projection
        h_path = self.path_proj(x_path) #[B, n_patchs, D]
        # pathlogy attention net
        h_path = self.path_attn_net(h_path) #[B, D]

        # Genomic SNN
        h_geno = self.geno_snn(x_geno).squeeze(1) #[B, D]
        
        if input_modality == 'path':
            h_path_read = h_path
            h_geno = h_geno_read
        elif input_modality == 'geno':
            h_geno_read = h_geno
            h_path = h_path_read
        elif input_modality == 'path_and_geno':
            pass
        else:
            raise NotImplementedError
        
        h_path_avg = (h_path + h_path_read) /2
        h_geno_avg = (h_geno + h_geno_read) /2

        ### Fusion Layer
        if self.fusion == 'bilinear':
            h = self.mm(h_path_avg, h_geno_avg).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path_avg, h_geno_avg], dim=-1))
        else:
            h = self.mm(h_path)
                
        ### Survival Layer
        logits = self.classifier(h)
        
        return logits


def get_sim_loss(similarity, label, censor):
    similarity_positive_mean = []
    similarity_negative_mean = []
    for i in range(label.shape[0]):
        if censor[i] == 0:
            mask = torch.zeros_like(similarity[i], dtype=torch.bool)
            mask[label[i].item(), :] = True
            similarity_positive = torch.masked_select(similarity[i], mask).view(-1, similarity.size(-1)) #[n_pos, size]
            similarity_negative = torch.masked_select(similarity[i], ~mask).view(-1, similarity.size(-1)) #[n_neg, size]]
            similarity_positive_mean.append(torch.mean(torch.mean(similarity_positive, dim=-1), dim=-1)) # tensor
            similarity_negative_mean.append(torch.mean(torch.mean(similarity_negative, dim=-1), dim=-1)) # tensor

        else:
            if label[i] == 0:
                similarity_positive_mean.append(torch.mean(torch.mean(similarity[i], dim=-1), dim=-1)) # tensor
                similarity_negative_mean.append(torch.tensor(0, dtype=torch.float).cuda())
            else:   
                mask = torch.zeros_like(similarity[i], dtype=torch.bool)
                mask[label[i].item():, :] = True
                similarity_positive = torch.masked_select(similarity[i], mask).view(-1, similarity.size(-1)) #[n_pos, size]
                similarity_negative = torch.masked_select(similarity[i], ~mask).view(-1, similarity.size(-1)) #[n_neg, size]]
                similarity_positive_mean.append(torch.mean(torch.mean(similarity_positive, dim=-1), dim=-1)) # tensor
                similarity_negative_mean.append(torch.mean(torch.mean(similarity_negative, dim=-1), dim=-1)) # tensor

    # 将列表转换为张量并求和
    similarity_positive_mean = torch.stack(similarity_positive_mean) #[B]
    similarity_negative_mean = torch.stack(similarity_negative_mean) #[B]

    positive_mean_sum = torch.sum(similarity_positive_mean)
    negative_mean_sum = torch.sum(similarity_negative_mean)

    sim_loss = -positive_mean_sum + negative_mean_sum

    return sim_loss


def get_align_loss(read_feat, original_feat, align_fn='mse', reduction='none'):
    if align_fn == 'mse':
        loss_fn = nn.MSELoss(reduction=reduction)
    elif align_fn == 'l1':
        loss_fn = nn.L1Loss(reduction=reduction)
    else:
        raise NotImplementedError
    
    return torch.sum(torch.mean(loss_fn(read_feat, original_feat.detach()), dim=-1), dim=-1)


class pathMIL(nn.Module):
    def __init__(self, model_type = 'TransMIL', input_dim = 256, dropout=0.1):
        super(pathMIL, self).__init__()

        self.model_type = model_type

        if model_type == 'TransMIL':
            self.translayer1 = TransLayer(dim = input_dim)
            self.translayer2 = TransLayer(dim = input_dim)
            self.pos_layer = PPEG(dim = input_dim)
        elif model_type == 'ABMIL':
            self.path_gated_attn = Attn_Net_Gated(L=input_dim, D=input_dim, dropout=dropout, n_classes=1)

    def forward(self, h_path):

        if self.model_type == 'TransMIL':
            H = h_path.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h_path_sa = torch.cat([h_path, h_path[:,:add_length,:]], dim = 1) #[B, N, 512]
            # cls_token
            # B = h_path_sa.shape[0]
            # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            # h_path_sa = torch.cat((cls_tokens, h_path_sa), dim=1)
            # Translayer1
            h_path_sa = self.translayer1(h_path_sa) #[B, N, 256]
            # PPEG
            h_path_sa = self.pos_layer(h_path_sa, _H, _W) #[B, N, 256]
            # Translayer2
            h_path_sa = self.translayer2(h_path_sa) #[B, N, 256]
            # cls_token
            # h_path_sa = self.norm(h_path_sa)[:,0]
            h_path_sa = torch.mean(h_path, dim=1) #[B, 256]

            return h_path_sa

        elif self.model_type == 'ABMIL':
            A, h_path = self.path_gated_attn(h_path)
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=-1) 
            h_path = torch.matmul(A, h_path).squeeze(1) #[B, D]
            return h_path
        
        else:
            raise NotImplementedError
            return 


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//4,
            heads = 4,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape

        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        # x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
    
