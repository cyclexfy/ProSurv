
from collections import OrderedDict
from os.path import join
import pdb
# import eniops

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            # add
            nn.LayerNorm(dim2),
            nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block (Linear + ReLU + Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class MultiheadAttention(nn.Module):
    def __init__(self,
                 q_dim = 256,
                 k_dim = 256,
                 v_dim = 256,
                 embed_dim = 256,
                 out_dim = 256,
                 n_head = 4,
                 dropout=0.1,
                 temperature = 1
                 ):
        super(MultiheadAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = self.embed_dim//self.n_head
        self.temperature = temperature


        self.w_q = nn.Linear(self.q_dim, embed_dim)
        self.w_k = nn.Linear(self.k_dim, embed_dim)
        self.w_v = nn.Linear(self.v_dim, embed_dim)

        self.scale = (self.embed_dim//self.n_head) ** -0.5

        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

        # self.layerNorm1 = nn.LayerNorm(out_dim)
        # self.layerNorm2 = nn.LayerNorm(out_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )
        
        # self.feedForward = nn.Sequential(
        #     nn.Linear(out_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim, out_dim)
        # )

    def forward(self, q, k, v, return_attn = False):
        q_raw = q
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch_size = q.shape[0] # B
        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)

        attention_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_score = F.softmax(attention_score / self.temperature, dim = -1)

        attention_score = self.attn_dropout(attention_score)

        x = torch.matmul(attention_score, v)

        attention_score = attention_score.sum(dim = 1)/self.n_head
        
        attn_out = x.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        attn_out = self.out_proj(attn_out)

        attn_out = self.proj_dropout(attn_out)

        # attn_out = attn_out + q_raw

        # attn_out = self.layerNorm1(attn_out)

        # out = self.feedForward(attn_out)

        # out = self.layerNorm2(out + attn_out)

        # out = self.dropout(out)
        if return_attn:
            return attn_out, attention_score
        else:
            return attn_out
        # return out, attention_score


class GAN(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hiden_dim=512):
        super(GAN, self).__init__()
        self.generator = Generator(input_dim, output_dim, hiden_dim)
        self.discriminator = Discriminator(input_dim, hiden_dim)
    def forward(self, x):
        return self.generator(x), self.discriminator(x)

# 生成器
class Generator(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hiden_dim=512):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hiden_dim),
            nn.ReLU(),
            nn.Linear(hiden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.fc(x)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hiden_dim=512):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hiden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hiden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)
        

class VAE_Encoder(nn.Module):
    def __init__(self, input_dim=256):
        super(VAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.fc3_mean = nn.Linear(input_dim//4, input_dim//8)
        self.fc3_logvar = nn.Linear(input_dim//4, input_dim//8)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        logvar = self.fc3_logvar(h)
        return mean, logvar

# 解码器
class VAE_Decoder(nn.Module):
    def __init__(self, input_dim=256):
        super(VAE_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim//8, input_dim//4)
        self.fc2 = nn.Linear(input_dim//4, input_dim//2)
        self.fc3 = nn.Linear(input_dim//2, input_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

# VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim=256):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_dim)
        self.decoder = VAE_Decoder(input_dim)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

def vae_loss(reconstructed_x, x, mean, logvar):
    mse_loss = nn.MSELosss(reconstructed_x, x.detach(), reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return mse_loss + kld_loss
