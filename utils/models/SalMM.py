import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba


from torchvision.transforms import Resize
from .clip import *


clip_path = 'clip'
device = torch.device('cuda:0')
backbone = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

id = 4
'''
[official note] The original text contains an error. 
[official note] We wrongly wrote the model with id=4 as 'RN50x16'.
[official note] In fact, the backbone used by CLIP is 'ViT-B/32'.
[official note] reference: https://arxiv.org/pdf/2502.16214
'''


'''
CrossModelAtt: q-->clip  k-->clip  v-->feature
'''
class CrossModelAtt(nn.Module):
    def __init__(self, backbone=backbone[id], device=device):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(backbone)
        self.model = self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, feature):
        # 1: get clip feature
        b, c, h, w = feature.shape

        torch_resize = Resize([224, 224])
        img = torch_resize(img)

        with torch.no_grad():
            clip_feature = self.model.encode_image(img)
            # projector: [B, N] ---> [B, C, H/2, W/2]
            clip_feature = clip_feature.view(b, c, h//2, w//2)

        # print(clip_feature.shape)
        # 2: perception matrix
        q = clip_feature.view(b, c, -1)
        k = clip_feature.view(b, c, -1).permute(0, 2, 1)
        # print('q', q.shape, 'k', k.shape)
        perception = torch.bmm(q, k)
        perception = torch.max(perception, -1, keepdim=True)[0].expand_as(perception) - perception
        perception = perception

        v = feature.view(b, c, -1)
        # print(perception.shape, v.shape)
        # print(type(perception), type(v))
        perception_info = torch.bmm(perception.float(), v.float())
        perception_info = perception_info.view(b, c, h, w)

        perception_info = self.gamma*perception_info + feature

        return perception_info


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return branch1, branch2, branch3, branch4


class SCPMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, dim_=[1, 1, 1, 1, 1, 1]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(output_dim)
        self.norm1 = nn.LayerNorm(dim_[0])
        self.norm2 = nn.LayerNorm(dim_[2])
        self.norm3 = nn.LayerNorm(dim_[4])
        self.norm4 = nn.LayerNorm(dim_[5])

        self.mamba1 = Mamba(
            d_model=dim_[0],  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        self.mamba2 = Mamba(
            d_model=dim_[2],  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        self.mamba3 = Mamba(
            d_model=dim_[4],  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        self.mamba4 = Mamba(
            d_model=dim_[5],  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        
        self.skip_scale = nn.Parameter(torch.ones(1))

        self.Inception = Inception(input_dim, dim_[0], dim_[1], dim_[2], dim_[3], dim_[4], dim_[5])
        self.dim_ = dim_

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        img_dims = x.shape[2:]
        # print(img_dims)
        assert C == self.input_dim
        # n_tokens = x.shape[2:].numel()
        # img_dims = x.shape[2:]
        # x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm = self.norm(x_flat)

        # print(x.shape)
        # print(x_norm.shape)
        x1, x2, x3, x4 = self.Inception(x)

        x1 = x1.reshape(B, self.dim_[0], -1).transpose(-1, -2)
        # print(x1.shape)
        x1 = self.norm1(x1)

        x2 = x2.reshape(B, self.dim_[2], -1).transpose(-1, -2)
        x2 = self.norm2(x2)

        x3 = x3.reshape(B, self.dim_[4], -1).transpose(-1, -2)
        x3 = self.norm3(x3)

        x4 = x4.reshape(B, self.dim_[5], -1).transpose(-1, -2)
        x4 = self.norm4(x4)

        # x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        # print(self.skip_scale.shape, x1.shape)
        # print(x_norm.shape)
        # print(self.mamba1(x1).shape)
        x_mamba1 = self.mamba1(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba2(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba3(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba4(x4) + self.skip_scale * x4
        # print(x_mamba1.shape)
        # print(x_mamba2.shape)
        # print(x_mamba3.shape)
        # print(x_mamba4.shape)
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)
        # print(x_mamba.shape)
        x_mamba = self.norm(x_mamba)
        # x_mamba = self.proj(x_mamba)
        # print('dim00',self.output_dim)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        # print('out', out.shape)
        return out

class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1), 
                         self.avgpool(t2), 
                         self.avgpool(t3), 
                         self.avgpool(t4), 
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
            
        return att1, att2, att3, att4, att5
    
    
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())
    
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

    
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5
        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5
        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5
        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_
    

class SalMM(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,12,16,16,24,32],
                split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            SCPMambaLayer(input_dim=c_list[2], output_dim=c_list[3], dim_=[6, 2, 4, 4, 2, 4])  # 16
        )
        self.encoder5 = nn.Sequential(
            SCPMambaLayer(input_dim=c_list[3], output_dim=c_list[4], dim_=[9, 3, 6, 6, 3, 6])  # 24
        )
        self.encoder6 = nn.Sequential(
            SCPMambaLayer(input_dim=c_list[4], output_dim=c_list[5], dim_=[12, 4, 8, 8, 4, 8])  # 32
        )

        if bridge: 
            self.scab = SC_Att_Bridge(c_list, split_att)
            # print('SC_Att_Bridge was used')
        
        self.decoder1 = nn.Sequential(
            SCPMambaLayer(input_dim=c_list[5], output_dim=c_list[4], dim_=[9, 3, 6, 6, 3, 6])  # 24
        ) 
        self.decoder2 = nn.Sequential(
            SCPMambaLayer(input_dim=c_list[4], output_dim=c_list[3], dim_=[6, 2, 4, 4, 2, 4])  # 16
        ) 
        self.decoder3 = nn.Sequential(
            SCPMambaLayer(input_dim=c_list[3], output_dim=c_list[2], dim_=[6, 2, 4, 4, 2, 4])  # 16
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

        self.CrossModelAtt = CrossModelAtt()

        self.conv5 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        # print(t4.shape)
        # exit(0)
        # print('t4', t4.shape)
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32
        # print('t5', t5.shape)
        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32

        # ================================================================
        # Cross-Model Attention
        out = self.CrossModelAtt(x, out)
        # ================================================================

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        # print(out5.shape)

        out5 = torch.cat([out5, t5], dim=1)  # b, c4, H/32, W/32
        out5 = self.conv5(out5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        # print(out4.shape)

        out4 = torch.cat([out4, t4], dim=1)  # b, c3, H/16, W/16
        out4 = self.conv4(out4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.cat([out3, t3], dim=1)  # b, c2, H/8, W/8
        out3 = self.conv3(out3)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.cat([out2, t2], dim=1)  # b, c1, H/4, W/4
        out2 = self.conv2(out2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.cat([out1, t1], dim=1)  # b, c0, H/2, W/2
        out1 = self.conv1(out1)

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0), None
