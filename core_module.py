import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cuda
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from einops import rearrange
from torch.cuda.amp import autocast
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import modules as md


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
        nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d,
        nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class Attention_S(nn.Module):
    def __init__(self, dim, num_heads, bias,):
        super(Attention_S, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv1conv_1 = nn.Conv2d(dim,dim,kernel_size=1)
        self.qkv2conv_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.qkv3conv_1 = nn.Conv2d(dim, dim, kernel_size=1)


        self.qkv1conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)


        self.conv_3      = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.conv_5      = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_s = torch.cat((self.qkv1conv_3(self.qkv1conv_1(x)),self.qkv1conv_5(self.qkv1conv_1(x))),1)
        k_s = torch.cat((self.qkv2conv_3(self.qkv2conv_1(x)),self.qkv2conv_5(self.qkv2conv_1(x))),1)
        v_s = torch.cat((self.qkv3conv_3(self.qkv3conv_1(x)),self.qkv3conv_5(self.qkv3conv_1(x))),1)

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = (attn_s @ v_s)
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_s_l = torch.cat((self.conv_3(x),self.conv_5(x)),1)
        out = self.project_out(torch.cat((out_s,out_s_l),1))

        return out

    def initialize(self):
        weight_init(self)


class PIAB(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1, dilation=1),
                                     nn.BatchNorm2d(in_channel // 4), nn.ReLU())

        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 2, dilation=2),
                                     nn.BatchNorm2d(in_channel // 4), nn.ReLU())

        self.conv5_1 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 3, dilation=3),
                                     nn.BatchNorm2d(in_channel // 4), nn.ReLU())

        self.conv7_1 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 4, dilation=4),
                                     nn.BatchNorm2d(in_channel // 4), nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                   nn.BatchNorm2d(in_channel // 4), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                   nn.BatchNorm2d(in_channel // 4), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                   nn.BatchNorm2d(in_channel // 4), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                   nn.BatchNorm2d(in_channel // 4), nn.ReLU())

        self.attn_S_1 = nn.Sequential(Attention_S(in_channel//4, num_heads=8, bias=False))
        self.attn_S_2 = nn.Sequential(Attention_S(in_channel//4, num_heads=8, bias=False))
        self.attn_S_3 = nn.Sequential(Attention_S(in_channel//4, num_heads=8, bias=False))
        self.attn_S_4 = nn.Sequential(Attention_S(in_channel//4, num_heads=8, bias=False))

        self.conv0 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel // 4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel // 4), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel // 4), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel // 4), nn.ReLU())
        self.conv10 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)

    def forward(self,x):
        x1, x3, x5, x7 = torch.split(x, self.in_channel // 4, dim=1)

        x1 = self.conv0(x1 + x3)
    
        x1_1 = self.conv1_1(x1)
        x1_1 = self.attn_S_1(x1_1)
        x1 = self.conv1(x1_1)

        x3 = self.conv2(x1 + x3 + x5)

        x3_1 = self.conv3_1(x3)
        x3_1 = self.attn_S_2(x3_1)
        x3 = self.conv3(x3_1)

        x5 = self.conv4(x3 + x5 + x7)

        x5_1 = self.conv5_1(x5)
        x5_1 = self.attn_S_3(x5_1)
        x5 = self.conv5(x5_1)

        x7 = self.conv6(x5 + x7)

        x7_1 = self.conv7_1(x7)
        x7_1 = self.attn_S_4(x7_1)
        x7 = self.conv7(x7_1)
        
        x = self.conv10(torch.cat((x1, x3, x5, x7), dim=1)) + x

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图


class FGM(nn.Module):
    def __init__(self,dim_in=64,dim_add=64):
        super().__init__()
        self.dim_in = dim_in
        self.dim_add = dim_add

        self.SG = nn.Sequential(nn.Conv2d(self.dim_in, 1, kernel_size=3,stride=1,padding=1),nn.Sigmoid())
        self.CG1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Sigmoid())
        self.CG2 = nn.Sequential(nn.Conv2d(self.dim_in,self.dim_add,kernel_size=1),nn.AdaptiveAvgPool2d((1,1)),nn.Sigmoid())

    def forward(self,x1,x2):
        b,c,h,w = x1.shape     
        CG1 = self.CG1(x1)
        CG2 = self.CG2(x1)
        SG = self.SG(x1)       
        x2 = x2*SG*nn.Parameter(torch.ones(1,1,h,w),requires_grad=True).cuda() + x2*CG2*nn.Parameter(torch.ones(1,self.dim_add,1,1),requires_grad=True).cuda() + x2
        return x1,x2      


class MGM(nn.Module):
    def __init__(self,dim_in):
        super().__init__()
        self.CBAM1 = CBAM(dim_in)
        self.CBAM2 = CBAM(dim_in)
        self.CBAM3 = CBAM(dim_in)
        self.cbr = nn.Sequential(CBAM(dim_in*3),nn.Conv2d(dim_in*3,dim_in,1,1,0),nn.Conv2d(dim_in,dim_in,3,1,1),nn.BatchNorm2d(dim_in),nn.ReLU())

    def forward(self,x,mask):
        b,c,h,w = x.shape
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
        x_f = x*mask
        x_b = x*(1-mask)

        x_f = self.CBAM1(x_f)
        x_b = self.CBAM2(x_b)
        x = self.CBAM3(x)
        x = self.cbr(torch.cat((x,x_f,x_b),dim=1))

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation) 


class DecoderBlock(nn.Module):
    def __init__(self,cin,cadd,cout,):
        super().__init__()
        self.cin = (cin + cadd)
        self.cout = cout
        self.conv1 = md.Conv2dReLU(self.cin,self.cout,kernel_size=3,padding=1,use_batchnorm=True)
        self.conv2 = md.Conv2dReLU(self.cout,self.cout,kernel_size=3,padding=1,use_batchnorm=True)
        if cadd != 0:
            self.FGM = FGM(dim_in=cin,dim_add=cadd)

    def forward(self, x1, x2=None,mask=None):
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None or mask is not None:
            x1,x2 = self.FGM(x1,x2)   
            x1 = torch.cat([x1, x2], dim=1)        
        x1 = self.conv1(x1[:,:self.cin]) 
        x1 = self.conv2(x1)           

        return x1


class ConvBNReLU(nn.Module):
    def __init__(self,in_c,out_c,ks,stride=1,norm=True,res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False)
        self.res = res
    def forward(self,x):
        if self.res:
            return (x + self.conv(x))
        else:
            return self.conv(x)


class FUSE1(nn.Module):
    def __init__(self,in_channels_list=(64,128,320,512)):#64,128,320,512
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2],in_channels_list[2],1)
        self.c32 = ConvBNReLU(in_channels_list[3],in_channels_list[2],1)
        self.c33 = ConvBNReLU(in_channels_list[2],in_channels_list[2],3)

        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

        self.MG1 = MGM(in_channels_list[3])
        self.MG2 = MGM(in_channels_list[2])
        self.MG3 = MGM(in_channels_list[1])

    def forward(self,x,mask=None):
        x, x1, x2, x3 = x
        h,w = x2.shape[-2:]

        x3 = self.MG1(x3,mask)

        x2 = self.c33((F.interpolate(self.c32(x3),size=(h,w))+self.c31(x2)))
        h,w = x1.shape[-2:]

        x2 = self.MG2(x2,mask)

        x1 = self.c23((F.interpolate(self.c22(x2),size=(h,w))+self.c21(x1)))
        h,w = x.shape[-2:]

        x1 = self.MG3(x1,mask)

        x = self.c13((F.interpolate(self.c12(x1),size=(h,w))+self.c11(x)))

        return [x,x1,x2,x3]


class FUSE2(nn.Module):
    def __init__(self,in_channels_list=(64,128,320)):
        super(FUSE2, self).__init__()

        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

        self.MG1 = MGM(in_channels_list[2])
        self.MG2 = MGM(in_channels_list[1])


    def forward(self,x,mask=None):
        x,x1,x2 = x
        h,w = x1.shape[-2:]

        x2 = self.MG1(x2,mask)

        x1 = self.c23((F.interpolate(self.c22(x2),size=(h,w),mode='bilinear',align_corners=True)+self.c21(x1)))
        h,w = x.shape[-2:]

        x1 = self.MG2(x1,mask)

        x = self.c13((F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x)))
        return x,x1,x2


class FUSE3(nn.Module):
    def __init__(self,in_channels_list=(64,128)):
        super(FUSE3, self).__init__()

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

        self.MG1 = MGM(in_channels_list[1])

    def forward(self,x,mask=None):
        x,x1 = x

        x1 = self.MG1(x1, mask) 

        h,w = x.shape[-2:]
    
        x = self.c13((F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x)))

        return x,x1


class PGAD(nn.Module):
    def __init__(self, encoder_channels, decoder_channels,seg_channels = [512,320,128,64]):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [64]     
        self.out_channels = decoder_channels
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1])
        self.decoder_convs = nn.ModuleDict(decoder_convs)
        self.seg_head = nn.ModuleList(SegmentationHead(in_channels=seg_channels[i],out_channels=1,upsampling=1) for i in range(4))
        self.seg_out = SegmentationHead(in_channels=60,out_channels=1,upsampling=2)

    def forward(self, features):
        decoder_features = {}

        mask_0 = self.seg_head[0](features[3])
 
        features = self.fuse1(features,mask_0)[::-1]         


        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0],features[1])
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1],features[2])
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2],features[3])

        mask_1 = self.seg_head[1](decoder_features["x_0_0"])

        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2((decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]),mask_1)

        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](decoder_features["x_0_0"], torch.cat((decoder_features["x_1_1"], features[2]),1))
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](decoder_features["x_1_1"], torch.cat((decoder_features["x_2_2"], features[3]),1))

        mask_2 = self.seg_head[2](decoder_features["x_0_1"])

        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3((decoder_features["x_1_2"], decoder_features["x_0_1"]),mask_2)


        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](decoder_features["x_0_1"], torch.cat((decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]),1))


        mask_3 = self.seg_head[3](decoder_features["x_0_2"])

        out = self.decoder_convs["x_0_3"](torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"]),1))
        mask = self.seg_out(out)       
       
        mask_0 = F.interpolate(mask_0, size=(512,512), mode='bilinear', align_corners=True)
        mask_1 = F.interpolate(mask_1, size=(512,512), mode='bilinear', align_corners=True)
        mask_2 = F.interpolate(mask_2, size=(512,512), mode='bilinear', align_corners=True)
        mask_3 = F.interpolate(mask_3, size=(512,512), mode='bilinear', align_corners=True)
        
        return mask,mask_3,mask_2,mask_1,mask_0