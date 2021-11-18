import logging
import math
import os
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from SETR.transformer_model import TransModel2d, TransConfig
import math
import torch.nn.functional as F
from  torchvision import  models
from FaPN.DCNv2.dcn_v2 import DCN,DCN_Fuse
from self_attention_cv import AxialAttentionBlock



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x


class Encoder2D(nn.Module):

    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.in_channels=config.in_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % (sample_v**2) == 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // (sample_v**2))
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v
        self.is_segmentation = is_segmentation

        ######################################################
        # resnet=models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
        ######################################################
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=self.in_channels, ch_out=64)
        self.dcn1=DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.dcn2=DCN(128,128, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.axial1 = AxialAttentionBlock(in_channels=1024, dim=32, heads=4)
        ######################################################
    def forward(self, x):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape   # 8,3,256,256
        #######################################################
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        # e1 = self.encoder1(x)
        # e2 = self.encoder2(e1)
        # e3 = self.encoder3(e2)
        # b, c, h, w = e2.shape
        ########################################################

        x1 = self.Conv1(x)
        # x1=self.dcn1(x1)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5= self.axial1(x5)

        ########################################################
        # assert self.config.in_channels == c, "in_channels != 输入图像channel"

        p1 = self.patch_size[0]
        p2 = self.patch_size[1]

        if h % p1 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        if w % p2 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        hh = h // p1  #每张patch的大小
        ww = w // p2

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2)
        
        encode_x = self.bert_model(x)[-1] # 取出来最后一层
        if not self.is_segmentation:
            return encode_x

        x = self.final_dense(encode_x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c = self.config.hidden_size)
        return encode_x, x ,x5,x4,x3,x2,x1


class PreTrainModel(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

class Vit(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        sample_rate=4,
                        ):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

#############################################################################################

class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


#########################################################################################################

class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
     # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
     # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x

###############################################################################################
class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512,256,128,64] ):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )


        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)
        self.context1=ContextBlock(inplanes=1024, ratio=1. / 16., pooling_type='att')
        self.context2 = ContextBlock(inplanes=512, ratio=1. / 16., pooling_type='att')
        self.context3 = ContextBlock(inplanes=256, ratio=1. / 16., pooling_type='att')
        self.context4 = ContextBlock(inplanes=128, ratio=1. / 16., pooling_type='att')
        self.context5 = ContextBlock(inplanes=64, ratio=1. / 16., pooling_type='att')
        self.context5 = ContextBlock(inplanes=64, ratio=1. / 16., pooling_type='att')



        self.axial1=AxialAttentionBlock(in_channels=1024, dim=32, heads=8)
        self.axial2 = AxialAttentionBlock(in_channels=256, dim=64, heads=8)
    # def __init__(self, img_ch=3, output_ch=1):
    #     super(Decoder2D, self).__init__()
    #     self.context1 = ContextBlock(inplanes=1024, ratio=1. / 16., pooling_type='att')
    #     self.Up5 = up_conv(ch_in=1024, ch_out=512)
    #     self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
    #     self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
    #
    #     self.Up4 = up_conv(ch_in=512, ch_out=256)
    #     self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
    #     self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
    #
    #     self.Up3 = up_conv(ch_in=256, ch_out=128)
    #     self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
    #     self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
    #
    #     self.Up2 = up_conv(ch_in=128, ch_out=64)
    #     self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
    #     self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
    #
    #     self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    #     self.sigmoid = nn.Sigmoid()
    #
    # def forward(self, x,x4,x3,x2,x1):
    #     x=self.context1(x)
    #     d5 = self.Up5(x)
    #     x4 = self.Att5(g=d5, x=x4)
    #     d5 = torch.cat((x4, d5), dim=1)
    #     d5 = self.Up_conv5(d5)
    #
    #     d4 = self.Up4(d5)
    #     x3 = self.Att4(g=d4, x=x3)
    #     d4 = torch.cat((x3, d4), dim=1)
    #     d4 = self.Up_conv4(d4)
    #
    #     d3 = self.Up3(d4)
    #     x2 = self.Att3(g=d3, x=x2)
    #     d3 = torch.cat((x2, d3), dim=1)
    #     d3 = self.Up_conv3(d3)
    #
    #     d2 = self.Up2(d3)
    #     x1 = self.Att2(g=d2, x=x1)
    #     d2 = torch.cat((x1, d2), dim=1)
    #     d2 = self.Up_conv2(d2)
    #
    #     d1 = self.Conv_1x1(d2)
    #     d1 = self.sigmoid(d1)
    #     return d1
    def forward(self, x):

        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x

####################################################################################
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
##############################################################################


class up_conv(nn.Module):

    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi

###########################################################
class SETRModel(nn.Module):
    def __init__(self, patch_size=(32, 32), 
                        in_channels=3, 
                        out_channels=1, 
                        hidden_size=1024,
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64],
                        sample_rate=4,):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config)
        self.aspp=ASPP(in_channel=1024,depth=1024)
        # self.decoder_2d = Decoder2D(img_ch=3, output_ch=1)
        self.decoder_2d = Decoder2D(in_channels=config.hidden_size, out_channels=config.out_channels,
                                    features=decode_features)
        self.Up_conv = conv_block(ch_in=2048, ch_out=1024)
        self.dcn=DCN_Fuse(1024,1024,kernel_size=(3,3),stride=1,padding=1,deformable_groups=2)
    def forward(self, x):
        _, final_x,x5,x4,x3,x2,x1 = self.encoder_2d(x)
        # final_x=self.aspp(final_x)
        fx=self.dcn(final_x,x5)
        # fx=torch.cat([final_x,x5],dim=1)
        # fx=self.Up_conv(fx)
        # x = self.decoder_2d(fx,x4,x3,x2,x1)
        x = self.decoder_2d(fx)
        return x 

   

