import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import BasicLayer
import numpy as np


class Conv_Stem(nn.Module):
    def __init__(self,input_chanel,output_chanel,norm=nn.BatchNorm3d):
        super(Conv_Stem,self).__init__()
        self.conv1 = nn.Conv3d(input_chanel,output_chanel,kernel_size=7,stride=1,padding=3,bias=False)
        self.bn1 = norm(output_chanel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x= self.relu(x)
        x = self.maxpool(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=(1,1),downsample=None,norm=nn.BatchNorm3d):
        super(ResBlock,self).__init__()
        #(in_planes, out_planes, kernel_size=3, stride=stride=1,padding=dilation=1, groups=groups=1, bias=False, dilation=dilation)
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=1,bias=False)
        self.bn1 = norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=1,bias=False)
        self.bn2 = norm(out_channels)
        self.downsample = downsample

    def forward(self,x):
        identity = x
      
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
     
        out += identity
        out = self.relu(out)
        return out

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)+x

class MSCB(nn.Module):
    def __init__(self,in_channel=2,out_channel=64,kernel_size = [2,4,8,16],norm_layer=nn.LayerNorm,norm_first=False,use_eca=False):
        super(MSCB,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm_first = norm_first
        self.use_eca = use_eca
        # self.eca = eca_layer(k_size=3)

        if norm_layer is not None and not norm_first:
            self.norm = norm_layer(out_channel)
        elif norm_layer is not None and norm_first:
            self.norm = norm_layer(in_channel)

        else:
            self.norm = None

        self.projs = nn.ModuleList()
        for i, ps in enumerate(kernel_size):
            if i == len(kernel_size) - 1:
                dim = out_channel // 2 ** i
            else:
                dim = out_channel // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.projs.append(nn.Conv3d(in_channel, dim, kernel_size=ps, stride=stride, padding=padding))

        if use_eca:
            self.eca = eca_layer(k_size=3)
        
    def forward(self, x):
        if self.norm and self.norm_first:
            B,C,H,W,D = x.shape
            x = x.flatten(2).transpose(1, 2) # to sequence
            x = self.norm(x)

            x = x.transpose(1, 2).view(B,C,H,W,D) #  to image

        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x)  
            xs.append(tx)  
        x = torch.cat(xs, dim=1) 

        if self.norm is not None and not self.norm_first:
            if self.use_eca:
                x = self.eca(x)
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2) # to sequence
            x = self.norm(x)
            
            wh, ww ,wd = x_shape[2], x_shape[3], x_shape[4]
            x = x.transpose(1, 2).view(-1, self.out_channel, wh, ww ,wd)#  to image
            
        else:
            if self.use_eca:
                x = self.eca(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels,norm=nn.BatchNorm3d):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels, kernel_size=3, padding=1), norm(out_channels), nn.ReLU(inplace=True))   
            #nn.Conv3d(2*out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.GELU()) 
           
    def forward(self, x1, x2):
         
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x  

class MFormer(nn.Module):
    def __init__(self,input_chanel=2,image_size=(96,96,96),num_classes=3,use_eca=False,return_all=False,norm = nn.BatchNorm3d):
        super(MFormer,self).__init__()
        
        embed_dim = 64

        self.image_size = image_size
        self.use_eca = use_eca
        self.return_all =return_all
        # define Conv_Stem
        self.conv_stem = Conv_Stem(input_chanel=input_chanel,output_chanel=embed_dim,norm=norm)
        ######

        # define PatchEmbeding and PatchMerging downsample using Multi-Scale Convolutional Block
        self.MS1 = MSCB(in_channel=input_chanel,out_channel=embed_dim,kernel_size=[2,4,8,16],norm_layer=nn.LayerNorm,norm_first=False,use_eca=self.use_eca)
        self.MS2 = MSCB(in_channel=embed_dim,out_channel=embed_dim*2,kernel_size=[2,4],norm_layer=nn.LayerNorm,norm_first=True,use_eca=self.use_eca)
        self.MS3 = MSCB(in_channel=embed_dim*2,out_channel=embed_dim*4,kernel_size=[2,4],norm_layer=nn.LayerNorm,norm_first=True,use_eca=self.use_eca)
        self.MS4 = MSCB(in_channel=embed_dim*4,out_channel=embed_dim*8,kernel_size=[2,4],norm_layer=nn.LayerNorm,norm_first=True,use_eca=self.use_eca)
        # #######

        # define resblock 
        self.resblock_layernums = [3,4,6,3]
        self.resblock1 = ResBlock(in_channels=embed_dim,out_channels=embed_dim,stride=(1,1))
        
        self.downsample2 = nn.Sequential(nn.Conv3d(embed_dim,embed_dim*2,kernel_size=1,stride=2,bias=False),norm(embed_dim*2))
        self.resblock2_1 = ResBlock(in_channels=embed_dim,out_channels=embed_dim*2,stride=(2,1),downsample=self.downsample2,norm=norm)
        self.resblock2_2 = ResBlock(in_channels=embed_dim*2,out_channels=embed_dim*2,stride=(1,1),norm=norm)

        self.downsample3 = nn.Sequential(nn.Conv3d(embed_dim*2,embed_dim*4,kernel_size=1,stride=2,bias=False),norm(embed_dim*4))
        self.resblock3_1 = ResBlock(in_channels=embed_dim*2,out_channels=embed_dim*4,stride=(2,1),downsample=self.downsample3,norm=norm)
        self.resblock3_2 = ResBlock(in_channels=embed_dim*4,out_channels=embed_dim*4,stride=(1,1),norm=norm)

        self.downsample4 = nn.Sequential(nn.Conv3d(embed_dim*4,embed_dim*8,kernel_size=1,stride=2,bias=False),norm(embed_dim*8))
        self.resblock4_1 = ResBlock(in_channels=embed_dim*4,out_channels=embed_dim*8,stride=(2,1),downsample=self.downsample4,norm=norm)
        self.resblock4_2 = ResBlock(in_channels=embed_dim*8,out_channels=embed_dim*8,stride=(1,1),norm=norm)
        ######

        # define SwinTransformer
        self.swin_layers = nn.ModuleList()
        num_layers = 4
        depths=[2, 2, 2, 2]
        num_heads=[2, 4, 8, 16]
        window_size = np.array(self.image_size) //16
        mlp_ratio = 4.0
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(num_layers):
            
            swin_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,mlp_ratio=mlp_ratio,
                               qkv_bias=True, drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swin_layers.append(swin_layer)
        ######
        # define decoder
        channels = [64,128,256,512]
        self.decode4 = Decoder(channels[3],channels[2],norm)
        self.decode3 = Decoder(channels[2],channels[1],norm)
        self.decode2 = Decoder(channels[1],channels[0],norm)
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                                     nn.Conv3d(channels[0], num_classes, kernel_size=1,bias=False))
        #######

    def forward(self,x):
        encoder=[]

        ms_patch = self.MS1(x)
        x = self.conv_stem(x) 
        ############################Feature Fusion Block################################
        #### resblock1 ####
        for i in range(self.resblock_layernums[0]):
            x = self.resblock1(x)
        ###################
        x = x+ms_patch 
        x = self.swin_layers[0] (x)
        encoder.append(x)
        ####################################################################

        ###########################Feature Fusion Block##################################
        ms2 = self.MS2(x) #patch Megging
        #### resblock2 ####
        for i in range(self.resblock_layernums[1]):
            if i == 0:
                x = self.resblock2_1(x)
            else :
                x = self.resblock2_2(x)
        ###################
        x = x+ms2
        x = self.swin_layers[1] (x)
        encoder.append(x)
        ####################################################################

        ###########################Feature Fusion Block##################################
        ms3 = self.MS3(x)
        #### resblock3 ####
        for i in range(self.resblock_layernums[2]):
            if i == 0:
                x = self.resblock3_1(x)
            else :
                x = self.resblock3_2(x)
        ###################
        x = x+ms3
        x = self.swin_layers[2](x)
        encoder.append(x)
        ####################################################################


        ###########################Feature Fusion Block##################################
        ms4 = self.MS4(x)
        #### resblock4 ####
        for i in range(self.resblock_layernums[3]):
            if i == 0:
                x = self.resblock4_1(x)
            else :
                x = self.resblock4_2(x)
        ###################
        x = x+ms4
        x = self.swin_layers[3](x)
        encoder.append(x)
        ####################################################################


        d4 = self.decode4(encoder[3], encoder[2]) 
        d3 = self.decode3(d4, encoder[1]) 
        d2 = self.decode2(d3, encoder[0]) 
        out = self.decode0(d2)

        if self.return_all:
            encoder.extend([d4,d3,d2,out])
            return encoder
        else:
            return out
        
class FusionBlcok(nn.Module):
    def __init__(self, in_channels_1=4, n_classes=2, is_deconv=True, is_batchnorm=True,norm=nn.BatchNorm3d):
        super(FusionBlcok, self).__init__()

        self.in_channels_1 = in_channels_1  
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters_base = 128


        self.Conv3d3x3 = nn.Sequential(nn.Conv3d(self.in_channels_1, filters_base, kernel_size=3, stride=1, padding=1),
                                       norm(filters_base),
                                       nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool3d(2)

        self.Conv3d3x3_2 = nn.Sequential(nn.Conv3d(filters_base, filters_base, kernel_size=3, stride=1, padding=1),
                                         norm(filters_base),
                                         nn.ReLU(inplace=True))

        self.upConv3d2x2 = nn.ConvTranspose3d(filters_base, filters_base, kernel_size=2, stride=2, padding=0)



        self.datafusionfinal = nn.Conv3d(filters_base, n_classes, 1)




    def forward(self, inputs1, inputs2, inputs3=None):
        if inputs3 is None:
            intput = torch.cat([inputs1, inputs2], 1)
        else:
            intput = torch.cat([inputs1, inputs2, inputs3], 1)
        conv3x3 = self.Conv3d3x3(intput)  #4 ->128
        maxpool_1 = self.maxpool(conv3x3)   # 128->128
        conv3x3_2 = self.Conv3d3x3_2(maxpool_1)
        up3d2x2 = self.upConv3d2x2(conv3x3_2) # 128
        conv3x3_3 = self.Conv3d3x3_2(up3d2x2) # 128->128
        datafusion =  self.datafusionfinal (conv3x3_3) #128->2

        # datafusion_output = nn.Sigmoid()(datafusion)

        return  datafusion

class MA4BTS(nn.Module):
    def __init__(self,model_tumor_input=2,model_gland_input=1,image_size=(96,96,96),num_classes=3,model_tumor_out=3,model_gland_out=3) -> None:
        super(MA4BTS,self).__init__()
        self.model_tumor_input = model_tumor_input
        self.model_gland_input = model_gland_input

        self.model_for_tumor = MFormer(input_chanel=model_tumor_input,image_size=image_size,num_classes=model_tumor_out,norm=nn.InstanceNorm3d)

        self.model_for_gland = MFormer(input_chanel=model_gland_input,image_size=image_size,num_classes=model_gland_out,norm=nn.InstanceNorm3d)

        self.model_for_fusion = FusionBlcok(in_channels_1=num_classes + model_gland_out +model_tumor_out, n_classes=num_classes,norm=nn.InstanceNorm3d)

    def forward(self,x,return_all=False):
        x1 = x[:,:self.model_tumor_input,:,:,:]
        x2 = x[:,self.model_tumor_input:self.model_tumor_input+self.model_gland_input,:,:,:]
        x3 = x[:,self.model_tumor_input+self.model_gland_input:,:,:,:]
        tumor = self.model_for_tumor(x1)
        gland = self.model_for_gland(x2)
        fusion = self.model_for_fusion(tumor,gland,x3)

        if return_all:
            return [tumor,gland,fusion]
        else:

            return fusion

