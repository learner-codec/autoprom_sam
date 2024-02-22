import torch.nn as nn
import torch
import math
import sys
from torchvision.ops import nms
from model_utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
import losses
from functools import reduce
import torch.optim as optim
from functools import partial
import torch.nn.functional as F
sys.path.append("./SAM/")
from modeling.image_encoder import ImageEncoderViT
sys.path.append("./")
from configs import CFG


class _BNRelu(nn.Module):
    def __init__(self, num_features):
        super(_BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x

class DenseUnit(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(DenseUnit, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                _BNRelu(in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            ))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = x + out  # Sum the input and output (dense connection)
        return x



class FPN(nn.Module):
    def __init__(self,use_dense = True ,num_levels=5, hidden_dim=768, fpn_dim=256):
        super(FPN, self).__init__()
        self.use_dense = use_dense
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.fpn_dim = fpn_dim
        self.use_transposed = CFG.use_transposed
        self.lateral_stages = [0,2,5,8,11]

        #lateral conv_blocl
        self.lateral_convs = nn.ModuleList()
        self.intermediate_conv = nn.ModuleList()
        if self.use_dense:
            self.dense = nn.ModuleList()
        self.bn_relu = nn.ModuleList()
        counter = 0
        self.lateral_stages = self.lateral_stages[::-1]
        #[11,8,5,2]
        for index in range(len(self.lateral_stages)-1):
            blocks = nn.ModuleList()
            counter+=1
            print("limits are ==> ",self.lateral_stages[index+1],self.lateral_stages[index] )
            for i in range(self.lateral_stages[index+1],self.lateral_stages[index]):
                lateral_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
                relu = nn.ReLU()
                setattr(self, f"block{counter}_{i+1}_conv", lateral_conv)
                setattr(self, f"relu{counter}", relu)
                blocks.append(lateral_conv)
                blocks.append(relu)
            #upconv in last level
            if index==len(self.lateral_stages)-1:
                #we only use upconv at first level
                up_conv = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1)
                setattr(self, f"up_conv{index+1}", up_conv)
                blocks.append(up_conv)
            
            #time for dense Unit
            if index ==1:
                down = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
                setattr(self, f"down1", down)
                self.intermediate_conv.append(down)
            if index ==0:
                tmp_inter = nn.ModuleList()
                down2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
                setattr(self, f"down2", down2)
                tmp_inter.append(down2)
                relu = nn.ReLU()
                setattr(self, f"relu{counter}", relu)
                tmp_inter.append(relu)
                down3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
                setattr(self, f"down3", down3)
                tmp_inter.append(down3)
                self.intermediate_conv.append(tmp_inter)
            if self.use_dense:
                dense = DenseUnit(256, CFG.dense_units[index])
                setattr(self, f"_dense{index+1}", dense)
                self.dense.append(dense)
            bn_relu = _BNRelu(fpn_dim)
            setattr(self, f"bnrelu{index+1}", bn_relu)
            self.lateral_convs.append(blocks)
            self.bn_relu.append(bn_relu)
        print("counter is ==> ",counter)
        #final layers of convolution

        conv5_1 = nn.Conv2d(fpn_dim,fpn_dim , kernel_size=3, stride=2, padding=1)
        # Convolutional layer 2 with 256 input channels, 256 output channels, kernel size (3, 3), padding 1
        relu14 = nn.ReLU()
        conv5_2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
        # Convolutional layer 3 with 256 input channels, 256 output channels, kernel size (3, 3), padding 1
        setattr(self, f"conv5_1", conv5_1)

        setattr(self, f"relu14", relu14)
        setattr(self, f"conv5_2", conv5_2)
        self.lateral_convs.append(conv5_1)
        self.lateral_convs.append(relu14)
        self.lateral_convs.append(conv5_2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)





    def forward(self, x):
        features = x
        for i, feature in enumerate(features):
            features[i] = feature.permute(0, 3, 1, 2)
        feature = feature[:-1] #reversed
        embedding_collections = []
        num_features = len(feature)-1
        for index,layers in enumerate(zip(self.lateral_conv[:-3])):
            #first layer feature
            x_prev = None
            x=None
            x_upsampled = None
            for layer in layers:
                if x_prev == None:
                    x_prev = layer(features[num_features])
                    x_prev = self.relu[num_features](x)
                    num_features-=1
                    continue
                x = layer(features[num_features])
                x_prev = self.relu[num_features](x) +x_prev
                num_features-=1
            
            if index==0:
                x = self.intermediate_conv[0](x_prev)
                x_upsampled = F.interpolate(x, scale_factor=2, mode='nearest')
                x= self.dense[index](x)
                x = self.bn_relu[index](x)
                embedding_collections.append(x)
            elif index !=2:
                x = self.intermediate_conv[0](x_prev) + x_upsampled
                x_upsampled = F.interpolate(x, scale_factor=2, mode='nearest')
                x= self.dense[index](x)
                x = self.bn_relu[index](x)
                embedding_collections.append(x)
            elif index ==2:
                x = x_prev+x_upsampled
                x_upsampled = F.interpolate(x, scale_factor=2, mode='nearest')
                x = self.dense[index](x)
                x = self.bn_relu[index](x)
                embedding_collections.append(x)
            


        # Lateral connections
        
        #got x1=p3x,x2=p5x,x3=p8x,x4=11x
        x5 = self.lateral_convs[-3](embedding_collections[0]) #12x
        x6 = self.lateral_convs[-2](x5)
        x6 = self.lateral_convs[-1](x6)#13x


        #second global attn index 3-5
        
        
        

        #fourth global attn index 9-11

        

        return [embedding_collections[0],embedding_collections[1],embedding_collections[2],embedding_collections[3],x5,x6]#256,128,128 256,64,64 256,32,32 256,16,16

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel,self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)



class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel,self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
