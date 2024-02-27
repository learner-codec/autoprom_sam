import torch.nn as nn
import torch
import math
import sys
from torchvision.ops import nms
from .model_utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from .anchors import Anchors
from functools import reduce
import torch.optim as optim
from functools import partial
import torch.nn.functional as F
from .SAM.modeling.image_encoder import ImageEncoderViT
from ..configs.configs import CFG


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
                #_BNRelu(in_channels),
                nn.ReLU(),
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
        self.use_dense = CFG.use_dense
        if not self.use_dense:
            print("not using dense layer")
        else:
            print("*********** using dense layers *********************")
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.fpn_dim = fpn_dim
        self.use_transposed = CFG.use_transposed

        self.block1_1_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.block1_2_conv = nn.Conv2d(hidden_dim,fpn_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.block1_3_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        #using interpolate
        if self.use_transposed:
            self.up_conv1 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1)
        #first dense unit
        if self.use_dense:
            self._dense1 = DenseUnit(256, CFG.dense_units[0])
        self.bnrelu1 = _BNRelu(fpn_dim)


        self.block2_1_conv = nn.Conv2d(hidden_dim,fpn_dim, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.block2_2_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.block2_3_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.block2_4_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.block2_5_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()
        if self.use_dense:
            self._dense2 = DenseUnit(256, CFG.dense_units[1])
        self.bnrelu2 = _BNRelu(fpn_dim)


        self.block3_1_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU()
        self.block3_2_conv = nn.Conv2d(hidden_dim,fpn_dim, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU()
        self.block3_3_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.down1 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
        if self.use_dense:
            self._dense3 = DenseUnit(256, CFG.dense_units[3])
        self.bnrelu3 = _BNRelu(fpn_dim)

        self.block4_1_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.block4_2_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.block4_3_conv = nn.Conv2d(hidden_dim, fpn_dim, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
        self.relu12 = nn.ReLU()
        self.relu13 = nn.ReLU()
        self.down3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
        if self.use_dense:
            self._dense4 = DenseUnit(256, CFG.dense_units[3])
        self.bnrelu4 = _BNRelu(fpn_dim)

        self.conv5_1 = nn.Conv2d(fpn_dim,fpn_dim , kernel_size=3, stride=2, padding=1)
        # Convolutional layer 2 with 256 input channels, 256 output channels, kernel size (3, 3), padding 1
        self.relu14 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
        # Convolutional layer 3 with 256 input channels, 256 output channels, kernel size (3, 3), padding 1
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    def forward(self, x):
        features = x
        for i, feature in enumerate(features):
            features[i] = feature.permute(0, 3, 1, 2)
        # Lateral connections
        x4_1 = self.block4_1_conv(features[11])
        x4_1 = self.relu10(x4_1)
        x4_2 = self.block4_2_conv(features[10])
        x4_2 = self.relu11(x4_2) + x4_1
        x4_3 = self.block4_3_conv(features[9])
        x4_3 = self.relu12(x4_3) + x4_2
        x4 = self.down2(x4_3)
        x4 = self.relu13(x4)
        x4 = self.down3(x4)
        x4_updampled = F.interpolate(x4, scale_factor=2, mode='nearest')
        if self.use_dense:
            x4 = self._dense4(x4)
        x4 = self.bnrelu4(x4)


        #third global attn index 6-8

        x3_1 = self.block3_1_conv(features[8])
        x3_1 =self.relu7(x3_1)
        x3_2 = self.block3_2_conv(features[7])
        x3_2 =self.relu8(x3_2) + x3_1
        x3_3 = self.block3_3_conv(features[6])
        x3 =self.relu9(x3_3) + x3_2
        x3 = self.down1(x3)
        x3 = x3+x4_updampled
        x3_updampled = F.interpolate(x3, scale_factor=2, mode='nearest')
        if self.use_dense:
            x3 = self._dense3(x3)
        x3  = self.bnrelu3(x3)



        x2_1 = self.block2_1_conv(features[5])
        x2_1 = self.relu4(x2_1)
        x2_2 = self.block2_2_conv(features[4])
        x2_2 = self.relu5(x2_2) + x2_1
        x2_3 = self.block2_3_conv(features[3])
        x2 = self.relu6(x2_3) + x2_2
        x2 = x2 + x3_updampled
        x2_updampled = F.interpolate(x2, scale_factor=2, mode='nearest')
        if self.use_dense:
            x2 = self._dense2(x2)
        x2 = self.bnrelu2(x2)

        # Lateral connections
        x1_1 = self.block1_1_conv(features[2])
        x1_1 = self.relu1(x1_1)
        x1_2 = self.block1_2_conv(features[1])
        x1_2 = self.relu2(x1_2) + x1_1
        x1_3 = self.block1_3_conv(features[0])
        x1_3 = self.relu3(x1_3) + x1_2
        if self.use_transposed:
            x1 = self.up_conv1(x1_3)
        else:
            x1 = F.interpolate(x1_3, scale_factor=2, mode='nearest')
        x1 = x1 + x2_updampled
        if self.use_dense:
            x1 = self._dense1(x1)
        x1 = self.bnrelu1(x1)
        #got x1=p3x,x2=p5x,x3=p8x,x4=11x
        x5 = self.conv5_1(x4) #12x
        x6 = self.relu14(x5)
        x6 = self.conv5_2(x6)#13x


        #second global attn index 3-5
        #fourth global attn index 9-11
        return [x1,x2,x3,x4,x5,x6]#256,128,128 256,64,64 256,32,32 256,16,16

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
        self.fatter = CFG.fatter
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
        if self.fatter:
            self.conv5 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
            self.act5 = nn.ReLU()
            
            self.conv6 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
            self.act6 = nn.ReLU()

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
        ########### This part is used in kidney dataset #################
        ############################################
        if self.fatter:
            out = self.conv5(out)
            out = self.act5(out)
            out = self.conv6(out)
            out = self.act6(out)
        ##################################################

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
