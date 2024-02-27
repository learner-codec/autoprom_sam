import torch.nn as nn
import torch
import torch.nn.functional as F
from .configs import CFG
from copy import copy
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
    def __init__(self, in_channels, num_layers,use_bn=True):
        super(DenseUnit, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if use_bn:
            for i in range(num_layers):
                self.layers.append(nn.Sequential(
                    _BNRelu(in_channels),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                ))
        else:
            for i in range(num_layers):
                self.layers.append(nn.Sequential(
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
    def get_num_layers(self):
        return self.num_layers

class Conv_relu(nn.Module):
    def __init__(self, in_channels, out_channel,kernel_size,padding):
        super(Conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x =  self.relu(x)
        return x 



class FPN(nn.Module):
    def __init__(self,use_dense = True ,num_levels=5, hidden_dim=768, fpn_dim=256,use_transposed=True):
        super(FPN, self).__init__()
        self.use_dense = use_dense
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.fpn_dim = fpn_dim
        self.use_transposed = use_transposed
        self.encoder_layers = CFG.encoder_layers
        encoder_block = 4
        blocks = [
           [Conv_relu(hidden_dim, fpn_dim, kernel_size=3, padding=1) for _ in range(3)],
            [Conv_relu(hidden_dim, fpn_dim, kernel_size=3, padding=1) for _ in range(3)],
            [Conv_relu(hidden_dim, fpn_dim, kernel_size=3, padding=1) for _ in range(3)],
            [Conv_relu(hidden_dim, fpn_dim, kernel_size=3, padding=1) for _ in range(3)],
            # Define more blocks as needed
        ]
        if use_dense:
            print("dense units dimentions are ==> ", CFG.dense_units)
            dense_blocks = [[DenseUnit(fpn_dim,unit,use_bn=CFG.use_bn)] for index,unit in enumerate(CFG.dense_units)]
        #extend the blocks to layers
        self.layers = nn.ModuleList()
        down3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
        down2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
        down1 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
        if use_dense:
            self.dense = nn.ModuleList()
        self.bn_relu = nn.ModuleList()
        for block in blocks:
            layer = nn.Sequential(*block)
            if self.use_transposed and  encoder_block==1:
                layer.add_module("up_conv1",nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1))
            if encoder_block ==4:
                layer.add_module("down3",down3)
                layer.add_module("relu3",nn.ReLU())
                layer.add_module("down2",down2)
            if encoder_block==3:
                layer.add_module("down1",down1)
            if self.use_dense:
                print(f"dense units layers: layer num {encoder_block} layer name {dense_blocks[encoder_block-1][0].get_num_layers()}")
                self.dense.add_module(f"_dense{encoder_block}",dense_blocks[encoder_block-1][0])
            #adding batchnorm at the end of each block
            self.bn_relu.add_module(f"bn_relu{encoder_block}",_BNRelu(fpn_dim))
            self.layers.add_module(f'layer_{encoder_block}', layer)
            encoder_block-=1
        

        self.conv5_1 = nn.Conv2d(fpn_dim,fpn_dim , kernel_size=3, stride=2, padding=1)
        # Convolutional layer 2 with 256 input channels, 256 output channels, kernel size (3, 3), padding 1
        self.relu1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
        # Convolutional layer 3 with 256 input channels, 256 output channels, kernel size (3, 3), padding 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)





    def forward(self, x):
        features = x
        for i, feature in enumerate(features):
            features[i] = feature.permute(0, 3, 1, 2)
        features = copy(features[::-1])
        # Lateral connections
        fpn_features = []
        feature_index = self.encoder_layers
        x_up_sampled_prev = 0
        for layer_index,layer in enumerate(self.layers):
            x_prev = 0
            #print("layer name ==> ",layer, feature_index)
            for index,l in enumerate(layer):
                if isinstance(l,nn.Conv2d) or isinstance(l,nn.ReLU) or isinstance(l,nn.ConvTranspose2d):
                    #print("using ==> ", l, "at index", feature_index)
                    x_prev = l(x_prev)
                else:
                    x_curr = l(features[feature_index])
                    x_prev = x_curr+x_prev
                    feature_index-=1
            x_prev =  x_prev + x_up_sampled_prev
            x_up_sampled_prev = F.interpolate(x_prev, scale_factor=2, mode='nearest')
            
            fpn_features.append(x_prev)   
        if self.use_dense:
            for dense_index,dense_layer in enumerate(self.dense):
                fpn_features[dense_index] = dense_layer(fpn_features[dense_index])
        

        for bn_index,bn_relu in enumerate(self.bn_relu):
            fpn_features[bn_index] = bn_relu(fpn_features[bn_index])
        x5 = self.conv5_1(copy(fpn_features[0]))
        x6 = self.relu1(x5)
        x6 = self.conv5_2(x6)
        fpn_features.append(x5)
        fpn_features.append(x6)

        return fpn_features

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
