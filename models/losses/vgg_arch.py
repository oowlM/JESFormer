import os
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import vgg as vgg

VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'
NAMES = {
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
    ]
}


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        layer_name_list,
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=False,
        requires_grad=False,
        remove_pooling=False,
        pooling_stride=2,
    ):
        super().__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.names = NAMES[vgg_type]

        max_idx = max(self.names.index(value) for value in layer_name_list)
        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(weights=None)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            weights_enum = getattr(vgg, f'{vgg_type.upper()}_Weights')
            vgg_net = getattr(vgg, vgg_type)(weights=weights_enum.IMAGENET1K_V1)

        features = vgg_net.features[:max_idx + 1]
        modified_net = OrderedDict()
        for name, layer in zip(self.names, features):
            if 'pool' in name:
                if remove_pooling:
                    continue
                modified_net[name] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[name] = layer
        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()

        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        return output
