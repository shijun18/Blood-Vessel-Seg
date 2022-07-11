from functools import partial

import torch
import torch.nn as nn

from timm.models.efficientnet import EfficientNet
from timm.models.efficientnet import decode_arch_def, round_channels, default_cfgs
from timm.models.layers.activations import Swish
from torch.hub import load_state_dict_from_url

model_urls = {
    'timm_efficientnet_b0': default_cfgs["tf_efficientnet_b0_ap"]["url"],
    'timm_efficientnet_b1': default_cfgs["tf_efficientnet_b1_ap"]["url"],
    'timm_efficientnet_b2': default_cfgs["tf_efficientnet_b2_ap"]["url"],
    'timm_efficientnet_b3': default_cfgs["tf_efficientnet_b3_ap"]["url"],
    'timm_efficientnet_b4': default_cfgs["tf_efficientnet_b4_ap"]["url"],
    'timm_efficientnet_b5': default_cfgs["tf_efficientnet_b5_ap"]["url"],
    'timm_efficientnet_b6': default_cfgs["tf_efficientnet_b6_ap"]["url"],
    'timm_efficientnet_b7': default_cfgs["tf_efficientnet_b7_ap"]["url"],
    'timm_efficientnet_b8': default_cfgs["tf_efficientnet_b8_ap"]["url"],
    'timm_efficientnet_l2': default_cfgs["tf_efficientnet_l2_ns"]["url"],
    'timm_efficientnet_lite0': default_cfgs["tf_efficientnet_lite0"]["url"],
    'timm_efficientnet_lite1': default_cfgs["tf_efficientnet_lite1"]["url"],
    'timm_efficientnet_lite2': default_cfgs["tf_efficientnet_lite2"]["url"],
    'timm_efficientnet_lite3': default_cfgs["tf_efficientnet_lite3"]["url"],
    'timm_efficientnet_lite4': default_cfgs["tf_efficientnet_lite4"]["url"],
}




def get_efficientnet_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """Create EfficientNet model.
    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946
    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=Swish,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs


def gen_efficientnet_lite_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """EfficientNet-Lite model.
    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946
    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),
    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16"],
        ["ir_r2_k3_s2_e6_c24"],
        ["ir_r2_k5_s2_e6_c40"],
        ["ir_r3_k3_s2_e6_c80"],
        ["ir_r3_k5_s1_e6_c112"],
        ["ir_r4_k5_s2_e6_c192"],
        ["ir_r1_k3_s1_e6_c320"],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=nn.ReLU6,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs


class EfficientNetBaseEncoder(EfficientNet):
    def __init__(self, out_channels,n_channels=1, **kwargs):
        super().__init__(in_chans=n_channels,**kwargs)

        self.out_channels = out_channels
        self.in_chans = n_channels

        del self.classifier

    def get_stages(self):
        return [
            nn.Sequential(self.conv_stem, self.bn1, self.act1),
            self.blocks[:2],
            self.blocks[2:3],
            self.blocks[3:5],
            self.blocks[5:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for layer in stages:
            x = layer(x)
            features.append(x)

        return features



class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(
        self,
        out_channels,
        n_channels,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = get_efficientnet_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(out_channels, n_channels, **kwargs)


class EfficientNetLiteEncoder(EfficientNetBaseEncoder):
    def __init__(
        self,
        out_channels,
        n_channels,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = gen_efficientnet_lite_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(out_channels, n_channels, **kwargs)


def get_efficientnet(arch,bulider,out_channels, channel_multiplier, depth_multiplier, drop_rate, pretrained, progress,**kwargs):
    model = bulider(out_channels=out_channels, channel_multiplier=channel_multiplier, depth_multiplier=depth_multiplier, drop_rate=drop_rate,**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if kwargs['n_channels'] != 3:
            mis_key = ['conv_stem.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'bn1.bias']
            for key in mis_key:
                del state_dict[key]
        model.load_state_dict(state_dict,strict=False)
        # model.load_state_dict(state_dict)
    return model

def timm_efficientnet_b0(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b0',EfficientNetEncoder,
    (32, 24, 40, 112, 320), 1.0, 1.0, 0.2, pretrained, progress,**kwargs)
    return model

def timm_efficientnet_b1(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b1',EfficientNetEncoder,
        (32, 24, 40, 112, 320), 1.0, 1.1, 0.2, pretrained, progress,**kwargs)
    return model

def timm_efficientnet_b2(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B2 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b2',EfficientNetEncoder,
        (32, 24, 48, 120, 352), 1.1, 1.2, .3, pretrained, progress,**kwargs)
    return model

def timm_efficientnet_b3(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B3 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b3',EfficientNetEncoder,
        (40, 32, 48, 136, 384), 1.2, 1.4, 0.3, pretrained, progress,**kwargs)
    return model

def timm_efficientnet_b4(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B4 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b4',EfficientNetEncoder,
        (48, 32, 56, 160, 448), 1.4, 1.8, 0.4, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_b5(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B5 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b5',EfficientNetEncoder,
        (48, 40, 64, 176, 512), 1.6, 2.2, 0.4, pretrained, progress,**kwargs)
    return model

def timm_efficientnet_b6(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B6 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b6',EfficientNetEncoder,
        (56, 40, 72, 200, 576), 1.8, 2.6, 0.5, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_b7(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B7 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b7',EfficientNetEncoder,
        (64, 48, 80, 224, 640), 2.0, 3.1, 0.5, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_b8(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-B8 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_b8',EfficientNetEncoder,
        (72, 56, 88, 248, 704), 2.2, 3.6, 0.5, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_l2(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-l2 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_l2',EfficientNetEncoder,
        (136, 104, 176, 480, 1376), 4.3, 5.3, 0.5, pretrained, progress,**kwargs)
    return model

def timm_efficientnet_lite0(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_lite0',EfficientNetLiteEncoder,
        (32, 24, 40, 112, 320), 1.0, 1.0, 0.2, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_lite1(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-lite1"""
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_lite1',EfficientNetLiteEncoder,
        (32, 24, 40, 112, 320), 1.0, 1.1, 0.2, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_lite2(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-lite2"""
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_lite2',EfficientNetLiteEncoder,
        (32, 24, 48, 120, 352), 1.1, 1.2, 0.3, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_lite3(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-lite3"""
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_lite3',EfficientNetLiteEncoder,
        (32, 32, 48, 136, 384), 1.2, 1.4, 0.3, pretrained, progress,**kwargs)
    return model


def timm_efficientnet_lite4(pretrained=False, progress=True,**kwargs):
    """ EfficientNet-lite4"""
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = get_efficientnet('timm_efficientnet_lite4',EfficientNetLiteEncoder,
        (32, 32, 56, 160, 448), 1.4, 1.8, 0.4, pretrained, progress,**kwargs)
    return model

