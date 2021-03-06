import torch
import torch.nn as nn
from functools import partial


def get_norm_layer(in_channels, num_groups=None):
    if num_groups is None:
        return nn.BatchNorm2d(num_features=in_channels, eps=1e-3, momentum=0.01)
    else:
        return nn.GroupNorm(num_channels=in_channels, eps=1e-3, num_groups=num_groups)


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == \
               len(self.model_cfg.NUM_FILTERS) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        num_groups = self.model_cfg.get('NUM_GROUPS', None)

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                get_norm_layer(num_filters[idx], num_groups),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    get_norm_layer(num_filters[idx], num_groups),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    num_filters[idx], num_upsample_filters[idx],
                    upsample_strides[idx],
                    stride=upsample_strides[idx], bias=False
                ),
                get_norm_layer(num_upsample_filters[idx], num_groups),
                nn.ReLU()
            ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                get_norm_layer(c_in, num_groups),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            ups.append(self.deblocks[i](x))

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
