# implementation is taken from: https://github.com/Yussef93/FewShotCellSegmentation

import torch.nn as nn
import torch.nn.init as init
import torch


def conv_bn_relu(in_channels, out_channels, kernel_size, affine=False):
    layer = []
    layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(out_channels, affine=affine))
    # layer.append(nn.InstanceNorm2d(out_channels,affine=affine))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def conv_bn_relu_transpose(in_channels, out_channels, kernel_size, affine=False):
    layer = []
    layer.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, bias=False))
    # layer.append(nn.InstanceNorm2d(out_channels, affine=affine))
    layer.append(nn.BatchNorm2d(out_channels, affine=affine))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


class UNet_Cells(nn.Module):
    """
    Implementation from few-shot cell segmentation.
    """

    def __init__(self, n_class, sigmoid=False, affine=False, norm_type="instance"):
        super(UNet_Cells, self).__init__()
        self.norm_type = norm_type
        # Encoder
        self.dconv_down1 = self.double_conv(1, 32, affine=affine)
        self.dconv_down2 = self.double_conv(32, 64, affine=affine)
        self.dconv_down3 = self.double_conv(64, 128, affine=affine)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.dconv_up2 = self.double_conv(64 + 128, 64, affine=affine)
        self.dconv_up1 = self.double_conv(32 + 64, 32, affine=affine)
        self.conv_last = nn.Conv2d(32, n_class, kernel_size=1)
        self.add_sigmoid = sigmoid
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def double_conv(self, in_channels, out_channels, affine):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=affine) if (self.norm_type == "instance") else nn.BatchNorm2d(
                out_channels, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=affine) if (self.norm_type == "instance") else nn.BatchNorm2d(
                out_channels, affine=affine),
            nn.ReLU(inplace=True))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        feature_distill = x
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        if self.add_sigmoid:
            out = nn.Sigmoid()(x)
        else:
            out = x
        return out, feature_distill
