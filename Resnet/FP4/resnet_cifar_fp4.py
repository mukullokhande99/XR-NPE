import torch
import torch.nn as nn
import torch.nn.functional as F
from fp4_quant import FP4Quantizer

class ConvBn(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, groups=1, bias=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))

class QuantConvBn(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, groups=1, bias=False,
                 w_per_channel=True, enable_wq=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.wq = FP4Quantizer(mode="weight", per_channel=w_per_channel, ch_axis=0, enabled=enable_wq)

    def forward(self, x):
        w = self.conv.weight
        if self.wq.enabled:
            qw = self.wq(w)
        else:
            qw = w
        out = F.conv2d(x, qw, self.conv.bias, stride=self.conv.stride,
                       padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        out = self.bn(out)
        return out

class ActQuant(nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.aq = None
        if enabled:
            from fp4_quant import FP4Quantizer
            self.aq = FP4Quantizer(mode="activation", per_channel=False, enabled=True)

    def enable(self, flag=True):
        self.enabled = flag
        if flag and self.aq is None:
            from fp4_quant import FP4Quantizer
            self.aq = FP4Quantizer(mode="activation", per_channel=False, enabled=True)

    def forward(self, x):
        if not self.enabled or self.aq is None:
            return x
    	
        return self.aq(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 quantize=True, w_per_channel=True):
        super().__init__()
        Conv = QuantConvBn if quantize else ConvBn

        self.conv1 = Conv(inplanes, planes, k=3, s=stride, p=1, bias=False,
                          w_per_channel=w_per_channel, enable_wq=quantize)
        self.relu1 = nn.ReLU(inplace=True)
        self.actq1 = ActQuant(enabled=quantize)

        self.conv2 = Conv(planes, planes, k=3, s=1, p=1, bias=False,
                          w_per_channel=w_per_channel, enable_wq=quantize)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.actq2 = ActQuant(enabled=quantize)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.actq1(out)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)
        out = self.actq2(out)
        return out

class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10, quantize=True, w_per_channel=True, first_last_fp32=True):
        super().__init__()
        self.inplanes = 64
        self.stem = ConvBn(3, 64, k=3, s=1, p=1, bias=False)

        self.layer1 = self._make_layer(64, 2, stride=1,
                                       quantize=(not first_last_fp32) and quantize,
                                       w_per_channel=w_per_channel)
        self.layer2 = self._make_layer(128, 2, stride=2, quantize=quantize, w_per_channel=w_per_channel)
        self.layer3 = self._make_layer(256, 2, stride=2, quantize=quantize, w_per_channel=w_per_channel)
        self.layer4 = self._make_layer(512, 2, stride=2,
                                       quantize=(not first_last_fp32) and quantize,
                                       w_per_channel=w_per_channel)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_downsample(self, inplanes, planes, stride):
        if stride != 1 or inplanes != planes:
            return nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        return None

    def _make_layer(self, planes, blocks, stride, quantize, w_per_channel):
        downsample = self._make_downsample(self.inplanes, planes, stride)
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, quantize=quantize, w_per_channel=w_per_channel))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1, None, quantize=quantize, w_per_channel=w_per_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(x, inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def enable_quant(self, flag=True):
        for m in self.modules():
            if isinstance(m, ActQuant):
                m.enable(flag)
            if isinstance(m, QuantConvBn):
                m.wq.enable(flag)
