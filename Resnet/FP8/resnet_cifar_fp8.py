import torch
import torch.nn as nn
import torch.nn.functional as F
from fp8_quant import FP8Quantizer

class ConvBn(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, groups=1, bias=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.bn(self.conv(x))

class QuantConvBn_FP8(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, groups=1, bias=False,
                 w_fmt="e5m2", enable_wq=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        # per-output-channel scaling for weights
        self.wq = FP8Quantizer(mode="weight", fmt=w_fmt, per_channel=True, ch_axis=0, enabled=enable_wq)
    def forward(self, x):
        w = self.conv.weight
        qw = self.wq(w) if self.wq.enabled else w
        out = F.conv2d(x, qw, self.conv.bias, stride=self.conv.stride,
                       padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        out = self.bn(out)
        return out

class ActFP8(nn.Module):
    def __init__(self, enabled=True, a_fmt="e4m3"):
        super().__init__()
        self.enabled = enabled
        self.aq = FP8Quantizer(mode="activation", fmt=a_fmt, per_channel=False, enabled=enabled)
    def enable(self, flag=True):
        self.enabled = flag
        self.aq.enable(flag)
    def forward(self, x):
        if not self.enabled:
            return x
        return self.aq(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 quantize=True, w_fmt="e5m2", a_fmt="e4m3"):
        super().__init__()
        Conv = QuantConvBn_FP8 if quantize else ConvBn
        self.conv1 = Conv(inplanes, planes, k=3, s=stride, p=1, bias=False, w_fmt=w_fmt, enable_wq=quantize)
        self.relu1 = nn.ReLU(inplace=True)
        self.actq1 = ActFP8(enabled=quantize, a_fmt=a_fmt)
        self.conv2 = Conv(planes, planes, k=3, s=1, p=1, bias=False, w_fmt=w_fmt, enable_wq=quantize)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.actq2 = ActFP8(enabled=quantize, a_fmt=a_fmt)
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

class ResNet18_CIFAR_FP8(nn.Module):
    def __init__(self, num_classes=10, quantize=True, first_last_fp32=True,
                 w_fmt="e5m2", a_fmt="e4m3"):
        super().__init__()
        self.inplanes = 64
        self.w_fmt = w_fmt
        self.a_fmt = a_fmt
        self.stem = ConvBn(3, 64, k=3, s=1, p=1, bias=False)
        self.layer1 = self._make_layer(64, 2, stride=1, quantize=(not first_last_fp32) and quantize)
        self.layer2 = self._make_layer(128, 2, stride=2, quantize=quantize)
        self.layer3 = self._make_layer(256, 2, stride=2, quantize=quantize)
        self.layer4 = self._make_layer(512, 2, stride=2, quantize=(not first_last_fp32) and quantize)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_downsample(self, inplanes, planes, stride):
        if stride != 1 or inplanes != planes:
            return nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        return None
    def _make_layer(self, planes, blocks, stride, quantize):
        downsample = self._make_downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, stride, downsample,
                             quantize=quantize, w_fmt=self.w_fmt, a_fmt=self.a_fmt)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1, None,
                                     quantize=quantize, w_fmt=self.w_fmt, a_fmt=self.a_fmt))
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
            if isinstance(m, ActFP8):
                m.enable(flag)
            if isinstance(m, QuantConvBn_FP8):
                m.wq.enable(flag)

