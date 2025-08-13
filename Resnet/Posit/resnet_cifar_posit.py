
import torch
import torch.nn as nn
import torch.nn.functional as F
from posit_quant import PositWeightQuantFn, PositActQuant

class ConvBn(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, groups=1, bias=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.bn(self.conv(x))

class QuantConvBn_Posit(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, groups=1, bias=False,
                 bits=8, es=1, enable_wq=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.wq = PositWeightQuantFn(bits=bits, es=es, enabled=True) if enable_wq else None
    def forward(self, x):
        w = self.conv.weight
        qw = self.wq(w) if (self.wq is not None and self.wq.enabled) else w
        out = F.conv2d(x, qw, self.conv.bias, stride=self.conv.stride,
                       padding=self.conv.padding, dilation=self.conv.dilation,
                       groups=self.conv.groups)
        out = self.bn(out)
        return out

class ActPosit(nn.Module):
    def __init__(self, enabled=True, bits=8, es=1):
        super().__init__()
        self.enabled = enabled
        self.bits = bits
        self.es = es
        self.aq = PositActQuant(bits=bits, es=es, enabled=True) if enabled else None
    def enable(self, flag=True):
        self.enabled = flag
        if flag and self.aq is None:
            self.aq = PositActQuant(bits=self.bits, es=self.es, enabled=True)
    def forward(self, x):
        if not self.enabled or self.aq is None:
            return x
        return self.aq(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 quantize=True, w_bits=8, w_es=0, a_bits=8, a_es=0):
        super().__init__()
        Conv = QuantConvBn_Posit if quantize else ConvBn
        self.conv1 = Conv(inplanes, planes, k=3, s=stride, p=1, bias=False,
                          bits=w_bits, es=w_es, enable_wq=quantize)
        self.relu1 = nn.ReLU(inplace=True)
        self.actq1 = ActPosit(enabled=quantize, bits=a_bits, es=a_es)
        self.conv2 = Conv(planes, planes, k=3, s=1, p=1, bias=False,
                          bits=w_bits, es=w_es, enable_wq=quantize)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.actq2 = ActPosit(enabled=quantize, bits=a_bits, es=a_es)
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

class ResNet18_CIFAR_POSIT(nn.Module):
    def __init__(self, num_classes=10, quantize=True, first_last_fp32=True,
                 w_bits=8, w_es=0, a_bits=8, a_es=0):
        super().__init__()
        self.inplanes = 64
        self.w_bits, self.w_es = w_bits, w_es
        self.a_bits, self.a_es = a_bits, a_es
        self.stem = ConvBn(3, 64, k=3, s=1, p=1, bias=False)
        self.layer1 = self._make_layer(64, 2, stride=1,
                                       quantize=(not first_last_fp32) and quantize)
        self.layer2 = self._make_layer(128, 2, stride=2, quantize=quantize)
        self.layer3 = self._make_layer(256, 2, stride=2, quantize=quantize)
        self.layer4 = self._make_layer(512, 2, stride=2,
                                       quantize=(not first_last_fp32) and quantize)
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
                             quantize=quantize,
                             w_bits=self.w_bits, w_es=self.w_es,
                             a_bits=self.a_bits, a_es=self.a_es)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1, None,
                                     quantize=quantize,
                                     w_bits=self.w_bits, w_es=self.w_es,
                                     a_bits=self.a_bits, a_es=self.a_es))
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
            if isinstance(m, ActPosit):
                m.enable(flag)
            if isinstance(m, QuantConvBn_Posit):
                m.wq.enable(flag)
