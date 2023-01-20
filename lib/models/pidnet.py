# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# modified by Ethan
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from lib.models.adf import ADFSoftmax

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


import time


class segmenthead(nn.Module):

    def __init__(self,
                 inplanes,
                 interplanes,
                 outplanes,
                 scale_factor=None,
                 bayes=False,
                 prob=False,
                 name=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes,
                               interplanes,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes,
                               outplanes,
                               kernel_size=1,
                               padding=0,
                               bias=True)
        self.scale_factor = scale_factor
        self.conv_var = nn.Conv2d(
            interplanes, outplanes, kernel_size=1, padding=0,
            bias=True) if bayes else nn.Identity()
        self.outplanes = outplanes
        self.name = name
        # if we are being bayesian and want to return a categorical probability, we should set the
        # output here to be an ADF Softmax operator
        self.prob = prob
        print('prob', prob)
        self.adf_softmax = ADFSoftmax() if prob else nn.Identity()

    def forward(self, x, return_var=False):

        x = self.conv1(self.relu(self.bn1(x)))
        feat_basis = self.relu(self.bn2(x))
        out = self.conv2(feat_basis)

        if return_var:
            var = self.conv_var(feat_basis**2.0)

            print(torch.sum(torch.isnan(var)).cpu().numpy())
            # print(self.conv_var.weight.data)
            # time.sleep(10)
            if self.prob:
                out, var = self.adf_softmax(out, var)
                print(torch.sum(torch.isnan(var)).cpu().numpy())
                print(var)
                time.sleep(10)
                # print('here')
            if self.scale_factor is not None:
                height = x.shape[-2] * self.scale_factor
                width = x.shape[-1] * self.scale_factor
                out = F.interpolate(out,
                                    size=[height, width],
                                    mode='bilinear',
                                    align_corners=algc)
                var = F.interpolate(var,
                                    size=[height, width],
                                    mode='bilinear',
                                    align_corners=algc)
            return out, var
        else:
            if self.scale_factor is not None:
                height = x.shape[-2] * self.scale_factor
                width = x.shape[-1] * self.scale_factor
                out = F.interpolate(out,
                                    size=[height, width],
                                    mode='bilinear',
                                    align_corners=algc)
            # otherwise just return the logit and a None value for the variance
            return out, None


class DAPPM(nn.Module):

    def __init__(self,
                 inplanes,
                 branch_planes,
                 outplanes,
                 BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes,
                      branch_planes,
                      kernel_size=3,
                      padding=1,
                      bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes,
                      branch_planes,
                      kernel_size=3,
                      padding=1,
                      bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes,
                      branch_planes,
                      kernel_size=3,
                      padding=1,
                      bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes,
                      branch_planes,
                      kernel_size=3,
                      padding=1,
                      bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((F.interpolate(self.scale1(x),
                                         size=[height, width],
                                         mode='bilinear',
                                         align_corners=algc) + x_list[0])))
        x_list.append((self.process2(
            (F.interpolate(self.scale2(x),
                           size=[height, width],
                           mode='bilinear',
                           align_corners=algc) + x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(self.scale3(x),
                                         size=[height, width],
                                         mode='bilinear',
                                         align_corners=algc) + x_list[2])))
        x_list.append(
            self.process4((F.interpolate(self.scale4(x),
                                         size=[height, width],
                                         mode='bilinear',
                                         align_corners=algc) + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class PAPPM(nn.Module):

    def __init__(self,
                 inplanes,
                 branch_planes,
                 outplanes,
                 BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale_process = nn.Sequential(
            BatchNorm(branch_planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4,
                      branch_planes * 4,
                      kernel_size=3,
                      padding=1,
                      groups=4,
                      bias=False),
        )

        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(
            F.interpolate(self.scale1(x),
                          size=[height, width],
                          mode='bilinear',
                          align_corners=algc) + x_)
        scale_list.append(
            F.interpolate(self.scale2(x),
                          size=[height, width],
                          mode='bilinear',
                          align_corners=algc) + x_)
        scale_list.append(
            F.interpolate(self.scale3(x),
                          size=[height, width],
                          mode='bilinear',
                          align_corners=algc) + x_)
        scale_list.append(
            F.interpolate(self.scale4(x),
                          size=[height, width],
                          mode='bilinear',
                          align_corners=algc) + x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out],
                                         1)) + self.shortcut(x)
        return out


class PagFM(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 after_relu=False,
                 with_channel=False,
                 BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels))
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels))
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1,
                          bias=False), BatchNorm(in_channels))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q,
                            size=[input_size[2], input_size[3]],
                            mode='bilinear',
                            align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y,
                          size=[input_size[2], input_size[3]],
                          mode='bilinear',
                          align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x


class Light_Bag(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels))
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class DDFMv2(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
            BatchNorm(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels))
        self.conv_i = nn.Sequential(
            BatchNorm(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class Bag(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


class PIDNet(nn.Module):

    def __init__(self,
                 num_classes,
                 m=2,
                 n=3,
                 planes=32,
                 ppm_planes=96,
                 head_planes=128,
                 augment=False,
                 aux_mode='train'):
        super(PIDNet, self).__init__()
        self.augment = augment
        self.aux_mode = aux_mode
        self.bayes = ('bayes' in aux_mode)
        print(aux_mode)
        self.apply_adf_softmax = (aux_mode == 'eval_bayes_prob')
        print(self.apply_adf_softmax)

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock,
                                       planes,
                                       planes * 2,
                                       m,
                                       stride=2)
        self.layer3 = self._make_layer(BasicBlock,
                                       planes * 2,
                                       planes * 4,
                                       n,
                                       stride=2)
        self.layer4 = self._make_layer(BasicBlock,
                                       planes * 4,
                                       planes * 8,
                                       n,
                                       stride=2)
        self.layer5 = self._make_layer(Bottleneck,
                                       planes * 8,
                                       planes * 8,
                                       2,
                                       stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2,
                                                    planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4,
                          planes,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8,
                          planes * 2,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2,
                                                    planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2,
                                                    planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4,
                          planes * 2,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8,
                          planes * 2,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4,
                                       head_planes,
                                       num_classes,
                                       scale_factor=8,
                                       name='final',
                                       bayes=self.bayes,
                                       prob=self.apply_adf_softmax)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_params(self):

        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def get_final_params(self):
        return [self.final_layer.conv2.weight, self.final_layer.conv2.bias]

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        return layer

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x),
                                  size=[height_output, width_output],
                                  mode='bilinear',
                                  align_corners=algc)
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x),
                                  size=[height_output, width_output],
                                  mode='bilinear',
                                  align_corners=algc)
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(self.spp(self.layer5(x)),
                          size=[height_output, width_output],
                          mode='bilinear',
                          align_corners=algc)

        x_, v_ = self.final_layer(self.dfm(x_, x, x_d), return_var=self.bayes)
        if 'eval_bayes' in self.aux_mode:
            return x_, v_

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


def get_seg_model(cfg, imgnet_pretrained):

    if 's' in cfg.MODEL.NAME:
        model = PIDNet(m=2,
                       n=3,
                       num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=32,
                       ppm_planes=96,
                       head_planes=128,
                       augment=True)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNet(m=2,
                       n=3,
                       num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=64,
                       ppm_planes=96,
                       head_planes=128,
                       augment=True)
    else:
        model = PIDNet(m=3,
                       n=4,
                       num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=64,
                       ppm_planes=112,
                       head_planes=256,
                       augment=True)

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED,
                                      map_location='cpu')['state_dict']
        model_dict = model.state_dict()
        pretrained_state = {
            k: v
            for k, v in pretrained_state.items()
            if (k in model_dict and v.shape == model_dict[k].shape)
        }
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict=False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {
            k[6:]: v
            for k, v in pretrained_dict.items()
            if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
        }
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    return model


def get_pred_model(name, num_classes):

    if 's' in name:
        model = PIDNet(m=2,
                       n=3,
                       num_classes=num_classes,
                       planes=32,
                       ppm_planes=96,
                       head_planes=128,
                       augment=False)
    elif 'm' in name:
        model = PIDNet(m=2,
                       n=3,
                       num_classes=num_classes,
                       planes=64,
                       ppm_planes=96,
                       head_planes=128,
                       augment=False)
    else:
        model = PIDNet(m=3,
                       n=4,
                       num_classes=num_classes,
                       planes=64,
                       ppm_planes=112,
                       head_planes=256,
                       augment=False)

    return model


if __name__ == '__main__':

    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name='pidnet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
