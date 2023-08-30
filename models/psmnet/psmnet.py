"""
From https://github.com/JiaRenChang/PSMNet
Licensed under MIT
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = convbn_3d(
            inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1
        )

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes * 2),
        )  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes),
        )  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
        )

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, im2, im3, hints=None, validhints=None):

        im2_fea = self.feature_extraction(im2)
        im3_fea = self.feature_extraction(im3)
        batch_size, _, height, width = im2_fea.shape

        #matching 2-3
        cost_23 = Variable(torch.FloatTensor(im2_fea.size()[0], im2_fea.size()[1]*2, self.maxdisp//4,  im2_fea.size()[2],  im2_fea.size()[3]).zero_())
        cost_23 = cost_23.cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost_23[:, :im2_fea.size()[1], i, :,i:] = im2_fea[:,:,:,i:]
             cost_23[:, im2_fea.size()[1]:2*im2_fea.size()[1], i, :,i:] = im3_fea[:,:,:,:-i]
            else:
             cost_23[:, :im2_fea.size()[1], i, :,:] = im2_fea
             cost_23[:, im2_fea.size()[1]:2*im2_fea.size()[1], i, :,:] = im3_fea

        cost_23 = cost_23.contiguous()
        cost = cost_23

        if hints is not None:
            ### MODULATION ###
            GAUSSIAN_HEIGHT = 10.
            GAUSSIAN_WIDTH = 1.
            features = im2_fea.size()[1] * 2
            num_disp = self.maxdisp // 4

            # image features are one fourth the original size: subsample the hints and divide them by four
            SUBSAMPLE = 4
            hints = F.upsample(hints, [hints.size()[2]//SUBSAMPLE, hints.size()[3]//SUBSAMPLE], mode='nearest').squeeze(1)
            #validhints = validhints.unsqueeze(1)
            
            validhints = F.upsample(validhints, [validhints.size()[2]//SUBSAMPLE, validhints.size()[3]//SUBSAMPLE], mode='nearest').squeeze(1)
            
            hints = hints*validhints / float(SUBSAMPLE)
            GAUSSIAN_WIDTH /= float(SUBSAMPLE)

            # add feature and disparity dimensions to hints and validhints
            # and repeat their values along those dimensions, to obtain the same size as cost
            hints = hints.unsqueeze(1).unsqueeze(2).expand(-1, features, num_disp, -1, -1)
            validhints = validhints.unsqueeze(1).unsqueeze(2).expand(-1, features, num_disp, -1, -1)

            # create a tensor of the same size as cost, with disparities
            # between 0 and num_disp-1 along the disparity dimension
            disparities = torch.linspace(start=0, end=num_disp - 1, steps=num_disp).cuda().unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(batch_size, features, -1, height, width)
            cost = cost * ((1 - validhints) + validhints * GAUSSIAN_HEIGHT * torch.exp(-(disparities - hints) ** 2 / (2 * GAUSSIAN_WIDTH ** 2)))
        
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,im2.size()[2],im2.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,im2.size()[2],im2.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,im2.size()[2],im2.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1.unsqueeze(1), pred2.unsqueeze(1), pred3.unsqueeze(1)
        else:
            return pred3.unsqueeze(1)
