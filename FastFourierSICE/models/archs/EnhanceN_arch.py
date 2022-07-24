"""
## ECCV 2022
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc,nc,3,1,1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x+self.block(x)



class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(nc,nc,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,1,1,0))

    def forward(self,x, mode):
        x = x+1e-8
        mag = torch.abs(x)
        pha = torch.angle(x)
        if mode == 'amplitude':
            mag = self.process(mag)
        elif mode == 'phase':
            pha = self.process(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out



class PhaseProcess(nn.Module):
    def __init__(self, in_nc,out_nc):
        super(PhaseProcess,self).__init__()
        self.cat = nn.Conv2d(2*in_nc,out_nc,1,1,0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.process = nn.Sequential(nn.Conv2d(in_nc * 2, in_nc // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(in_nc // 2, in_nc * 2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())

    def forward(self,x_amp,x):
        x_amp_freq = torch.fft.rfft2(x_amp, norm='backward')
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_amp_freq_amp = torch.abs(x_amp_freq)
        x_freq_pha = torch.angle(x_freq)
        real = x_amp_freq_amp * torch.cos(x_freq_pha)
        imag = x_amp_freq_amp * torch.sin(x_freq_pha)
        x_recom = torch.complex(real, imag)+1e-8
        x_recom = torch.fft.irfft2(x_recom)+1e-8
        x_recom = torch.abs(x_recom)+1e-8
        xcat = torch.cat([x_recom,x_amp], 1)
        xcat = self.process(self.contrast(xcat) + self.avgpool(xcat)) * xcat
        x_out = self.cat(xcat)


        return x_out


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class ProcessBlock(nn.Module):
    def __init__(self, in_nc):
        super(ProcessBlock,self).__init__()
        self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process1 = SpaBlock(in_nc)
        self.frequency_process1 = FreBlock(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_process2 = SpaBlock(in_nc)
        self.frequency_process2 = FreBlock(in_nc)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)


    def forward(self, x, mode = 'amplitude'):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        x = self.spatial_process1(x)
        x_freq = self.frequency_process1(x_freq,mode=mode)+1e-8
        x = x+self.frequency_spatial(torch.abs(torch.fft.irfft2(x_freq, s=(H, W), norm='backward'))+1e-8)
        x_freq = x_freq+torch.fft.rfft2(self.spatial_frequency(x), norm='backward')
        x = self.spatial_process2(x)+1e-8
        x_freq = self.frequency_process2(x_freq,mode=mode)+1e-8
        x_freq_spatial = torch.abs(torch.fft.irfft2(x_freq, s=(H, W), norm='backward'))
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori



class SFT(nn.Module):
    def __init__(self, nc):
        super(SFT,self).__init__()
        self.convmul = nn.Conv2d(nc,nc,3,1,1)
        self.convadd = nn.Conv2d(nc, nc, 3, 1, 1)
        self.convfuse = nn.Conv2d(2*nc, nc, 1, 1, 0)

    def forward(self, x, res):
        res = res.detach()
        mul = self.convmul(res)
        add = self.convadd(res)
        fuse = self.convfuse(torch.cat([x,mul*x+add],1))
        return fuse




class PhaseNet(nn.Module):
    def __init__(self, nc):
        super(PhaseNet,self).__init__()
        self.conv0 = PhaseProcess(3, nc)
        self.conv1 = ProcessBlock(nc)
        self.conv2 = ProcessBlock(nc)
        self.conv3 = ProcessBlock(nc)
        self.conv4 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc, 3, 1, 1, 0)
        self.trans = nn.Conv2d(3, nc, 1, 1, 0)
        self.combine1 = SFT(nc)
        self.combine2 = SFT(nc)

    def forward(self,x_amp,x):
        x_ori = x
        x_res = x_amp - x_ori
        x = self.conv0(x_amp,x)
        x_res = self.trans(x_res)
        x1 = self.conv1(x,mode = 'phase')
        x1 = torch.nan_to_num(x1, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x1 = self.combine1(x1, x_res)
        x2 = self.conv2(x1,mode = 'phase')
        x2 = torch.nan_to_num(x2, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x2 = self.combine2(x2, x_res)
        x3 = self.conv3(x2,mode = 'phase')
        x3 = torch.nan_to_num(x3, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x4 = self.conv4(x3,mode = 'phase')
        x4 = torch.nan_to_num(x4, nan=1e-5, posinf=1e-5, neginf=1e-5)
        xout = self.convout(x4)+x_ori

        return xout


class AmplitudeNet(nn.Module):
    def __init__(self, nc):
        super(AmplitudeNet,self).__init__()
        self.conv0 = nn.Conv2d(3,nc,1,1,0)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
        self.conv2 = ProcessBlock(nc*2)
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
        self.conv3 = ProcessBlock(nc*3)
        self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
        self.conv4 = ProcessBlock(nc*2)
        self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc,3,1,1,0)
        self.convoutfinal = nn.Conv2d(3, 3, 1, 1, 0)
        self.pro = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):
        x_ori = x
        # x_freq_ori = torch.fft.rfft2(x, norm='backward')
        # x_mag_ori = torch.abs(x_freq_ori)
        # x_pha_ori = torch.angle(x_freq_ori)
        x = self.conv0(x)
        x01 = self.conv1(x,mode = 'amplitude')
        x01 = torch.nan_to_num(x01, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1,mode = 'amplitude')
        x12 = torch.nan_to_num(x12, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2,mode = 'amplitude')
        x3 = torch.nan_to_num(x3, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x34 = self.up1(torch.cat([F.interpolate(x3,size=(x12.size()[2],x12.size()[3]),mode='bilinear'),x12],1))
        x4 = self.conv4(x34,mode = 'amplitude')
        x4 = torch.nan_to_num(x4, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x4 = self.up2(torch.cat([F.interpolate(x4,size=(x01.size()[2],x01.size()[3]),mode='bilinear'),x01],1))
        x5 = self.conv5(x4,mode = 'amplitude')
        x5 = torch.nan_to_num(x5, nan=1e-5, posinf=1e-5, neginf=1e-5)
        xout = self.convout(x5)
        # x_freq_out = torch.fft.rfft2(xout, norm='backward')
        # x_mag_out = torch.abs(x_freq_out)+self.pro(x_mag_ori)
        # real = x_mag_out * torch.cos(x_pha_ori)
        # imag = x_mag_out * torch.sin(x_pha_ori)
        # x_recom = torch.complex(real, imag)
        # x_recom = torch.fft.irfft2(x_recom)
        xout = x_ori + xout
        xfinal = self.convoutfinal(xout)

        return xfinal



class InteractNet(nn.Module):
    def __init__(self, nc):
        super(InteractNet,self).__init__()
        self.AmplitudeNet = AmplitudeNet(nc)
        self.PhaseNet = PhaseNet(nc*2)

    def forward(self, x):

        x_amp = self.AmplitudeNet(x)
        out = self.PhaseNet(x_amp,x)

        return torch.clamp(x_amp+0.00001,0.0,1.0) ,torch.clamp(x_amp+0.00001,0.0,1.0)