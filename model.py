import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import os
from mmcv.cnn import build_norm_layer
from torch import einsum
from einops.layers.torch import Rearrange
from mmcv.runner import BaseModule
from typing import Sequence, Tuple, Union, List
import pywt


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _as_wavelet(wavelet):
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet
    

def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor

def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


def construct_2d_filt(lo, hi) -> torch.Tensor:
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    return filt

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b (g f) h w -> b g f h w', g=self.groups)
        x = rearrange(x, 'b g f h w -> b f g h w')
        x = rearrange(x, 'b f g h w -> b (f g) h w')
        return x

def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2
    if data_len % 2 != 0:
        padr += 1

    return padr, padl
    
def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:

    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad


class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]
    

class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None: 
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:   
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
              
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

         
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), ""
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), ""
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component


class MDFEM(nn.Module):
    def __init__(self, dim, wavelet='haar', initialize=True, use_ca=False, use_sa=False):
        super(MDFEM, self).__init__()
        self.dim = dim
        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=True)
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)

        self.conv1 = nn.Conv2d(dim*4, dim, 1)
        self.edge=MdDetailPerception(dim)  
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim*4, 1)
        self.use_sa = use_sa
        self.use_ca = use_ca
        if self.use_sa:
            self.sa_h = nn.Sequential(
                nn.PixelShuffle(2), 
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)  # c -> 1
            )
            self.sa_v = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)
            )
 
        if self.use_ca:
            self.ca_h = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True), 
            )
            self.ca_v = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True)
            )
            self.shuffle = ShuffleBlock(2)

        

    def forward(self, x):
        _, _, H, W = x.shape
        ya, (yh, yv, yd) = self.wavedec(x)
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.conv1(dec_x)
        x = self.edge(x)
        x = self.act(x)
        x = self.conv3(x)
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        y = self.waverec([ya, (yh, yv, yd)], None)
        if self.use_sa:
            sa_yh = self.sa_h(yh)
            sa_yv = self.sa_v(yv)
            y = y * (sa_yv + sa_yh)
        if self.use_ca:
            yh = torch.nn.functional.interpolate(yh, scale_factor=2, mode='area')
            yv = torch.nn.functional.interpolate(yv, scale_factor=2, mode='area')
            ca_yh = self.ca_h(yh)
            ca_yv = self.ca_v(yv)
            ca = self.shuffle(torch.cat([ca_yv, ca_yh], 1))
            ca_1, ca_2 = ca.chunk(2, dim=1)
            ca = ca_1 * ca_2  
            y = y * ca
        return y



class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias
    
class Conv2dc1_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2dc1_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 1).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 2] = conv_weight[:, :, 2] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias
    
class Conv2dc2_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2dc2_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 1 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 2] = conv_weight[:, :, 2] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff

class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias
    
class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_vd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class MdDetailPerception(nn.Module):
    def __init__(self, dim):
        super(MdDetailPerception, self).__init__() 

        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)


        self.conv2_1 = Conv2dc1_cd(dim, dim, (3,1),padding=(1,0), bias=True)
        self.conv_3x1 = nn.Conv2d(dim, dim, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv2_2 = Conv2dc2_cd(dim, dim, (1,3),padding=(0,1), bias=True)
        self.conv_1x3 = nn.Conv2d(dim, dim, kernel_size=(1,3), stride=1, padding=(0,1))
        self.conv_1x1 = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            Conv(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            Conv(dim // 4, dim, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

        self.conv_out=nn.Sequential(
            nn.Conv2d(dim, dim, padding=0, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):

        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias
        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res1 = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        w1, b1 = self.conv2_1.get_weight()
        w5, b5 = self.conv_3x1.weight, self.conv_3x1.bias
        w=w1+w5
        b=b1+b5
        res2 = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=(1,0), groups=1)

        w1, b1 = self.conv2_2.get_weight()
        w5, b5 = self.conv_1x3.weight, self.conv_1x3.bias
        w=w1+w5
        b=b1+b5
        res3 = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=(0,1), groups=1)
        res2=torch.concat((res2,res3),dim=1)
        res2=self.conv_1x1(res2)

        b,c,h,w=res1.shape
        avg_res1 = self.avg_pooling(res1).view(b, c, 1, 1)
        max_res1 = self.max_pooling(res1).view(b, c, 1, 1)
        v = self.fc_layers(avg_res1) + self.fc_layers(max_res1)
        v = self.sigmoid(v).view(b, c,1,1)
        res=(res1*v)+(res2*(1-v))
        
        res=self.conv_out(res)

        return res


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        return x

class MLCM(nn.Module):
    def __init__(self, channels):
        super(MLCM, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels, channels)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b, c, -1)  
        x21 = self.softmax(self.agp(x2).reshape(b, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b, c, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b, 1, h, w)
        results = weights.sigmoid()*group_x
        return results

class DAGAM(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()

        self.dim=dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.channel=ChannelAttentionModule(dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.linear_transform = nn.Linear(dim*2, dim)

    def forward(self, x):
        B, N, C = x.shape
        x1,x2=torch.split(x,N//2,dim=1)
        qkv1 = self.qkv(x1).reshape(B, N//2, 3, C).permute(2, 0, 1, 3)
        qkv2 = self.qkv(x2).reshape(B, N//2, 3, C).permute(2, 0, 1, 3)
        qkv1=qkv1.reshape(3,B,N//2,C//self.num_heads,self.num_heads).permute(0,1,4,2,3)
        qkv2=qkv2.reshape(3,B,N//2,C//self.num_heads,self.num_heads).permute(0,1,4,2,3)
        q1,k1,v1=qkv1[0],qkv1[1],qkv1[2]
        q2,k2,v2=qkv2[0],qkv2[1],qkv2[2]

        k1 = k1.transpose(1,2) 
        q1 = q1.transpose(1,2) 
        v1 = v1.transpose(1,2)
        k2 = k2.transpose(1,2)  
        v2 = v2.transpose(1,2)
        # 1
        attn_weights1 = torch.matmul(q1, k1.transpose(-1, -2))
        attn_weights1 = self.softmax(attn_weights1)
        attn1 = torch.matmul(attn_weights1, v1)
        # 2
        attn_weights2 = torch.matmul(q1, k2.transpose(-1, -2))
        attn_weights2 = self.softmax(attn_weights2)
        attn2 = torch.matmul(attn_weights2, v2)
        # difference
        attn = attn1-attn2


        attn = attn.permute(0,1,3,2).reshape(B,N//2,C).permute(0,2,1)
        attn1 = attn1.permute(0,1,3,2).reshape(B,N//2,C).permute(0,2,1)
        attn2 = attn2.permute(0,1,3,2).reshape(B,N//2,C).permute(0,2,1)
        attn=attn.unsqueeze(3)
        attn1=attn1.unsqueeze(3)
        attn2=attn2.unsqueeze(3)
        attn_channel = self.channel(attn)
        attn1_channel = self.channel(attn1)
        attn2_channel = self.channel(attn2)
        # 
        attn_channel=attn_channel.view(-1, attn_channel.size()[1], attn_channel.size()[2]*attn_channel.size()[3])
        attn1_channel=attn1_channel.view(-1, attn1_channel.size()[1], attn1_channel.size()[2]*attn1_channel.size()[3])
        attn2_channel=attn2_channel.view(-1, attn2_channel.size()[1], attn2_channel.size()[2]*attn2_channel.size()[3])
        #
        A2=torch.bmm(attn_channel, torch.transpose(attn1_channel,1,2).contiguous())
        A3=torch.bmm(attn_channel, torch.transpose(attn2_channel,1,2).contiguous())
        A = F.softmax(A2, dim = 1) 
        AA = F.softmax(A3, dim = 1) 
        # 
        attnview=attn.view(-1, attn.size()[1], attn.size()[2]*attn.size()[3])
        attnview1=attn1.view(-1, attn1.size()[1], attn1.size()[2]*attn1.size()[3])
        attnview2=attn2.view(-1, attn2.size()[1], attn2.size()[2]*attn2.size()[3])
        # 
        attn_att = torch.bmm(torch.transpose(A,1,2).contiguous(),attnview).contiguous()
        attn_att2 = torch.bmm(torch.transpose(AA,1,2).contiguous(),attnview).contiguous()
        # 
        attn_att=attn_att+attnview1
        attn_att2=attn_att2+attnview2

        attn = torch.concat((attn_att,attn_att2),dim=2)
        attn=attn.permute(0,2,1)

        # Output projection.
        x = self.proj(attn)
        x = self.proj_drop(x)

        return x

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.ws = window_size
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
        norm_layer=nn.LayerNorm
        self.norm1 = norm_layer(dim)
        qk_scale=None
        attn_drop=0.
        drop=0.0
        drop_path_rate=0.0
        self.cpe = ConvPosEnc(dim=dim, k=3)
        self.att = DAGAM(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,)
        self.drop_path_rate = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        act_layer=nn.GELU
        mlp_ratio=4.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.wavelet_block = MDFEM(dim, wavelet='haar', initialize=True) 

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.pad(x, self.ws)

        size=(H,W)
        x=x.reshape(B,C,H*W).permute(0,2,1)
        
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.att(cur, size)
        x = x + self.drop_path_rate(cur)

        cur = self.norm2(x)
        cur=cur.permute(0,2,1).reshape(B,C,H,W)
        x=x.permute(0,2,1).reshape(B,C,H,W)
        cur = self.mlp(cur)
        out = x + self.drop_path_rate(cur)

        out=self.wavelet_block(out)

        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.fusion_conv = MLCM(out_channels)

    def forward(self, x1, x2):
        res=x1
        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.down(x_fused)
        x_fused = self.fusion_conv(x_fused)
        x_fused=x_fused+res
        return x_fused
    

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        if dim_scale == 2:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            self.adjust_channels = nn.Conv2d(dim // 4, dim // dim_scale, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.pixel_shuffle = nn.Identity()
            self.adjust_channels = nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        if self.dim_scale == 2:
            x = self.pixel_shuffle(x)
            x = self.adjust_channels(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 decode_channels=96,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.b4 = Block(dim=encoder_channels[-2], num_heads=8, window_size=window_size)
        self.linear4=nn.Linear(encoder_channels[-1]*2,encoder_channels[-1])
        self.expand4=PatchExpand(encoder_channels[-1])

        self.b3 = Block(dim=encoder_channels[-2], num_heads=8, window_size=window_size)
        self.linear3=nn.Linear(encoder_channels[-2]*2,encoder_channels[-2])
        self.expand3=PatchExpand(encoder_channels[-2])

        self.b2 = Block(dim=encoder_channels[-3], num_heads=8, window_size=window_size)
        self.linear2=nn.Linear(encoder_channels[-3]*2,encoder_channels[-3])
        self.expand2=PatchExpand(encoder_channels[-3])


        self.b1 = Block(dim=encoder_channels[-4], num_heads=8, window_size=window_size)
        self.linear1=nn.Linear(encoder_channels[-4]*2,encoder_channels[-4])


        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

        self.AFFs = nn.ModuleList([            
            MSAA(decode_channels * 3, decode_channels),
            MSAA(decode_channels * 3, decode_channels * 2),
            MSAA(decode_channels * 6, decode_channels * 4)
        ])

        if self.training:
            self.conv4 = nn.Conv2d(decode_channels * 4, num_classes, 1, bias=False)
            self.conv3 = nn.Conv2d(decode_channels * 4, num_classes, 1, bias=False)
            self.conv2 = nn.Conv2d(decode_channels * 2, num_classes, 1, bias=False)

    def forward(self, res1, res2, res3, res4, h, w):
        x2_1 = F.interpolate(res2, scale_factor=2.0, mode="bilinear", align_corners=True)

        
        msaa1 = self.AFFs[0](res1, x2_1)
        msaa1F = F.interpolate(msaa1, scale_factor=0.5, mode="bilinear", align_corners=True)
       

        msaa2 = self.AFFs[1](res2, msaa1F)
        msaa2F = F.interpolate(msaa2, scale_factor=0.5, mode="bilinear", align_corners=True)
       
        msaa3 = self.AFFs[2](res3, msaa2F)

        msaa1 = torch.permute(msaa1, (0,2,3,1))
        msaa2 = torch.permute(msaa2, (0,2,3,1))
        msaa3 = torch.permute(msaa3, (0,2,3,1))


        # stage4
        x=res4
        x=x.permute(0,2,3,1)
        x = self.expand4(x)
        x=x.permute(0,3,1,2)
        x = self.b4(x)
        if self.training:
            h4=self.conv4(x)
        x=x.permute(0,2,3,1)
        

        # stage3
        x = torch.cat([x, msaa3],-1)
        x = self.linear3(x)
        x=x.permute(0,3,1,2)
        x = self.b3(x)
        if self.training:
            h3=self.conv3(x)
        x=x.permute(0,2,3,1)
        x = self.expand3(x)


        # stage2
        x = torch.cat([x, msaa2],-1)
        x = self.linear2(x)
        x=x.permute(0,3,1,2)
        x = self.b2(x)
        if self.training:
            h2=self.conv2(x)
        x=x.permute(0,2,3,1)
        x = self.expand2(x)
        

        # stage1
        x = torch.cat([x, msaa1],-1)
        x = self.linear1(x)
        x=x.permute(0,3,1,2)
        x=self.b1(x)
        

        # head
        if self.training:
            ah=[h2 ,h3 ,h4]
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            return x,ah
        
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Zhuizhong(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        
        pretrained_cfg = timm.create_model('convnext_tiny.in12k_ft_in1k_384').default_cfg
        pretrained_cfg['file']=r'/T2007061/yyc_worksapce/GeoSeg-main/ConvNext_timm_pretrain.bin'
        self.backbone=timm.models.convnext_tiny(features_only=True,pretrained=pretrained, output_stride=32,
                                          out_indices=(0, 1, 2, 3),pretrained_cfg=pretrained_cfg)

        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x,ah= self.decoder(res1, res2, res3, res4, h, w)
            return x,ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x