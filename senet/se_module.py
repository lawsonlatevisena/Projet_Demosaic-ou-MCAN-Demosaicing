'''from torch import nn
import torch
import torch.nn.functional as F

class MALayer(nn.Module):
    def __init__(self, channel, num_heads=4, reduction=16):
        super(MALayer, self).__init__()
        self.shuffledown = Shuffle_d(4)
        self.proj = nn.Linear(channel * 16, channel * 16 // reduction, bias=False)
        self.mhsa = nn.MultiheadAttention(embed_dim=channel * 16 // reduction, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(channel * 16 // reduction, channel * 16, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(4)
    
    def forward(self, x):
        ex_x = self.shuffledown(x)  # Réduction de résolution
        b, c, h, w = ex_x.size()
        ex_x_flat = ex_x.view(b, c, -1).permute(2, 0, 1)  # (h*w, batch, channels)
        
        # Projection linéaire
        proj_x = self.proj(ex_x_flat)
        
        # Self-Attention
        attn_output, _ = self.mhsa(proj_x, proj_x, proj_x)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
        
        # Génération de la carte d'attention
        y = self.fc(attn_output).view(b, c, 1, 1)
        ex_x = ex_x * y.expand_as(ex_x)
        x = self.shuffleup(ex_x)  # Remise en forme
        return x

class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)

'''

''' Version originale

from torch import nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MALayer, self).__init__()
        self.shuffledown = Shuffle_d(4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*16, channel*16 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*16 // reduction, channel*16, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(4)

    def forward(self, x):
        ex_x = self.shuffledown(x)
        b, c, _, _ = ex_x.size()
        y = self.avg_pool(ex_x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        ex_x = ex_x * y.expand_as(ex_x)
        x = self.shuffleup(ex_x)
        # buff_error = buff_x - x
        # buff_error = buff_x - x
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.mean(x,1)
        return torch.mean(x, 1).unsqueeze(1)

class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)
        
'''


import torch
from torch import nn

class Shuffle_d(nn.Module):
    def __init__(self, scale_factor=4):
        super(Shuffle_d, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h // self.scale_factor, self.scale_factor, w // self.scale_factor, self.scale_factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * (self.scale_factor ** 2), h // self.scale_factor, w // self.scale_factor)
        return x

class MALayer(nn.Module):
    def __init__(self, channel, num_heads=4, reduction=16):
        super(MALayer, self).__init__()
        self.shuffledown = Shuffle_d(4)

        # Projection linéaire
        self.proj = nn.Linear(channel * 16, channel * 16 // reduction, bias=False)

        # Attention multi-têtes
        self.mhsa = nn.MultiheadAttention(embed_dim=channel * 16 // reduction, num_heads=num_heads, batch_first=True)

        # Fully Connected pour la carte d'attention
        self.fc = nn.Sequential(
            nn.Linear(channel * 16 // reduction, channel * 16, bias=False),
            nn.Sigmoid()
        )

        # Restitution de la résolution
        self.shuffleup = nn.PixelShuffle(4)

    def forward(self, x):
        ex_x = self.shuffledown(x)  # Réduction de résolution
        b, c, h, w = ex_x.size()

        # Aplatir les dimensions spatiales
        ex_x_flat = ex_x.view(b, c, -1).permute(0, 2, 1)  # (batch_size, h*w, channels)

        # Projection linéaire
        proj_x = self.proj(ex_x_flat)  # (batch_size, h*w, embed_dim)

        # Self-Attention
        attn_output, _ = self.mhsa(proj_x, proj_x, proj_x)  # (batch_size, h*w, embed_dim)

        # Redimensionner attn_output
        attn_output = attn_output.permute(0, 2, 1).view(b, -1, h, w)  # (batch_size, embed_dim, h, w)

        # Génération de la carte d'attention
        y = self.fc(attn_output.mean(dim=[2, 3]))  # (batch_size, channels * 16)
        y = y.view(b, c, 1, 1)

        # Appliquer la carte d'attention
        ex_x = ex_x * y.expand_as(ex_x)

        # Restitution de la résolution
        x = self.shuffleup(ex_x)  # (batch_size, channels, h*4, w*4)
        return x
