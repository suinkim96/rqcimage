import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pdb import set_trace as stx
from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1


class QRestormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        q_lower=0.0,
        q_upper=1.0,
        q_sr_ratio = 4
    ):

        super(QRestormer, self).__init__()
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.q_sr_ratio = q_sr_ratio

        self.patch_embed = OverlapPatchEmbed(inp_channels + 1, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        # 각 픽셀마다 U(q_lower, q_upper)에서 난수 추출 (shape: [B, 1, H, W])
        low_H = inp_img.size(2) // self.q_sr_ratio
        low_W = inp_img.size(3) // self.q_sr_ratio
        if self.training:
            # 각 픽셀마다 Uniform(0,1)에서 난수 추출 (저해상도)
            low_quantile = torch.rand((inp_img.size(0), 1, low_H, low_W), device=inp_img.device)
            # Scale quantile if needed (if self.q_lower != 0 or self.q_upper != 1)
            low_quantile = low_quantile * (self.q_upper - self.q_lower) + self.q_lower
        else:
            # 평가 시, 전체를 0.5로 채운 텐서를 생성 (즉, median)
            low_quantile = torch.full((inp_img.size(0), 1, low_H, low_W), 0.5, device=inp_img.device)

        # Upsample the low-res map to the original image resolution with bilinear interpolation
        quantile = F.interpolate(low_quantile, size=(inp_img.size(2), inp_img.size(3)), mode='bilinear', align_corners=False)

        # Concatenate the quantile map with the input image along the channel dimension
        aug_img = torch.cat([inp_img, quantile], dim=1)

        inp_enc_level1 = self.patch_embed(aug_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1, quantile


class QRestormerModified(QRestormer):
    def forward(self, inp_img):
        # q_sr_ratio에 따른 low resolution 크기 계산 (quantile map 생성 시에만 사용)
        low_H = inp_img.size(2) // self.q_sr_ratio
        low_W = inp_img.size(3) // self.q_sr_ratio

        if self.training:
            # 학습 모드: 기존 방식으로 난수를 생성 후 업샘플링
            low_quantile = torch.rand((inp_img.size(0), 1, low_H, low_W), device=inp_img.device)
            low_quantile = low_quantile * (self.q_upper - self.q_lower) + self.q_lower
            quantile = F.interpolate(low_quantile, size=(inp_img.size(2), inp_img.size(3)),
                                    mode='bilinear', align_corners=False)
            aug_img = torch.cat([inp_img, quantile], dim=1)

            # 일반적인 forward pass (레이어는 그대로 사용)
            inp_enc_level1 = self.patch_embed(aug_img)
            out_enc_level1 = self.encoder_level1(inp_enc_level1)
            inp_enc_level2 = self.down1_2(out_enc_level1)
            out_enc_level2 = self.encoder_level2(inp_enc_level2)
            inp_enc_level3 = self.down2_3(out_enc_level2)
            out_enc_level3 = self.encoder_level3(inp_enc_level3) 
            inp_enc_level4 = self.down3_4(out_enc_level3)        
            latent = self.latent(inp_enc_level4) 
                            
            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3) 

            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2) 

            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1)
            out_dec_level1 = self.refinement(out_dec_level1)

            if self.dual_pixel_task:
                out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
                out_dec_level1 = self.output(out_dec_level1)
            else:
                out_dec_level1 = self.output(out_dec_level1) + inp_img

            return out_dec_level1, quantile

        else:
            # 평가 모드: 먼저 0.1과 0.9 분위수로 각각 예측하여 diff(불확실성)를 구합니다.
            def quantile_forward(q):
                # low resolution에서 q 값으로 채운 텐서를 생성하고 바로 full resolution으로 업샘플링
                low_quantile = torch.full((inp_img.size(0), 1, low_H, low_W), q, device=inp_img.device)
                quantile = F.interpolate(low_quantile, size=(inp_img.size(2), inp_img.size(3)),
                                        mode='bilinear', align_corners=False)
                aug_img = torch.cat([inp_img, quantile], dim=1)

                inp_enc_level1 = self.patch_embed(aug_img)
                out_enc_level1 = self.encoder_level1(inp_enc_level1)
                inp_enc_level2 = self.down1_2(out_enc_level1)
                out_enc_level2 = self.encoder_level2(inp_enc_level2)
                inp_enc_level3 = self.down2_3(out_enc_level2)
                out_enc_level3 = self.encoder_level3(inp_enc_level3) 
                inp_enc_level4 = self.down3_4(out_enc_level3)        
                latent = self.latent(inp_enc_level4) 
                                
                inp_dec_level3 = self.up4_3(latent)
                inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
                inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
                out_dec_level3 = self.decoder_level3(inp_dec_level3) 

                inp_dec_level2 = self.up3_2(out_dec_level3)
                inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
                inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
                out_dec_level2 = self.decoder_level2(inp_dec_level2) 

                inp_dec_level1 = self.up2_1(out_dec_level2)
                inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
                out_dec_level1 = self.decoder_level1(inp_dec_level1)
                out_dec_level1 = self.refinement(out_dec_level1)

                if self.dual_pixel_task:
                    out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
                    out_dec_level1 = self.output(out_dec_level1)
                else:
                    out_dec_level1 = self.output(out_dec_level1) + inp_img

                return out_dec_level1

            # 0.1, 0.9 분위수에 대해 각각 예측 수행
            pred_low = quantile_forward(0.1)
            pred_high = quantile_forward(0.9)

            # 두 예측치의 차이를 구하여 불확실성(diff) 계산
            diff = torch.abs(pred_high - pred_low)
            max_diff = diff.max()
            # 0으로 나누는 상황을 피하기 위해, max_diff가 0이면 모두 0으로 설정
            alpha = diff / max_diff if max_diff > 0 else torch.zeros_like(diff)
            alpha = torch.clamp(alpha, 0, 1)
            # diff가 작으면 0.7, 크면 0.5에 가깝도록 선형 보간 (alpha=0 -> 0.7, alpha=1 -> 0.5)
            final_quantile_map = 0.7 - 0.2 * alpha

            # 최종 quantile map을 그대로 사용하여 마지막 forward pass 수행
            aug_img = torch.cat([inp_img, final_quantile_map], dim=1)
            inp_enc_level1 = self.patch_embed(aug_img)
            out_enc_level1 = self.encoder_level1(inp_enc_level1)
            inp_enc_level2 = self.down1_2(out_enc_level1)
            out_enc_level2 = self.encoder_level2(inp_enc_level2)
            inp_enc_level3 = self.down2_3(out_enc_level2)
            out_enc_level3 = self.encoder_level3(inp_enc_level3)
            inp_enc_level4 = self.down3_4(out_enc_level3)
            latent = self.latent(inp_enc_level4)
                            
            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3)
            
            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2)
            
            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1)
            out_dec_level1 = self.refinement(out_dec_level1)
            
            if self.dual_pixel_task:
                out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
                out_dec_level1 = self.output(out_dec_level1)
            else:
                out_dec_level1 = self.output(out_dec_level1) + inp_img

            return out_dec_level1, final_quantile_map


##########################################################################


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR_sr(nn.Module):
    def __init__(self,
                 n_resblock=16,
                 n_feats=64,
                 n_colors=3,
                 scale=2,
                 res_scale=1.0,
                 conv=None):
        super(EDSR_sr, self).__init__()
        
        # conv 인자가 None이면 default_conv 사용 (미리 정의되어 있어야 함)
        if conv is None:
            conv = default_conv
        
        kernel_size = 3
        act = nn.ReLU(True)
        self.up_factor = scale

        # head module: 입력 채널 -> n_feats
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # body module: 여러 ResBlock과 마지막 conv layer
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # tail module: Upsampler 후 원래 채널 수 복원
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
        # 추가 var_conv 모듈
        self.var_conv = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ELU(),
            conv(n_feats, n_feats, kernel_size),
            nn.ELU(),
            conv(n_feats, n_colors, kernel_size),
            nn.ELU()
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        out = self.tail(res)
        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
                

class EDSR_blur(nn.Module):
    def __init__(self,
                 n_resblock=16,
                 n_feats=64,
                 n_colors=3,
                 scale=2,  # scale 인자는 그대로 남겨두지만 사용하지 않음
                 res_scale=1.0,
                 conv=None):
        super(EDSR_blur, self).__init__()
        
        # conv 인자가 None이면 default_conv 사용 (미리 정의되어 있어야 함)
        if conv is None:
            conv = default_conv
        
        kernel_size = 3
        act = nn.ReLU(True)

        # head module: 입력 채널 -> n_feats
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # body module: 여러 ResBlock과 마지막 conv layer
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # tail module: Upsampler 대신 단순히 채널 복원을 위한 conv 레이어만 사용
        m_tail = [conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
        # var_conv 모듈은 필요에 따라 제거하거나 수정할 수 있습니다.
        # 여기서는 사용하지 않으므로 주석 처리합니다.
        # self.var_conv = nn.Sequential(
        #     conv(n_feats, n_feats, kernel_size),
        #     nn.ELU(),
        #     conv(n_feats, n_feats, kernel_size),
        #     nn.ELU(),
        #     conv(n_feats, n_colors, kernel_size),
        #     nn.ELU()
        # )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        out = self.tail(res)
        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
                


class QEDSR_blur(nn.Module):
    def __init__(self,
                 n_resblock=16,
                 n_feats=64,
                 n_colors=3,
                 res_scale=1.0,
                 q_lower=0.0,
                 q_upper=1.0,
                 conv=None):
        super(QEDSR_blur, self).__init__()
        
        # conv 인자가 None이면 default_conv 사용 (미리 정의되어 있어야 함)
        if conv is None:
            conv = default_conv
        
        kernel_size = 3
        act = nn.ReLU(True)
        
        # q_lower, q_upper를 인스턴스 변수로 저장
        self.q_lower = q_lower
        self.q_upper = q_upper

        # Head: 입력 채널은 원래 이미지 채널(n_colors) + 1 (quantile 채널)
        m_head = [conv(n_colors + 1, n_feats, kernel_size)]
        
        # Body: 여러 ResBlock과 마지막 conv layer
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        
        # Tail: 단순히 n_feats를 n_colors로 복원 (출력은 원래 이미지 채널 수)
        m_tail = [conv(n_feats, n_colors, kernel_size)]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
    def forward(self, x):
        """
        x: 입력 이미지 텐서, shape: [B, n_colors, H, W]
        """
        # 각 픽셀마다 U(q_lower, q_upper)에서 난수 추출 (shape: [B, 1, H, W])
        quantile = torch.rand((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
        quantile = quantile * (self.q_upper - self.q_lower) + self.q_lower

        # 원본 이미지와 quantile 채널을 concat하여 입력 채널 확장: [B, n_colors+1, H, W]
        x_aug = torch.cat([x, quantile], dim=1)
        
        # 네트워크 통과
        x_head = self.head(x_aug)
        res = self.body(x_head)
        res = res + x_head  # residual 연결
        out = self.tail(res)
        
        # 최종 출력은 복원된 이미지와 생성한 quantile 채널
        return out, quantile

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
