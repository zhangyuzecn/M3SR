"""
M3SR: Multi-scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction

Author: Yuze Zhang
"""
import os
import math
import warnings
from functools import partial
from typing import Callable


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import _calculate_fan_in_and_fan_out
from thop import profile
from timm.models.layers import DropPath, trunc_normal_
from pytorch_wavelets import DWTForward, DWTInverse
from mamba_ssm import Mamba


try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except ImportError:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except ImportError:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except ImportError:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,  
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            disable_z=True,
            # ======================
            with_initial_state=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_state = d_state
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        assert d_inner % dt_rank == 0
        self.with_dconv = d_conv > 1
        Linear = nn.Linear

        self.disable_z = disable_z
        self.out_norm = nn.LayerNorm(d_inner)
        k_group = 4
        self.forward_core = partial(self.forward_core, force_fp32=False, dstate=d_state)
        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        self.conv2d = nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            ),
            Permute(0, 2, 3, 1),
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
        self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)),
                                              requires_grad=False)

    def forward_core(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False, 
            chunk_size=64,
            dstate=16,
            # ==============================
            selective_scan_backend=None,
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode,
                           force_torch=scan_force_torch)  # (B, H, W, 4, D)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
            backend=selective_scan_backend,
        )
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False,
                                         scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias,
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W,
            ))
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs): # B,H,W,C
        x = self.in_proj(x) # B,H,W,C*2
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            z = self.act(z)

        x = self.conv2d(x)  # B,H,W,C*2
        x = self.act(x)
        y = self.forward_core(x)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class FreqMamba(nn.Module):
    def __init__(self,
                 hidden_dim: int = 0,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 d_state: int = 8,
                 ssm_ratio: float = 1.,
                 **kwargs,
                 ):
        super(FreqMamba, self).__init__()
        self.ln = norm_layer(hidden_dim*4)
        self.drop_path = DropPath(drop_path)
        self.att_spa = SS2D(d_model=hidden_dim*4, d_state=d_state, ssm_ratio=ssm_ratio, conv_bias=False,
                            dropout=attn_drop_rate)

    def forward(self, input):
        """
            input: [B, C, H, W]
            return out: [B, C, H, W]
        """
        B, C, H, W = input.shape
        xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()
        ifm = DWTInverse(mode='zero', wave='haar').cuda()
        Yl, Yh = xfm(input)  # Yl: (B, C, H/2, W/2), Yh: list [(B, C, 3, H/2, W/2)]
        Yh = Yh[0] 
        Yh = Yh.permute(0, 2, 1, 3, 4).reshape(B, 3 * C, H // 2, W // 2)  #  (B, 3*C, H/2, W/2)
        x = torch.cat([Yl, Yh], dim=1)  # (B, 4*C, H/2, W/2)
        x = x.permute(0, 2, 3, 1)  # (B, H/2, W/2, 4*C)
        B, H, W, C = x.shape
        y = x + self.drop_path(self.att_spa(self.ln(x)))
        y = y.view(B, H, W, C).contiguous().permute(0, 3, 1, 2)  #  (B, 4*C, H/2, W/2)

        Yl = y[:, :C // 4, :, :]  #  (B, C, H/2, W/2)
        Yh = y[:, C // 4:, :, :].reshape(B, 3, C // 4, H, W).permute(0, 2, 1, 3, 4)  #  (B, C, 3, H/2, W/2)

        out = ifm((Yl, [Yh]))  # (B, C, H, W)
        return out



class SSFM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            ssm_ratio: float = 2.,
            d_group=4,
            use_mlp=True,
            **kwargs,
    ):
        super().__init__()
        self.use_mlp=use_mlp
        self.conv_in = nn.Conv2d(hidden_dim, hidden_dim * d_group, kernel_size=1)
        self.spe_mamba = SpeMamba(channels=hidden_dim*d_group, token_num=d_group, use_residual=True, group_num=d_group)
        self.conv_out = nn.Conv2d(hidden_dim * d_group, hidden_dim, kernel_size=1)
        self.fre_mamba = FreqMamba(hidden_dim=hidden_dim, d_state=d_state, ssm_ratio=ssm_ratio, conv_bias=False,
                             dropout=attn_drop_rate)
        self.ln = norm_layer(hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.att_spa = SS2D(d_model=hidden_dim, d_state=d_state, ssm_ratio=ssm_ratio, conv_bias=False,
                             dropout=attn_drop_rate, **kwargs)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=2*hidden_dim, out_features=hidden_dim)
        self.weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)


    def forward(self, input):
        # Spe
        spe_x = self.conv_out(self.spe_mamba(self.conv_in(input)))
        # Fre
        fre_x = self.fre_mamba(input)
        # Spa
        x = input.permute(0, 2, 3, 1)
        B,H,W,C = x.shape
        spa_x = x + self.drop_path(self.att_spa(self.ln(x)))
        if self.use_mlp:
            spa_x = spa_x + self.drop_path(self.mlp(self.ln(spa_x)))  
        spa_x = spa_x.view(B, H, W, C).contiguous().permute(0, 3, 1, 2)
        # Fusion
        weights = self.softmax(self.weights)
        fusion_x = spa_x * weights[0] + spe_x * weights[1] + fre_x * weights[2] +input
        return fusion_x


class SSFMB(nn.Module):
    def __init__(
            self,
            dim,
            d_group,
            num_blocks
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                SSFM(
                    hidden_dim=dim,
                    drop_path=0,  
                    norm_layer=nn.LayerNorm,
                    mlp_ratio=1.,  
                    d_state=8,
                    d_group=d_group,
                    use_mlp=False
                )
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for block in self.blocks:
            x = block(x) + x 
        return x


class MSM(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, d_group=4, num_blocks=[1, 1, 1]):
        super(MSM, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                SSFMB(dim=dim_stage,d_group=d_group,num_blocks=num_blocks[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SSFMB(dim=dim_stage,d_group=d_group,num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                SSFMB(dim=dim_stage // 2,d_group=d_group,num_blocks=num_blocks[stage - 1 - i]),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (SSFM, FeaDownSample) in self.encoder_layers:
            fea = SSFM(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class M3SR(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=1, d_group=4,num_blocks=[1,1,1]):
        super(M3SR, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        modules_body = [MSM(dim=31, stage=2, d_group=d_group,num_blocks=num_blocks) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        y= self.body(x)
        y = self.conv_out(y)
        y += x
        return y[:, :, :h_inp, :w_inp]


if __name__ == "__main__":
    gpus = '5'
    print("=> use gpu id: '{}'".format(gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    input_tensor = torch.rand(1, 3, 482, 512).cuda()
    model = M3SR(in_channels=3, out_channels=31).cuda()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    macs, params = profile(model, inputs=(input_tensor,))
    flops = macs * 2  
    flops_g = flops / 1e9 
    params_m = params / 1e6  
    print(f"Parameters: {params_m:.3f} M; FLOPs: {flops_g:.3f} G")
    print(torch.__version__)











