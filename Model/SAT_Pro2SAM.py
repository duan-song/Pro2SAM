import torch
import torch.nn as nn
from functools import partial
from utils.cfgs import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from utils.accuracy import *
from utils.func import *
from skimage import measure
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


__all__ = [
    'deit_sat_tiny_patch16_224', 'deit_sat_small_patch16_224', 'deit_sat_base_patch16_224',
]

def get_kernel(kernlen=3, nsig=6):    
    interval = (2*nsig+1.)/kernlen  
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)                                 
    kern1d = np.diff(st.norm.cdf(x))    
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))   
    kernel = kernel_raw/kernel_raw.sum()          
    return kernel

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim   
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_merging = PatchMerging(
                input_resolution=(32, 32), embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.loc_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.glo_embed = nn.Parameter(torch.zeros(1,  4, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.glo_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.loc_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'loc_embed', 'loc_token'} 

class SAT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1) 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)

        kernel = get_kernel(kernlen=3,nsig=6)  
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
    def forward_features(self, x):
        B = x.shape[0]

        global_token = self.patch_merging(x)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        # loc_tokens = self.loc_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, global_token), dim=1)
        pos_embed = torch.cat([self.pos_embed, self.glo_embed], 1)
        x = x + pos_embed
        x = self.pos_drop(x) 
        mask_all = []
        mask0_all = []
        mask1_all = []
        mask2_all = []
        mask3_all = []
        for cur_depth, blk in enumerate(self.blocks):
            x,  mask, mask0, mask1, mask2, mask3 = blk(x, cur_depth)
            mask_all.append(mask)
            mask0_all.append(mask0)
            mask1_all.append(mask1)
            mask2_all.append(mask2)
            mask3_all.append(mask3)

        x = self.norm(x)
        return x[:, 0], x[:, -1], x[:, 1:-4],  mask_all, mask0_all, mask1_all, mask2_all, mask3_all

    def forward(self, x, label=None, phase='train'):
        batch = x.size(0)
        x_cls, x_loc, x_patch,  mask_all, mask0_all, mask1_all, mask2_all, mask3_all = self.forward_features(x)
        n, p, c = x_patch.shape


        mask_all = torch.stack(mask_all)
        # print("mask_all.shape = {}".format(mask_all.shape))
        mask_all = mask_all[-3:,:,:,:,1:-4]
        mask_all = torch.mean(mask_all, dim=2) 
        mask_all = torch.mean(mask_all, dim=0)
        # print("mask_all.shape = {}".format(mask_all.shape))
        mask_all = mask_all.reshape(batch,1,14,14)

        # obtain mask0 of global token 0
        mask0_all = torch.stack(mask0_all)
        mask0 = mask0_all[-3:,:,:,:,1:-4]
        mask0 = torch.mean(mask0, dim=2)
        mask0 = torch.mean(mask0, dim=0)
        mask0 = mask0.reshape(batch, 1, 14, 14)
        # computing the ba and norm loss of mask0
        mask0_avg = mask0.clone()
        mask0_avg = F.conv2d(mask0_avg, self.weight, padding=1)
        mask0_ba = mask0.view(batch, -1).mean(-1)
        mask0_norm = ((1 - mask0_avg) * mask0_avg).view(batch, -1).mean(-1)

        # obtain mask1 of global token 1
        mask1_all = torch.stack(mask1_all)
        mask1 = mask1_all[-3:, :, :, :, 1:-4]
        mask1 = torch.mean(mask1, dim=2)
        mask1 = torch.mean(mask1, dim=0)
        mask1 = mask1.reshape(batch, 1, 14, 14)
        # computing the ba and norm loss of mask1
        mask1_avg = mask1.clone()
        mask1_avg = F.conv2d(mask1_avg, self.weight, padding=1)
        mask1_ba = mask1.view(batch, -1).mean(-1)
        mask1_norm = ((1 - mask1_avg) * mask1_avg).view(batch, -1).mean(-1)

        # obtain mask2 of global token 2
        mask2_all = torch.stack(mask2_all)
        mask2 = mask2_all[-3:, :, :, :, 1:-4]
        mask2 = torch.mean(mask2, dim=2)
        mask2 = torch.mean(mask2, dim=0)
        mask2 = mask2.reshape(batch, 1, 14, 14)
        # computing the ba and norm loss of mask2
        mask2_avg = mask2.clone()
        mask2_avg = F.conv2d(mask2_avg, self.weight, padding=1)
        mask2_ba = mask2.view(batch, -1).mean(-1)
        mask2_norm = ((1 - mask2_avg) * mask2_avg).view(batch, -1).mean(-1)

        # obtain mask3 of global token 3
        mask3_all = torch.stack(mask3_all)
        mask3 = mask3_all[-3:, :, :, :, 1:-4]
        mask3 = torch.mean(mask3, dim=2)
        mask3 = torch.mean(mask3, dim=0)
        mask3 = mask3.reshape(batch, 1, 14, 14)
        # computing the ba and norm loss of mask3
        mask3_avg = mask3.clone()
        mask3_avg = F.conv2d(mask3_avg, self.weight, padding=1)
        mask3_ba = mask3.view(batch, -1).mean(-1)
        mask3_norm = ((1-mask3_avg)*mask3_avg).view(batch, -1).mean(-1)

        x_patch = torch.reshape(x_patch , [n, int(p**0.5), int(p**0.5), c])  
        x_patch = x_patch.permute([0, 3, 1, 2])   
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)

        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2) 
        mask_avg = mask_all.clone() 
        mask_avg = F.conv2d(mask_avg, self.weight, padding=1)
        mask_ba = mask_all.view(batch,-1).mean(-1)
        mask_norm = ((1-mask_avg)*mask_avg).view(batch,-1).mean(-1)

        if phase == 'train':
            return x_logits, mask_ba, mask_norm, mask0_ba, mask0_norm, mask1_ba, mask1_norm, mask2_ba, mask2_norm, mask3_ba, mask3_norm
        else: 
            n, c, h, w = x_patch.shape 
            
            mask_all = mask_all.reshape([n, h, w])
            return x_logits, mask_all


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cur_depth=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # mask = nn.Sigmoid()(attn[:,:,-1,:].unsqueeze(2).mean(1).unsqueeze(1))
        # print("attn.shape = {}".format(attn.shape))
        # print("mask.shape = {}".format(mask.shape))
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        mask0 = nn.Sigmoid()(attn[:, :, -1, :].unsqueeze(2).mean(1).unsqueeze(1))   # B 1 1 201
        mask1 = nn.Sigmoid()(attn[:, :, -2, :].unsqueeze(2).mean(1).unsqueeze(1))   # B 1 1 201
        mask2 = nn.Sigmoid()(attn[:, :, -3, :].unsqueeze(2).mean(1).unsqueeze(1))   # B 1 1 201
        mask3 = nn.Sigmoid()(attn[:, :, -4, :].unsqueeze(2).mean(1).unsqueeze(1))   # B 1 1 201
        mask = torch.cat([mask0, mask1, mask2, mask3], dim=-2).mean(-2).unsqueeze(-2)
        # print("attn.shape = {}".format(attn.shape))
        # print("mask.shape = {}".format(mask.shape))
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")




        attn = attn.softmax(dim=-1)  
        if cur_depth >= 9 :    
            attn = attn * mask
         
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask, mask0, mask1, mask2, mask3


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, vis=vis)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, cur_depth=None):
        o,  mask, mask0, mask1, mask2, mask3 = self.attn(self.norm1(x), cur_depth=cur_depth)
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,  mask, mask0, mask1, mask2, mask3

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution=(32,32), embed_dim=768):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = embed_dim
        # self.reduction = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)
        # self.norm = norm_layer(4 * embed_dim)

        self.proj0 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.proj1 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.proj2 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.proj3 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution

        x = F.interpolate(x, size=(H, W), mode='bilinear')

        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2

        t0 = self.proj0(x0).squeeze(-1).transpose(1, 2)     # B 1 768
        t1 = self.proj1(x1).squeeze(-1).transpose(1, 2)     # B 1 768
        t2 = self.proj1(x2).squeeze(-1).transpose(1, 2)     # B 1 768
        t3 = self.proj1(x3).squeeze(-1).transpose(1, 2)     # B 1 768

        t = torch.cat([t0, t1, t2, t3], dim=1)  # B 4 768
        # print(t.shape)
        # x = x.view(B, H, W, C)
        #
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        #
        # x = self.norm(x)
        # x = self.reduction(x)

        return t


@register_model
def deit_sat_tiny_patch16_224(pretrained=False, **kwargs):
    model = SAT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def deit_sat_small_patch16_224(pretrained=False, **kwargs):
    model = SAT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_sat_base_patch16_224(pretrained=False, **kwargs):
    model = SAT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model



