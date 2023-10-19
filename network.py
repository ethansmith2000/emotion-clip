

from torch import nn
import torch
import torchvision.transforms as T
import numpy as np


import torch
import xformers
import xformers.ops as xops
from torch import nn

import random


class Attention(nn.Module):
    def __init__(self, query_dim=768, context_dim=1024,
                 heads=8, dropout=0.0,
                 use_xformers=True,
                 ):
        super().__init__()
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.heads = heads
        self.dim_head = query_dim // heads
        self.scale = 1 / (self.dim_head ** 0.5)
        self.out_proj = nn.Linear(query_dim, query_dim)


        self.norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_xformers = use_xformers

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor


    def forward(self, x, context):
        b, n, _ = x.shape

        resid_x = x
        norm_x = self.norm(x)

        q = self.head_to_batch_dim(self.to_q(norm_x))
        k = self.head_to_batch_dim(self.to_k(context))
        v = self.head_to_batch_dim(self.to_v(context))

        if self.use_xformers:
            attn_output = xops.memory_efficient_attention(
                q.contiguous(), k.contiguous(), v.contiguous(), scale=self.scale,
                attn_bias=xops.LowerTriangularMask()
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            attn_output = (attn @ v).transpose(1, 2).reshape(b, n, -1)

        attn_output = self.batch_to_head_dim(attn_output)

        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        x = resid_x + attn_output

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))


class TransformerLayer(nn.Module):

    def __init__(self, query_dim=768, context_dim=1024,
                 heads=8, dropout=0.0,
                 use_xformers=True, ff_mult=4, use_cross_attn=False):

        super().__init__()
        # self.self_attn = Attention(query_dim=query_dim,
        #          context_dim=context_dim,
        #          heads=heads,
        #          dropout=dropout,
        #          use_xformers=use_xformers,)
        if use_cross_attn:
            self.cross_attn = Attention(query_dim=query_dim,
                    context_dim=context_dim,
                    heads=heads,
                    dropout=dropout,
                    use_xformers=use_xformers,)
        else:
            self.cross_attn = None

        # self.ff = FeedForward(query_dim, mult=ff_mult, dropout=dropout)
        self.gradient_checkpointing = False

    def forward(self, x, context):
        if self.gradient_checkpointing:
            #x = torch.utils.checkpoint.checkpoint(self.self_attn, x, x)
            if self.cross_attn is not None:
                x = torch.utils.checkpoint.checkpoint(self.cross_attn, x, context)
            #x = torch.utils.checkpoint.checkpoint(self.ff, x)

        else:
            #x = self.self_attn(x, x)
            if self.cross_attn is not None:
                x = self.cross_attn(x, context)
            #x = self.ff(x)

        return x


def gaussian(M, std, sym=True):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w

def preprocess_image(image, fit_method="pad",num_cutouts=5, is_clip=True, color_jitter=0.1):

    if isinstance(fit_method, list):
        fit_method = random.choice(fit_method)

    transforms = [
        T.Resize(224),
        T.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=color_jitter),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
    if is_clip:
        transforms.append(T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]))
    else:
        #DINOv2
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if fit_method == "sliding":
        crop_fn = lambda x: sliding_cutouts(x, num_cuts=num_cutouts)
    elif fit_method == "center":
        crop_fn = T.CenterCrop(224)
    elif fit_method == "resize":
        crop_fn = T.Resize((224,224))
    elif fit_method == "pad":
        def pad_and_resize(img):
            h,w = img.shape[-2:]
            if not h == w:
                if w > h:
                    img = T.Pad((0, (w-h)//2, 0, (w-h)//2))(img)
                else:
                    img = T.Pad(((h-w)//2, 0, (h-w)//2, 0))(img)
                return T.Resize((224,224))(img)
            else:
                return img
        crop_fn = pad_and_resize

    transforms = T.Compose(transforms)
    image = transforms(image)

    image = crop_fn(image)

    return image

def sliding_cutouts(tensor, num_cuts=5, cut_size=224):
    cutouts = []
    sideY, sideX = tensor.shape[-2:]
    largersize = max(sideX, sideY)
    if sideX < sideY:
        smaller_side = "sideX"
    else:
        smaller_side = "sideY"

    cut_spots = torch.linspace(0, largersize - cut_size, num_cuts)
    for cut in cut_spots:
        if smaller_side == "sideX":
            cutout = tensor[:, int(cut):int(cut) + cut_size, :]
        else:
            cutout = tensor[:, :, int(cut):int(cut) + cut_size]
        cutouts.append(cutout)

    cutouts = torch.stack(cutouts)

    return cutouts


class Network(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=384, act_fn="silu", pre_layernorm=True, num_transformer_layers=3, num_queries=2, ff_mult=2):
        super().__init__()

        if "silu" in act_fn.lower():
            act_fn = nn.SiLU()
        elif "gelu" in act_fn.lower():
            act_fn = nn.GELU()
        elif "relu" in act_fn.lower():
            act_fn = nn.ReLU()
        else:
            act_fn = None



        if num_transformer_layers > 0:
            self.transformer_layers = nn.ModuleList([TransformerLayer(query_dim=hidden_dim, context_dim=hidden_dim, use_cross_attn=True, ff_mult=ff_mult) for _ in range(num_transformer_layers)])
            self.learned_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

            if pre_layernorm:
                self.proj_in = nn.Linear(input_dim, hidden_dim)
                self.layer_norm = nn.LayerNorm(hidden_dim)
            else:
                self.layer_norm = None
                self.proj_in = None

        else:
            self.transformer_layers = None
            self.learned_queries = None

        def make_classifier(_in_dim, _out_dim, _act_fn=None):
            if _act_fn is None:
                return nn.Linear(_in_dim, 1)
            else:
                return nn.Sequential(
                    nn.Linear(_in_dim, _out_dim),
                    act_fn,
                    nn.Linear(_out_dim, 1),
                )

        self.valence_classifier = make_classifier(hidden_dim, hidden_dim, act_fn)
        self.arousal_classifier = make_classifier(hidden_dim, hidden_dim, act_fn)
        # self.dominance_classifier = make_classifier(in_dim, out_dim, act_fn)
        # self.dominance_2_classifier = make_classifier(in_dim, out_dim, act_fn)

        self.gradient_checkpointing = False
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        if self.transformer_layers is not None:
            for layer in self.transformer_layers:
                layer.gradient_checkpointing = True

    def forward(self, context):

        if self.transformer_layers is not None:
            queries = self.learned_queries.unsqueeze(0).repeat(context.shape[0], 1, 1)

            if self.layer_norm is not None:
                context = self.proj_in(context)
                context = self.layer_norm(context)

            for layer in self.transformer_layers:
                queries = layer(queries, context)

            queries = queries.chunk(4, dim=1)
            embeds = [q.squeeze(1) for q in queries]
        else:
            embeds = context[:, 0, :]
            embeds = [embeds, embeds]

        valence = self.valence_classifier(embeds[0])
        arousal = self.arousal_classifier(embeds[1])
        # dominance = self.dominance_classifier(queries[2])
        # dominance_2 = self.dominance_2_classifier(queries[3])

        return valence, arousal#, dominance, dominance_2



