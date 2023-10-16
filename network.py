

from torch import nn
import torch
import torchvision.transforms as T
import numpy as np

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

def preprocess_image(image, fit_method="pad",num_cutouts=5, is_clip=True):
    transforms = [
        T.Resize(224),
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
    def __init__(self, in_dim=768, out_dim=384, act_fn="silu", pre_layernorm=True):
        super().__init__()

        if "silu" in act_fn.lower():
            self.act_fn = nn.SiLU()
        elif "gelu" in act_fn.lower():
            self.act_fn = nn.GELU()
        elif "relu" in act_fn.lower():
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.Identity()

        if pre_layernorm:
            self.layer_norm = nn.LayerNorm(in_dim)
        else:
            self.layer_norm = None

        if "Identity" not in self.act_fn.__class__.__name__:
            self.linear = nn.Linear(in_dim, out_dim)
            self.classifier = nn.Linear(out_dim, 1)
        else:
            self.linear = None
            self.classifier = nn.Linear(in_dim, 1)


    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, 0]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.linear is not None:
            x = self.act_fn(self.linear(x))

        return self.classifier(x)



