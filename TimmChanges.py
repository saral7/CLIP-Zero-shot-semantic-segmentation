import open_clip
from open_clip import timm_model

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

try:
    import timm

    """ try: """
    # new timm imports >= 0.8.1
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d
    from timm.layers import Mlp, to_2tuple

    """ except ImportError as e:
        # fallback, try old timm imports < 0.8.1
        from timm.models.layers.attention_pool2d import RotAttentionPool2d
        from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
        from timm.models.layers import Mlp, to_2tuple """
except ImportError:
    timm = None


def init(
    self,
    model_name,
    embed_dim,
    image_size=224,
    pool="avg",
    proj="linear",
    proj_bias=False,
    drop=0.0,
    drop_path=None,
    patch_drop=None,
    pretrained=False,
):
    nn.Module.__init__(self)
    if timm is None:
        raise RuntimeError("Please `pip install timm` to use timm models.")
    self.image_size = to_2tuple(image_size)

    # setup kwargs that may not be common across all models
    timm_kwargs = {}
    if drop_path is not None:
        timm_kwargs["drop_path_rate"] = drop_path
    if patch_drop is not None:
        timm_kwargs["patch_drop_rate"] = patch_drop

    custom_pool = pool in ("abs_attn", "rot_attn")
    if proj:
        assert proj in ("linear", "mlp", "none")
    extra_proj = proj in ("linear", "mlp")
    if not extra_proj and not custom_pool:
        # use network classifier head as projection if no proj specified and no custom pooling used
        # if projection is explicitly set to "none" will be pass through from network trunk
        proj_dim = 0 if proj == "none" else embed_dim
        self.trunk = timm.create_model(
            model_name,
            num_classes=proj_dim,
            global_pool=pool,
            pretrained=pretrained,
            **timm_kwargs,
        )
        prev_chs = embed_dim
    else:
        print("kreiram tim")
        timm_kwargs["global_pool"] = ""  # ADDED
        self.trunk = timm.create_model(
            model_name,
            pretrained=pretrained,
            **timm_kwargs,
        )
        feat_size = self.trunk.default_cfg.get("pool_size", None)
        feature_ndim = 1 if not feat_size else 2
        if custom_pool:
            assert feature_ndim == 2
            # if attn pooling used, remove both classifier and default pool
            self.trunk.reset_classifier(0, global_pool="")
        else:
            # reset global pool if pool config set, otherwise leave as network default
            reset_kwargs = dict(global_pool=pool) if pool else {}
            self.trunk.reset_classifier(0, **reset_kwargs)
        prev_chs = self.trunk.num_features

    head_layers = OrderedDict()

    # Add custom pooling to head
    if pool == "abs_attn":
        head_layers["pool"] = AbsAttentionPool2d(
            prev_chs, feat_size=feat_size, out_features=embed_dim
        )
        prev_chs = embed_dim
    elif pool == "rot_attn":
        head_layers["pool"] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
        prev_chs = embed_dim

    # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
    if proj == "linear":
        head_layers["drop"] = nn.Dropout(drop)
        head_layers["proj"] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
    elif proj == "mlp":
        head_layers["mlp"] = Mlp(
            prev_chs,
            2 * embed_dim,
            embed_dim,
            drop=(drop, 0),
            bias=(True, proj_bias),
        )

    self.head = nn.Sequential(head_layers)


# Promijenjena forward funkcija
def myForward(self, x):
    x = self.trunk(x)
    # tu permutacija
    # tu bilinearno tenzora
    # iz validate, mapiraj
    print(np.shape(x[0]))
    # print("shape", np.shape(x))  # torch.Size([1, 1024, 32, 64])
    x = torch.permute(x, (0, 2, 3, 1))[0]
    print("shape3", np.shape(x))
    x = self.head(x)
    # print("shape4", np.shape(x))  # torch.Size([32, 64, 640])
    x = x.unsqueeze(0)
    # print("shape5", np.shape(x))  # torch.Size([1, 32, 64, 640])
    x = torch.permute(x, (0, 3, 1, 2))
    # print("shape6", np.shape(x))

    print("shape2", np.shape(x))
    return x


def myForward(self, x):
    x = self.trunk(x)  # torch.Size([1, 1024, 32, 64])
    x = torch.permute(x, (0, 2, 3, 1))[0]  # torch.Size([32, 64, 1024])

    x = self.head(x)  # torch.Size([32, 64, 640])
    x = x.unsqueeze(0)  # torch.Size([1, 32, 64, 640])
    x = torch.permute(x, (0, 3, 1, 2))  # torch.Size([1, 640, 32, 64])

    return x
