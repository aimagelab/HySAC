import torch
from torch import nn
import torch.nn.functional as F
import hysac.lorentz as L

class LorentzianCLIPContrastive(nn.Module):
    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_feats, text_feats, curv, unique_logit_scale=None, image_logit_scale=None, text_logit_scale=None, validation=False):
        image_embed = image_feats
        text_embed = text_feats
        if image_logit_scale is None and text_logit_scale is None:
            if unique_logit_scale is None:
                raise ValueError('At least (unique_logit_scale) or (image_logit_scale and text_logit_scale) must be provided')
            image_logit_scale = unique_logit_scale
            text_logit_scale = unique_logit_scale
        elif image_logit_scale is None or text_logit_scale is None:
            raise ValueError('At least (unique_logit_scale) or (image_logit_scale and text_logit_scale) must be provided')
        else:
            image_logit_scale = image_logit_scale
            text_logit_scale = text_logit_scale

        _curv = curv
        local_batch_size = text_embed.shape[0]

        if local_batch_size != self.last_local_batch_size:
            self.labels = torch.arange(local_batch_size, device=image_embed.device)   
            self.last_local_batch_size = local_batch_size

        image_embed_all = image_embed
        text_embed_all = text_embed

        image_logits = -L.pairwise_dist(image_embed, text_embed_all, _curv)
        text_logits = -L.pairwise_dist(text_embed, image_embed_all, _curv)

        contrastive_loss = 0.5 * (
            nn.functional.cross_entropy(image_logit_scale * image_logits, self.labels)
            + nn.functional.cross_entropy(text_logit_scale * text_logits, self.labels)
        )
        
        return {
            'loss': contrastive_loss, 
            'values': {
                'image': image_logits[:local_batch_size, :local_batch_size], 
                'text': text_logits[:local_batch_size, :local_batch_size],
                'text_logit_scale': text_logit_scale,
                'image_logit_scale': image_logit_scale,
                'curv': _curv
            },
        }

class CLIPContrastive(nn.Module):
    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_feats, text_feats, unique_logit_scale=None, image_logit_scale=None, text_logit_scale=None, validation=False):
        image_embed = image_feats
        text_embed = text_feats
        if image_logit_scale is None and text_logit_scale is None:
            if unique_logit_scale is None:
                raise ValueError('At least (unique_logit_scale) or (image_logit_scale and text_logit_scale) must be provided')
            image_logit_scale = unique_logit_scale
            text_logit_scale = unique_logit_scale
        elif image_logit_scale is None or text_logit_scale is None:
            raise ValueError('At least (unique_logit_scale) or (image_logit_scale and text_logit_scale) must be provided')
        else:
            image_logit_scale = image_logit_scale
            text_logit_scale = text_logit_scale

        local_batch_size = text_embed.shape[0]

        if local_batch_size != self.last_local_batch_size:
            self.labels = torch.arange(local_batch_size, device=image_embed.device)   
            self.last_local_batch_size = local_batch_size

        image_embed_all = image_embed
        text_embed_all = text_embed

        image_logits = image_embed @ text_embed_all.T
        text_logits = text_embed @ image_embed_all.T

        contrastive_loss = 0.5 * (
            nn.functional.cross_entropy(image_logit_scale * image_logits, self.labels)
            + nn.functional.cross_entropy(text_logit_scale * text_logits, self.labels)
        )
        
        return {
            'loss': contrastive_loss, 
            'values': {
                'image': image_logits[:local_batch_size, :local_batch_size], 
                'text': text_logits[:local_batch_size, :local_batch_size],
                'text_logit_scale': text_logit_scale,
                'image_logit_scale': image_logit_scale,
            },
        }

def reversedEntailmentLoss(text_safe_feats, text_nsfw_feats, image_nsfw_feats, curv):
    # exterior angle computation
    nsfw_image_angle = L.oxy_angle(text_nsfw_feats, image_nsfw_feats, curv)

    nsfw_aperture = L.half_aperture(text_nsfw_feats, curv)
    safe_aperture = L.half_aperture(text_safe_feats, curv)

    a_loss = torch.clamp(nsfw_aperture - nsfw_image_angle, min=0).mean()
    a_reg = torch.clamp(nsfw_image_angle - safe_aperture, min=0).mean()

    return a_loss + a_reg

def entailmentLoss_A(text_feats, image_feats, _curv, aperture_scale=1.0):
    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle(text_feats, image_feats, _curv)
    _aperture = L.half_aperture(text_feats, _curv)
    entailment_loss = torch.clamp(_angle - (aperture_scale * _aperture), min=0).mean()

    return entailment_loss

def entailmentLoss_B(safe_feats, nsfw_feats, _curv, aperture_scale=1.0):
    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle(safe_feats, nsfw_feats, _curv)
    _aperture = L.half_aperture(safe_feats, _curv)
    entailment_loss = torch.clamp(_angle - (aperture_scale * _aperture), min=0).mean()

    return entailment_loss

def entailmentLoss_D(safe_vision_feats, nsfw_text_feats, _curv, aperture_scale=1.0):
    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle(safe_vision_feats, nsfw_text_feats, _curv)
    _aperture = L.half_aperture(safe_vision_feats, _curv)
    entailment_loss = torch.clamp(_angle - (aperture_scale * _aperture), min=0).mean()

    return entailment_loss

def entailmentLoss(x, y, curv):
    # Hyperbolic entailment loss: x entails y with respect to the origin O.
    _angle = L.oxy_angle(x, y, curv)
    _aperture = L.half_aperture(x, curv)
    entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()
    return entailment_loss

def entailmentLoss_withRoot(x, y, r, curv):
    # Hyperbolic entailment loss: x entails y with respect to the root r.
    _angle = L.oxy_angle_withroot(x, y, r, curv)
    _aperture = L.half_aperture_withroot(x, r, curv)
    entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()
    return entailment_loss