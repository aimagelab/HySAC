import math
import json
import torch
from torch import nn
from torch.nn import functional as F

from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, PeftModel
import hysac.utils.distributed as dist
from hysac import lorentz as L


class CLIPBaseline(nn.Module):
    def __init__(
        self,
        visual: CLIPVisionModelWithProjection,
        textual: CLIPTextModelWithProjection,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        freeze_logit_scale: bool = False,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        if freeze_logit_scale:
            self.register_buffer("logit_scale", torch.tensor(4.6052))
        else:
            self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log()) # learnt in logaritmic space

        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):
        # normalization happens in dataloader
        
        image_feats = self.visual(pixel_values=images).image_embeds

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.config.max_position_embeddings:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.config.max_position_embeddings]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(input_ids=tokens).text_embeds

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats


class HySAC(CLIPBaseline):
    """
    Our HySAC model, that modifies CLIP to embed images and text in a safety-aware hyperbolic space.
    """

    def __init__(
        self,
        visual: CLIPVisionModelWithProjection | PeftModel,
        textual: CLIPTextModelWithProjection,
        embed_dim: int = 768,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        freeze_logit_scale: bool = False,
        bounds: dict[str, float] = None,
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
        """
        super().__init__(visual, textual, embed_dim, pixel_mean, pixel_std, freeze_logit_scale)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

        self.bounds = bounds

    @classmethod
    def from_pretrained(cls, repo_id: str = "aimagelab/HySAC", device="cpu"):
        # Download model weights
        model_path = hf_hub_download(repo_id, "hysac_model.pth")
        bounds_path = hf_hub_download(repo_id=repo_id, filename="hysac_bounds.json")

        with open(bounds_path, "r") as f:
            bounds = json.load(f)

        # Load CLIP models
        clip_backbone='openai/clip-vit-large-patch14'
        peft_config = LoraConfig(
            r=16,
            lora_alpha=1,
            target_modules=["k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"],
            lora_dropout=0.1,
            bias="none",
        )

        text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_backbone)
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_backbone)
        text_encoder = get_peft_model(text_encoder, peft_config)
        vision_encoder = get_peft_model(vision_encoder, peft_config)

        # Instantiate model
        model = cls(visual=vision_encoder, textual=text_encoder, bounds=bounds)
        model.load_state_dict(torch.load(model_path, map_location=device))

        return model

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(images, project=False)
        
        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            image_feats = L.exp_map0(image_feats, self.curv.exp())

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(tokens, project=False)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            text_feats = L.exp_map0(text_feats, self.curv.exp())

        return text_feats
    
    def traverse_to_safe_image(self, emb, alpha=0.8):
        """
        Modify an input embedding by performing safety traversal towards the safe_image area.
        The traversal is only performed if the input embedding is farther then the NSFW threshold. 
        """
        upper_bound = self.bounds["V"]["up_95"] # 95th percentile of the safe_image distribution
        with torch.inference_mode():
            root_feat = torch.zeros_like(emb)
            distance = L.pairwise_dist(root_feat, emb, curv=self.curv.exp())

            upper_bound = upper_bound + (math.tanh((upper_bound - alpha) / self.curv.exp()) + 1)
            
            if distance > upper_bound:

                # move point to upper_bound distance from root_feat
                direction = emb - root_feat
                n_direction = direction / torch.norm(direction, dim=-1)
                target_emb = root_feat + upper_bound * n_direction
                return target_emb
            else:
                return emb
    
    def traverse_to_safe_text(self, emb, alpha=0.8):
        """
        Modify an input embedding by performing safety traversal towards the safe_text area.
        The traversal is only performed if the input embedding is farther then the NSFW threshold. 
        """
        upper_bound = self.bounds["S"]["up_95"] # 95th percentile of the safe_text distribution
        with torch.inference_mode():
            root_feat = torch.zeros_like(emb)
            distance = L.pairwise_dist(root_feat, emb, curv=self.curv.exp())

            upper_bound = upper_bound + (math.tanh((upper_bound - alpha) / self.curv.exp()) + 1)
            
            if distance > upper_bound:

                # move point to upper_bound distance from root_feat
                direction = emb - root_feat
                n_direction = direction / torch.norm(direction, dim=-1)
                target_emb = root_feat + upper_bound * n_direction
                return target_emb
            else:
                return emb

class CLIPWrapper(nn.Module):
    def __init__(self, te, ve, normalize=True):
        super().__init__()
        self.text_encoder = te
        self.vision_encoder = ve
        self.normalize = normalize

    def encode_text(self, x):
        feats = self.text_encoder(**{'input_ids':x})['text_embeds']
        if self.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_image(self, x):
        feats = self.vision_encoder(**{'pixel_values': x})['image_embeds']
        if self.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats