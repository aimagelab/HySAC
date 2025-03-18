import json
from transformers import CLIPProcessor, CLIPTokenizer
from datasets import load_dataset
from tqdm import tqdm
from hysac.models import HySAC
import argparse
import torch
from hysac import lorentz as L
from PIL import Image

def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]`.
    """

    # Linear interpolation between root and image features in the tangent space of the origin.
    feats = L.log_map0(feats, model.curv.exp())

    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Lift on the Hyperboloid
    interp_feats = L.exp_map0(interp_feats, model.curv.exp())

    # Reverse the traversal order: (image first, root last)
    return interp_feats.flip(0)


def calc_scores(
    model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the given image and text features.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    scores = L.pairwise_inner(image_feats, text_feats, model.curv.exp())

    # exclude text embeddings that do not entail the given image.
    _aper = L.half_aperture(text_feats, model.curv.exp())
    _oxy_angle = L.oxy_angle(
        text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
    )
    entailment_energy = _oxy_angle - _aper[..., None]

    # Root entails everything.
    if has_root:
        entailment_energy[-1, ...] = 0

    # Set a large negative score if text does not entail image.
    scores[entailment_energy.T > 0] = -1e12
    return scores

@torch.inference_mode()
def get_text_feats_bad_words(model, clip_backbone:str) -> tuple[list[str], torch.Tensor]:
    with open('assets/bad_words.txt') as f:
        bad_words = f.readlines()
        bad_words = [s.strip() for s in bad_words]

    # Use very simple prompts for noun and adjective tags.
    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)

    all_text_feats = []

    caption_tokens = tokenizer(bad_words, return_tensors='pt', padding='max_length', truncation=True)
    all_text_feats = model.encode_text(caption_tokens["input_ids"], project=True)
    return bad_words, all_text_feats


@torch.inference_mode()
def get_text_feats_pexels(model, clip_backbone:str) -> tuple[list[str], torch.Tensor]:
    # Get all captions, nouns, and ajectives collected from pexels.com website
    pexels_text = json.load(open("assets/pexels_text.json"))

    # Use very simple prompts for noun and adjective tags.
    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)

    NOUN_PROMPT = "a photo of a {}."
    ADJ_PROMPT = "this photo is {}."

    all_text_feats = []

    # Tokenize and encode captions.
    caption_tokens = tokenizer(pexels_text["captions"], return_tensors='pt', padding='max_length', truncation=True)
    all_text_feats.append(model.encode_text(caption_tokens["input_ids"], project=True))
    noun_prompt_tokens = tokenizer(
        [NOUN_PROMPT.format(tag) for tag in pexels_text["nouns"]], return_tensors='pt', padding='max_length', truncation=True
    )
    all_text_feats.append(model.encode_text(noun_prompt_tokens["input_ids"], project=True))
    adj_prompt_tokens = tokenizer(
        [ADJ_PROMPT.format(tag) for tag in pexels_text["adjectives"]], return_tensors='pt', padding='max_length', truncation=True
    )
    all_text_feats.append(model.encode_text(adj_prompt_tokens["input_ids"], project=True))

    all_text_feats = torch.cat(all_text_feats, dim=0)
    all_pexels_text = [
        *pexels_text["captions"],
        *pexels_text["nouns"],
        *pexels_text["adjectives"],
    ]
    return all_pexels_text, all_text_feats

@torch.inference_mode()
def get_text_feats(model, clip_backbone:str, config:str, use_pexels=False, use_bad_words=False) -> tuple[list[str], torch.Tensor]:
    visu_text = load_dataset('aimagelab/ViSU-Text', split='test')
    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)
                                                
    # Tokenize and encode captions.
    all_visu_text = []
    all_text_feats = []
    for el in tqdm(visu_text):
        nsfw_text = [el["nsfw"]]
        safe_text = [el["safe"]]
        all_visu_text.extend(nsfw_text)
        all_visu_text.extend(safe_text)
        
        caption_tokens_nsfw = tokenizer(nsfw_text, return_tensors='pt', padding='max_length', truncation=True)
        caption_tokens_safe = tokenizer(safe_text, return_tensors='pt', padding='max_length', truncation=True)

        all_text_feats.append(model.encode_text(caption_tokens_nsfw["input_ids"], project=True))
        all_text_feats.append(model.encode_text(caption_tokens_safe["input_ids"], project=True))
    
    all_text_feats = torch.cat(all_text_feats, dim=0)

    if use_pexels:
        all_pexels_text, all_pexels_text_feats = get_text_feats_pexels(model,clip_backbone)
    if use_bad_words:
        all_bad_words, all_bad_words_feats = get_text_feats_bad_words(model,clip_backbone)
    
    if not use_pexels and not use_bad_words:
        return all_visu_text, all_text_feats
    elif use_pexels and not use_bad_words:
        return all_visu_text + all_pexels_text, torch.cat([all_text_feats, all_pexels_text_feats], dim=0)
    elif not use_pexels and use_bad_words:
        return all_visu_text + all_bad_words, torch.cat([all_text_feats, all_bad_words_feats], dim=0)
    else:
        return all_visu_text + all_pexels_text + all_bad_words, torch.cat([all_text_feats, all_pexels_text_feats, all_bad_words_feats], dim=0)


@torch.inference_mode()
def get_traversal(model, clip_backbone: str, image_path: str = None, config: dict = None):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    root_feat = torch.zeros(model.embed_dim, device=device)
    text_pool, text_feats_pool = get_text_feats(model, clip_backbone, config=config, use_pexels=True, use_bad_words=True)

    # Add [ROOT] to the pool of text feats.
    text_pool.append("[ROOT]")
    text_feats_pool = torch.cat([text_feats_pool, root_feat[None, ...]])
    print(f"\nPerforming image traversals with source: {image_path}...")
    imageprocessor = CLIPProcessor.from_pretrained(clip_backbone)

    image = Image.open(image_path)
    image = imageprocessor(images=image, return_tensors="pt")['pixel_values']
    image_feats = model.encode_image(image[None, ...].to(device), project=True)[0]

    interp_feats = interpolate(model, image_feats, root_feat, 50)
    nn1_scores = calc_scores(model, interp_feats, text_feats_pool, has_root=True)

    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_texts = [text_pool[_idx.item()] for _idx in _nn1_idxs]

    unique_nn1_texts = []
    for _text in nn1_texts:
        if _text not in unique_nn1_texts:
            print(_text)
            unique_nn1_texts.append(_text)

def main(args):
    clip_backbone='openai/clip-vit-large-patch14'
    model = HySAC.from_pretrained(repo_id="aimagelab/hysac", device="cuda").to("cuda")
    image_path = args.image_path

    with torch.inference_mode():
        get_traversal(model, clip_backbone=clip_backbone, image_path=image_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="Path to the image for which the traversal is to be computed.")
    args = parser.parse_args()
    main(args)