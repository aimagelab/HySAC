import os
import tqdm
import torch
from torch.utils.data import DataLoader, Subset

from transformers import CLIPTokenizer

import hysac.lorentz as L

def encode_dataset(model, dataloader, clip_backbone:str, batch_size:int=32, debug:bool=False, device='cuda'):
    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)

    all_text_safe_embeddings = []
    all_visual_safe_embeddings = []
    all_text_nsfw_embeddings = []
    all_visual_nsfw_embeddings = []
    
    split = dataloader.dataset.dataset.split if debug else dataloader.dataset.split
    
    with torch.inference_mode():
        for (safe_image, nsfw_image, safe_caption, nsfw_caption) in tqdm.tqdm(dataloader, desc=f"Extracting {split}-dataset features"):
            safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True)
            safe_ids['input_ids'] = safe_ids['input_ids'].to(device)
            nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True)
            nsfw_ids['input_ids']  = nsfw_ids['input_ids'].to(device)

            text_safe_embeddings = model.encode_text(safe_ids["input_ids"], project=True)
            text_nsfw_embeddings = model.encode_text(nsfw_ids["input_ids"], project=True)
            safe_ids = safe_ids.to('cpu')
            nsfw_ids = nsfw_ids.to('cpu')

            safe_image = safe_image.to(device)
            nsfw_image = nsfw_image.to(device)

            visual_safe_embeddings = model.encode_image(safe_image, project=True)
            visual_nsfw_embeddings = model.encode_image(nsfw_image, project=True)
            safe_image = safe_image.to('cpu')
            nsfw_image = nsfw_image.to('cpu')

            all_text_safe_embeddings.append(text_safe_embeddings)
            all_text_nsfw_embeddings.append(text_nsfw_embeddings)
            all_visual_safe_embeddings.append(visual_safe_embeddings)
            all_visual_nsfw_embeddings.append(visual_nsfw_embeddings)

        all_text_safe_embeddings = torch.cat(all_text_safe_embeddings, 0)
        all_text_nsfw_embeddings = torch.cat(all_text_nsfw_embeddings, 0)
        all_visual_safe_embeddings = torch.cat(all_visual_safe_embeddings, 0)
        all_visual_nsfw_embeddings = torch.cat(all_visual_nsfw_embeddings, 0)    

    return {
        "all_text_safe_embeddings": all_text_safe_embeddings,
        "all_text_nsfw_embeddings": all_text_nsfw_embeddings,
        "all_visual_safe_embeddings": all_visual_safe_embeddings,
        "all_visual_nsfw_embeddings": all_visual_nsfw_embeddings
    }

def compute_recall(model, dataset, clip_backbone, batch_size=32, debug=False):
    if debug:
        dataset = Subset(dataset, range(20))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=len(os.sched_getaffinity(0)), pin_memory=True)
    
    encoded_data = encode_dataset(model, dataloader, clip_backbone, batch_size, debug=debug)

    all_text_safe_embeddings = encoded_data["all_text_safe_embeddings"]
    all_text_nsfw_embeddings = encoded_data["all_text_nsfw_embeddings"]
    all_visual_safe_embeddings = encoded_data["all_visual_safe_embeddings"]
    all_visual_nsfw_embeddings = encoded_data["all_visual_nsfw_embeddings"]

    K=(1,10,20)

    S_V_ranks = recall(all_text_safe_embeddings, all_visual_safe_embeddings, K=K, model=model)
    S_G_ranks = recall(all_text_safe_embeddings, all_visual_nsfw_embeddings, K=K, model=model)
    U_V_ranks = recall(all_text_nsfw_embeddings, all_visual_safe_embeddings, K=K, model=model)
    U_G_ranks = recall(all_text_nsfw_embeddings, all_visual_nsfw_embeddings, K=K, model=model)

    return S_V_ranks, S_G_ranks, U_V_ranks, U_G_ranks


def recall(temb, vemb, mode='hyp', K=(1,10,20), model=None): 
    num_text = temb.shape[0]
    num_im = vemb.shape[0]
    text_to_image_map = image_to_text_map = torch.LongTensor(tuple(i for i in range(num_text)))#.unsqueeze(-1)

    # text-to-image recall
    print("Text-to-image recall...")

    if mode == 'euc':
        dist_matrix = temb @ vemb.T
    elif mode == 'hyp':
        dist_matrix = L.pairwise_inner(temb.cpu(), vemb.cpu(), curv=model.curv.exp().cpu())
    else:
        raise ValueError("Invalid mode")

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    text_to_image_recall = []

    for k in K:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)


    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    image_to_text_recall = []

    for k in K:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.eq(topk, image_to_text_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)

    print("Done.")
    return text_to_image_recall, image_to_text_recall