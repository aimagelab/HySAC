import os
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

from hysac.dataset.visu import ViSU
from hysac.recall_computation import compute_recall
from hysac.utils.logger import WandbLogger
from hysac.models import MERUft
from hysac.losses import LorentzianCLIPContrastive, entailmentLoss_A, entailmentLoss_B, entailmentLoss_D
import hysac.lorentz as L

@torch.inference_mode()
def validate(
    model: MERUft,
    tokenizer:CLIPTokenizer,
    validation_dataset:ViSU,
    rank,
    validation_sampler,
    mode,
    contrastive_loss_function=LorentzianCLIPContrastive(),
    lambdas=(1,1,1,1,1,1,1,1),
    batch_size=2,
    clip_backbone='openai/clip-vit-large-patch14',
    wandb_activated=False,
    run=None,
    device='cuda',
    debug=False,
    wandb_logger: WandbLogger | None = None,
    aperture_scale=1.0
):
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=len(os.sched_getaffinity(0)))
    model.eval().to(device)
    lambdas = lambdas.to(device).float()

    safe_contrastive_loss_cumulative = 0
    nsfw_contrastive_loss_cumulative = 0
    safe_unsafe_contrastive_loss_a_cumulative = 0
    safe_unsafe_contrastive_loss_b_cumulative = 0

    entail_A_safe_cumulative = 0
    entail_A_nsfw_cumulative = 0
    entail_D_cumulative = 0
    validation_loss = 0

    model.curv.data = torch.clamp(model.curv.data, **model._curv_minmax)
    _curv = model.curv.exp()

    model.visual_alpha.data = torch.clamp(model.visual_alpha.data, max=0.0)
    model.textual_alpha.data = torch.clamp(model.textual_alpha.data, max=0.0)
    model.logit_scale.data = torch.clamp(model.logit_scale.data, max=4.6052)
    _scale = model.logit_scale.exp()

    for (safe_image, nsfw_image, safe_caption, nsfw_caption) in tqdm(validation_dataloader):
        this_batch_size = len(safe_caption)
        
        text_safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True)
        text_safe_ids['input_ids'] = text_safe_ids['input_ids'].to(device)
        text_safe_ids['attention_mask'] = text_safe_ids['attention_mask'].to(device)
        text_nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True)
        text_nsfw_ids['input_ids']  = text_nsfw_ids['input_ids'].to(device)
        text_nsfw_ids['attention_mask'] = text_nsfw_ids['attention_mask'].to(device)
        
        safe_image = safe_image.to(device)
        nsfw_image = nsfw_image.to(device)

        model_text_safe_embeddings = model.encode_text(text_safe_ids["input_ids"], project=True)
        model_text_nsfw_embeddings = model.encode_text(text_nsfw_ids["input_ids"], project=True)

        model_vision_safe_embeddings = model.encode_image(safe_image, project=True)
        model_vision_nsfw_embeddings = model.encode_image(nsfw_image, project=True)


        # * Losses 
        safe_contrastive_loss = contrastive_loss_function(text_feats=model_text_safe_embeddings, image_feats=model_vision_safe_embeddings, curv=_curv, text_logit_scale=_scale, image_logit_scale=_scale)
        nsfw_contrastive_loss = contrastive_loss_function(text_feats=model_text_nsfw_embeddings, image_feats=model_vision_nsfw_embeddings, curv=_curv, text_logit_scale=_scale, image_logit_scale=_scale)

        safe_unsafe_contrastive_loss_a = contrastive_loss_function(text_feats=model_text_safe_embeddings, image_feats=model_vision_nsfw_embeddings, curv=_curv, text_logit_scale=_scale, image_logit_scale=_scale)
        safe_unsafe_contrastive_loss_b = contrastive_loss_function(text_feats=model_text_nsfw_embeddings, image_feats=model_vision_safe_embeddings, curv=_curv, text_logit_scale=_scale, image_logit_scale=_scale)

        # entailment loss A
        entail_A_safe = entailmentLoss_A(model_text_safe_embeddings, model_vision_safe_embeddings, _curv, aperture_scale=aperture_scale)
        entail_A_nsfw = entailmentLoss_A(model_text_nsfw_embeddings, model_vision_nsfw_embeddings, _curv, aperture_scale=aperture_scale)

        # entailment loss D
        entail_D = entailmentLoss_D(model_vision_safe_embeddings, model_text_nsfw_embeddings, _curv, aperture_scale=aperture_scale)


        safe_contrastive_loss_cumulative += (safe_contrastive_loss['loss'] * this_batch_size).cpu()
        nsfw_contrastive_loss_cumulative += (nsfw_contrastive_loss['loss'] * this_batch_size).cpu()
        entail_A_safe_cumulative += (entail_A_safe * this_batch_size).cpu()
        entail_A_nsfw_cumulative += (entail_A_nsfw * this_batch_size).cpu()
        entail_D_cumulative += (entail_D * this_batch_size).cpu()

        safe_unsafe_contrastive_loss_a_cumulative += (safe_unsafe_contrastive_loss_a['loss'] * this_batch_size).cpu()
        safe_unsafe_contrastive_loss_b_cumulative += (safe_unsafe_contrastive_loss_b['loss'] * this_batch_size).cpu()


        losses = torch.cat(
            [
                x[(None,)+(...,)] \
                    for x in (safe_contrastive_loss["loss"], nsfw_contrastive_loss["loss"], safe_unsafe_contrastive_loss_a["loss"], safe_unsafe_contrastive_loss_b["loss"], entail_A_safe, entail_A_nsfw, entail_D)
            ]
        )

        validation_loss += (lambdas @ losses[(None,)+(...,)].T) / lambdas.numel()

        if debug:
            break

    if wandb_activated and (run is not None) and rank == 0:
        wandb_logger.log_validation_custom(
            contrastive_pres_loss=safe_contrastive_loss_cumulative / len(validation_dataset),
            nsfw_contrastive_loss=nsfw_contrastive_loss_cumulative / len(validation_dataset),
            safe_unsafe_contrastive_loss_a=safe_unsafe_contrastive_loss_a_cumulative / len(validation_dataset),
            safe_unsafe_contrastive_loss_b=safe_unsafe_contrastive_loss_b_cumulative / len(validation_dataset),
            entail_A_safe=entail_A_safe_cumulative / len(validation_dataset),
            entail_A_nsfw=entail_A_nsfw_cumulative / len(validation_dataset),
            entail_D=entail_D_cumulative / len(validation_dataset),
            validation_loss=validation_loss / this_batch_size
            )

    print('Computing Recall...')
    S_V_recall, S_G_recall, U_V_recall, U_G_recall = compute_recall(model, clip_backbone=clip_backbone, batch_size=batch_size, dataset=validation_dataset, debug=debug)
    recall_sum = S_V_recall[0][0] + S_V_recall[1][0] + S_G_recall[0][0] + S_G_recall[1][0] + U_V_recall[0][0] + U_V_recall[1][0] # we do not care to evaluate on U_G_recall

    if wandb_activated and (run is not None) and rank == 0:
        wandb_logger.log_recall([S_V_recall, S_G_recall, U_V_recall, U_G_recall], recall_sum)

    text_safe_ids['input_ids'] = text_safe_ids['input_ids'].to('cpu')
    text_safe_ids['attention_mask'] = text_safe_ids['attention_mask'].to('cpu')
    text_nsfw_ids['input_ids'] = text_nsfw_ids['input_ids'].to('cpu')
    text_nsfw_ids['attention_mask'] = text_nsfw_ids['attention_mask'].to('cpu')
    
    safe_image = safe_image.to('cpu')
    nsfw_image = nsfw_image.to('cpu')
    
    return (recall_sum, [S_V_recall, S_G_recall, U_V_recall, U_G_recall], validation_loss / len(validation_dataset))