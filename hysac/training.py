import os
import itertools

import torch
from torch import cuda
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from torch.optim.adamw import AdamW
from torch.cuda import amp

from hysac.dataset.visu import ViSU
from hysac.models import HySAC
from hysac.losses import LorentzianCLIPContrastive, entailmentLoss_A, entailmentLoss_D # these two losses are the same but kept separated for clarity
from hysac.optim import set_weight_decay_per_param
from hysac.utils.checkpointing import CheckpointManager
from hysac.utils.logger import WandbLogger, summarize
from hysac.validation import validate
from hysac.utils.distributed import gather_across_processes

@torch.enable_grad()
def training(
    model: HySAC,
    tokenizer:CLIPTokenizer,
    train_dataset:ViSU,
    validation_dataset:ViSU,
    rank,
    device,
    training_sampler,
    validation_sampler,
    mode,
    contrastive_loss_function=LorentzianCLIPContrastive(),
    lambdas=(1,1,1,1,1,1,1,1),
    batch_size=2,
    lr=1e-5,
    epoches=10,
    gradient_accumulation_steps=1,
    initial_patience=5,
    wandb_activated=False,
    run=None,
    best_checkpoint_saving_path='',
    clip_backbone = 'openai/clip-vit-large-patch14',
    debug=False,
    wandb_logger: WandbLogger | None = None,
    resume=False,
    enable_grad_scaler=True,
    enable_amp_autocast=False,
    aperture_scale=1.0
):

    for n,m in itertools.chain(model.textual.named_parameters(), model.visual.named_parameters()): 
        if 'lora' in n: 
            m.requires_grad_(True)
        else: 
            m.requires_grad_(False)

    optimizer = AdamW(
        params=set_weight_decay_per_param(
            model,
            weight_decay=0.2,
            gain_bias_decay=0.0,
            exclude_params=[
                "logit_scale", "visual_alpha", "textual_alpha", "curv"
            ],
        ),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=0.2
    )

    scaler = amp.GradScaler(enabled=enable_grad_scaler)
    
    checkpoint_manager = CheckpointManager(
        best_checkpoint_saving_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler
    )

    initial_epoch, best_validation_loss, best_recall_sum, patience = checkpoint_manager.resume(mode='last') if resume else (-1, 9999, 0, initial_patience)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=training_sampler, num_workers=len(os.sched_getaffinity(0)))

    # * Initialization of state variables
    for epoch in range(initial_epoch + 1, epoches):

        training_sampler.set_epoch(epoch)

        model.train().to(device)
        lambdas = lambdas.to(device).float()

        training_start = cuda.Event(enable_timing=True)
        validation_start = cuda.Event(enable_timing=True)
        training_end = cuda.Event(enable_timing=True)
        validation_end = cuda.Event(enable_timing=True)

        training_start.record()

        for idx, (safe_image, nsfw_image, safe_caption, nsfw_caption) in enumerate(tqdm(train_dataloader)):

            optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(True):

                with amp.autocast(enabled=enable_amp_autocast):
                    # * Encoding the input data
                    text_safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True)
                    text_safe_ids['input_ids'] = text_safe_ids['input_ids'].to(device)
                    text_safe_ids['attention_mask'] = text_safe_ids['attention_mask'].to(device)
                    text_nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True)
                    text_nsfw_ids['input_ids'] = text_nsfw_ids['input_ids'].to(device)
                    text_nsfw_ids['attention_mask'] = text_nsfw_ids['attention_mask'].to(device)
                    
                    safe_image = safe_image.to(device)
                    nsfw_image = nsfw_image.to(device)

                    model.curv.data = torch.clamp(model.curv.data, **model._curv_minmax)
                    _curv = model.curv.exp()

                    # Clamp scaling factors such that they do not up-scale the feature norms.
                    # Once `exp(scale) = 1`, they can simply be removed during inference.
                    model.visual_alpha.data = torch.clamp(model.visual_alpha.data, max=0.0)
                    model.textual_alpha.data = torch.clamp(model.textual_alpha.data, max=0.0)
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, max=4.6052, min=torch.tensor(1/0.07).log().item())
                    _scale = model.logit_scale.exp()

                    model_text_safe_embeddings = model.encode_text(text_safe_ids["input_ids"], project=True)
                    model_text_nsfw_embeddings = model.encode_text(text_nsfw_ids["input_ids"], project=True)

                    model_vision_safe_embeddings = model.encode_image(safe_image, project=True)
                    model_vision_nsfw_embeddings = model.encode_image(nsfw_image, project=True)

                    all_model_text_safe_embeddings = gather_across_processes(model_text_safe_embeddings)
                    all_model_text_nsfw_embeddings = gather_across_processes(model_text_nsfw_embeddings)

                    all_model_vision_safe_embeddings = gather_across_processes(model_vision_safe_embeddings)
                    all_model_vision_nsfw_embeddings = gather_across_processes(model_vision_nsfw_embeddings)

                    all_model_text_safe_embeddings = torch.cat(all_model_text_safe_embeddings, dim=0)
                    all_model_text_nsfw_embeddings = torch.cat(all_model_text_nsfw_embeddings, dim=0)

                    all_model_vision_safe_embeddings = torch.cat(all_model_vision_safe_embeddings, dim=0)
                    all_model_vision_nsfw_embeddings = torch.cat(all_model_vision_nsfw_embeddings, dim=0)

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

                losses = torch.cat(
                    [
                        x[(None,)+(...,)] \
                            for x in (safe_contrastive_loss["loss"], nsfw_contrastive_loss["loss"], safe_unsafe_contrastive_loss_a["loss"], safe_unsafe_contrastive_loss_b["loss"], entail_A_safe, entail_A_nsfw, entail_D)
                    ]
                )

                training_loss = (lambdas @ losses[(None,)+(...,)].T) / lambdas.numel()

                if wandb_activated and (run is not None) and rank == 0:
                    wandb_logger.log_training_iteration_custom(
                                    safe_contrastive_loss=safe_contrastive_loss["loss"].mean(),
                                    nsfw_contrastive_loss=nsfw_contrastive_loss["loss"].mean(),
                                    safe_unsafe_contrastive_loss_a=safe_unsafe_contrastive_loss_a["loss"].mean(),
                                    safe_unsafe_contrastive_loss_b=safe_unsafe_contrastive_loss_b["loss"].mean(),
                                    entail_A_safe=entail_A_safe.mean(),
                                    entail_A_nsfw=entail_A_nsfw.mean(),
                                    entail_D=entail_D.mean(),
                                    training_loss=training_loss,
                                    batch_id=idx
                    ) 

                scaler.scale(training_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if debug:
                break
        
        training_end.record()
        cuda.synchronize()
        training_time = training_start.elapsed_time(training_end)/1000
        validation_start.record()
        print('Validating...')
        this_recall_sum, this_recalls, this_validation_loss = validate(
            model=model,
            tokenizer=tokenizer,
            validation_dataset=validation_dataset,
            rank=rank,
            validation_sampler=validation_sampler,
            contrastive_loss_function=contrastive_loss_function,
            lambdas=lambdas,
            batch_size=batch_size,
            clip_backbone=clip_backbone,
            wandb_activated=wandb_activated,
            run=run,
            device=device,
            debug=debug,
            wandb_logger=wandb_logger,
            mode=mode,
            aperture_scale=aperture_scale
        )
        validation_end.record()
        cuda.synchronize()
        validation_time = validation_start.elapsed_time(validation_end)/1000

        # * Evaluating exit criterion
        if rank == 0:
            criterion_to_best_checkpoint = this_recall_sum > best_recall_sum

            if this_validation_loss > best_validation_loss and not criterion_to_best_checkpoint:
                patience -= 1
                if patience == 0:
                    break

            else:
                patience = initial_patience
                
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    if wandb_activated and (run is not None) and rank == 0:
                        run.log({
                            'best_validation_loss': best_validation_loss
                        })

                    checkpoint_manager.best('validation-loss', epoch, best_validation_loss, best_recall_sum, patience)
                
                if criterion_to_best_checkpoint:
                    best_recall_sum = this_recall_sum
                    if wandb_activated and (run is not None) and rank == 0:
                        run.log({
                            'best_recall_sum': best_recall_sum
                        })
                    checkpoint_manager.best('recall', epoch, best_validation_loss, best_recall_sum, patience)

            if wandb_activated and (run is not None) and rank == 0:
                wandb_logger.log_patience(patience)

            checkpoint_manager.step(epoch, best_validation_loss, best_recall_sum, patience)

            summarize(
                epoch, patience, training_loss, this_validation_loss, this_recalls, best_recall_sum, best_validation_loss, training_time, validation_time, best_checkpoint_saving_path
            )