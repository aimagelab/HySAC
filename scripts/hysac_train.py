import os
import time
import json
import ast
import math
import torch.distributed
import wandb
import torch
import argparse
torch.distributed.init_process_group(backend='nccl')
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from transformers import CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from peft import LoraConfig, get_peft_model

from hysac.dataset.visu import ViSU
from hysac.losses import LorentzianCLIPContrastive
from hysac.models import HySAC
from hysac.utils.argumentparser import parse_arguments
from hysac.training import training
from hysac.utils.logger import WandbLogger

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print(f'Rank: {rank}, Local rank: {local_rank}, Device: {device}')


# reproducibility
torch.manual_seed(42 + rank)

def main(args):
    if args.debug or 'leonardo' in os.path.abspath(__file__):
        os.environ["WANDB_MODE"] = "offline"

    hyperparameters = {
        'clip_backbone': args.clip_backbone,
        'lora_r': args.lora_r,
        'epoches': args.epoches,
        'lr': args.lr,
        'wandb_activated': args.wandb_activated,
        'wandb_config': ast.literal_eval(args.wandb_config),
        'batch_size': args.bs,
        'device': args.device,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'initial_patience': args.initial_patience,
        'best_checkpoint_saving_root': args.best_checkpoint_saving_root,
        'last_best_state': args.last_best_state,
        'wandb_run_id': args.wandb_run_id,
        'debug': args.debug,
        'freeze_logit_scale': args.freeze_logit_scale,
        'mode': args.mode,
        'aperture_scale': args.aperture_scale,
        'entailment_loss_lambda': args.entailment_loss_lambda
    }
    
    clip_backbone = hyperparameters['clip_backbone']
    lora_r = hyperparameters['lora_r']
    epoches = hyperparameters['epoches']
    lr = hyperparameters['lr']
    wandb_activated = hyperparameters['wandb_activated']
    wandb_config = hyperparameters['wandb_config']
    wandb_run_id = hyperparameters['wandb_run_id']
    batch_size = hyperparameters['batch_size']
    device = hyperparameters['device']
    gradient_accumulation_steps = hyperparameters['gradient_accumulation_steps']
    initial_patience = hyperparameters['initial_patience']
    best_checkpoint_saving_root = hyperparameters['best_checkpoint_saving_root']
    resuming_wandb_run = False if wandb_run_id == 'None' else True
    debug = hyperparameters['debug']
    freeze_logit_scale = hyperparameters['freeze_logit_scale']
    mode = hyperparameters['mode']
    aperture_scale = hyperparameters['aperture_scale']
    entailment_loss_lambda = hyperparameters['entailment_loss_lambda']

    if wandb_activated and rank == 0:
        if not resuming_wandb_run:
            run = wandb.init(
                # settings=wandb.Settings(start_method="fork"),
                reinit=True, config=hyperparameters, **wandb_config
            )
        else:
            run = wandb.init(
                # settings=wandb.Settings(start_method="fork"),
                reinit=True, config=hyperparameters, **wandb_config,
                resume='must', id=wandb_run_id
            )
        print(f'Wandb ID: {run.id}')
        wandb_logger = WandbLogger(run)
    else:
        wandb_logger = None
        run=None

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=1,
        target_modules=["k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"],
        lora_dropout=0.1,
        bias="none",
    )

    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)

    text_encoder_original = CLIPTextModelWithProjection.from_pretrained(clip_backbone)
    text_encoder_lora = get_peft_model(text_encoder_original, peft_config)

    vision_encoder_original = CLIPVisionModelWithProjection.from_pretrained(clip_backbone)
    vision_encoder_lora = get_peft_model(vision_encoder_original, peft_config)
    
    lambdas = torch.tensor([1., 1., 1., 1., entailment_loss_lambda, entailment_loss_lambda, entailment_loss_lambda])

    model = HySAC(visual=vision_encoder_lora, textual=text_encoder_lora, freeze_logit_scale=freeze_logit_scale).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = model.module

    visu_root = "<VISU_ROOT>" # this dataset is not public available (only textual part is public) 
    coco_root = "<COCO_ROOT>"

    training_dataset = ViSU(root=visu_root, coco_root=coco_root, split='train', clip_backbone=clip_backbone)
    validation_dataset = ViSU(root=visu_root, coco_root=coco_root, split='validation', clip_backbone=clip_backbone)

    training_sampler = DistributedSampler(training_dataset)
    validation_sampler = DistributedSampler(validation_dataset)
    contrastive_loss_function = LorentzianCLIPContrastive()

    lambdas = lambdas / lambdas.sum()

    if rank == 0:
        job_id = os.environ.get("SLURM_JOB_ID")
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        proc_id = os.environ.get("SLURM_PROCID")
        pid = os.getpid()
        timestamp = math.floor(time.time())

        unique_dir = f"job_{job_id}_task_{task_id}_proc_{proc_id}_pid_{pid}_time_{timestamp}" 
        best_checkpoint_saving_root = os.path.join(best_checkpoint_saving_root, unique_dir)


        if not os.path.isdir(best_checkpoint_saving_root):
            try:
                os.makedirs(best_checkpoint_saving_root)
            except Exception as e:
                print(e)
        else:
            raise ValueError(f'{best_checkpoint_saving_root} already exists. Please remove it or change the best_checkpoint_saving_root.')

    _hyp = {k:str(v) for k,v in hyperparameters.items()}

    if rank == 0:
        with open(os.path.join(best_checkpoint_saving_root, 'config'), 'w') as f:
            json.dump(_hyp, f)

    best_checkpoint_saving_path = os.path.join(best_checkpoint_saving_root)
    if wandb_activated and run is not None:
        run.log({'best_checkpoint': best_checkpoint_saving_path})

    if rank == 0:
        print('Training...')
    training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        rank=rank,
        device=device,
        training_sampler=training_sampler,
        validation_sampler=validation_sampler,
        contrastive_loss_function=contrastive_loss_function,
        lambdas=lambdas,
        batch_size=batch_size,
        lr=lr,
        epoches=epoches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        initial_patience=initial_patience,
        wandb_activated=wandb_activated,
        run=run,
        best_checkpoint_saving_path=best_checkpoint_saving_path,
        clip_backbone=clip_backbone,
        debug=debug,
        wandb_logger=wandb_logger,
        mode=mode,
        aperture_scale=aperture_scale
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    main(args)