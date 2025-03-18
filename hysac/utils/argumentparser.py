import os

def parse_arguments(parser):
    parser.add_argument("--clip-backbone", type=str, help="Path to the CLIP backbone", default='openai/clip-vit-large-patch14')
    parser.add_argument("--lora-r", type=int, help="LoRa bottleneck dimension", default=16)
    parser.add_argument("--epoches", type=int, help="Epoches number", default=50)
    parser.add_argument("--lr", type=float, help="Learning rate value", default=1e-4)
    parser.add_argument("--wandb-activated", action="store_true", help="WandB enabled", default=False)
    parser.add_argument("--wandb-config", type=str, help="WandB configs [str but in dict shape]")
    parser.add_argument("--bs", type=int, help="Batch size", default=32)
    parser.add_argument("--device", type=str, help="Chosen device", choices=('cuda','cpu'), default='cuda')
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of gradient accumulation steps", default=1)
    parser.add_argument("--initial-patience", type=int, help="Initial patience value", default=5)
    parser.add_argument("--best-checkpoint-saving-root", type=str, help="Root where to save the best checkpoint")
    parser.add_argument("--last-best-state", type=str, help="Last best state", default='None')
    parser.add_argument("--wandb-run-id", type=str, help="WandB run ID to resume it", default='None')
    parser.add_argument("--debug", action="store_true", help="Debug mode", default=False)
    parser.add_argument("--freeze-logit-scale", action="store_true", help="Freeze logit scale", default=False)
    parser.add_argument("--mode", type=str, help="Mode to use", choices=('euc','hyp'), required=True)
    parser.add_argument("--aperture-scale", type=float, help="Aperture scale value", default=1.0)
    parser.add_argument("--entailment-loss-lambda", type=float, help="Entailment losses lambda value", default=1.0)
    args = parser.parse_args()
    return args
    
