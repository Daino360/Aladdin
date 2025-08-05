import os

# set the current working directory as the project root directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

from data.data import (
    DatasetOutputFormat,
    EnvironmentDataset,
    TransformsGenerator,
)


from models import construct_model
from training import Trainer

import torch.distributed as dist

from utils.utils import debug

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of params in total: ", pytorch_total_params)
    return total_params


def run(args):

    dataset_folder = (
        f"{args.train.dataset_root_dpath}/{args.train.dataset_name}/{args.train.dataset_name}"
    )
    cache_dpath = args.train.wandb_dpath if args.train.wandb_dpath != "./wandb" else "./cache"

    model = construct_model(args)

    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)
    train_ds = EnvironmentDataset(
        dataset_folder,
        seq_length_input=args.train.num_frames - 1,
        seq_step=1,
        split_type="instance",
        split="train",
        transform=transforms["train"],
        format=DatasetOutputFormat.IVG,
        enable_cache=True,
        cache_dpath=f"{cache_dpath}/cache/{args.train.dataset_name}",
    )

    valid_ds = EnvironmentDataset(
        dataset_folder,
        seq_length_input=args.train.num_frames - 1,
        seq_step=args.train.seq_step,
        split_type="instance",
        split="validation",
        transform=transforms["train"],
        format=DatasetOutputFormat.IVG,
        enable_cache=True,
        cache_dpath=f"{cache_dpath}/cache/{args.train.dataset_name}",
    )
    
    save_dpath = f"{args.train.save_root_dpath}/{args.model}/{args.train.wandb_name}"

    trainer = Trainer(
        model,
        args.train.batch_size,
        (train_ds, valid_ds),
        num_frames=args.train.num_frames,
        sample_num_frames=args.train.sample_num_frames,
        wandb_mode=args.train.wandb_mode,
        wandb_project=args.train.wandb_project,
        wandb_name=args.train.wandb_name,
        grad_accum_every=args.train.grad_accum,  # use this as a multiplier of the batch size
        num_train_steps=args.train.num_train_steps,
        lr=args.optimizer.learning_rate,  # Learning rate
        wd=args.optimizer.weight_decay,  # Weight decay
        linear_warmup_start_factor=args.optimizer.linear_warmup_start_factor,
        linear_warmup_total_iters=args.optimizer.linear_warmup_total_iters,
        cosine_annealing_T_max=args.optimizer.cosine_annealing_t_max,
        cosine_annealing_eta_min=args.optimizer.cosine_annealing_min_lr,
        max_grad_norm=args.optimizer.max_grad_norm,  # gradient clipping
        save_dpath=save_dpath,
        save_model_every=args.train.save_model_every,
        wandb_dpath=args.train.wandb_dpath,
        validate_every=args.train.validate_every,
        train_config=args,
    )

    torch.cuda.empty_cache()

    if trainer.accelerator.is_main_process:

        debug("config: ", args)
        print("GenieRedux training is starting...\n")
        print("Dataset : ", dataset_folder)
        count_parameters(model)

    if args.train.resume_ckpt != "no":
        trainer.load(args.train.resume_ckpt)
        print("Model loaded from file:", args.train.resume_ckpt)
        print("Starting training from step:", trainer.step)

    model.train()
    trainer.train()

    dist.destroy_process_group()
