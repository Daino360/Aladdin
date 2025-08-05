import torch

from models import (
    Tokenizer,
    LatentActionModel,
    MaskGit,
    Dynamics,
    GenieReduxGuided,
    GenieRedux,
)


def construct_model(config):
    
    if config.model not in ["tokenizer", "genie_redux", "genie_redux_guided"]:
        raise ValueError(f"Unknown model: {config.model}")
    
    tokenizer = Tokenizer(
        dim=config.tokenizer.dim,
        codebook_size=config.tokenizer.codebook_size,
        image_size=config.tokenizer.image_size,
        patch_size=config.tokenizer.patch_size,
        wandb_mode=config.train.wandb_mode,
        temporal_patch_size=config.tokenizer.temporal_patch_size,  # temporal patch size
        num_blocks=config.tokenizer.num_blocks,  # nb of blocks in st transformer
        dim_head=config.tokenizer.dim_head,  # hidden size in transfo
        heads=config.tokenizer.heads,  # nb of heads for multi head transfo
        ff_mult=config.tokenizer.ff_mult,  # 32 * 64 = 2048 MLP size in transfo out
        vq_loss_w=config.tokenizer.vq_loss_weight,  # commit loss weight
        recon_loss_w=config.tokenizer.recons_loss_weight,  # reconstruction loss weight
    )

    if config.model == "tokenizer":
        return tokenizer

    tokenizer_state_dict = torch.load(
        config.tokenizer_fpath, map_location=torch.device("cpu")
    )
    tokenizer.load_state_dict(tokenizer_state_dict["model"])
    del tokenizer_state_dict

    is_guided = config.model == "genie_redux_guided"

    maskgit = MaskGit(
        dim=config.dynamics.dim,
        is_guided=is_guided,
        action_dim=config.dynamics.action_dim,
        num_tokens=config.tokenizer.codebook_size,
        heads=config.dynamics.heads,
        dim_head=config.dynamics.dim_head,
        num_blocks=config.dynamics.num_blocks,
        max_seq_len=config.dynamics.max_seq_len,
        image_size=config.dynamics.image_size,
        patch_size=config.dynamics.patch_size,
    )

    dynamics = Dynamics(
        maskgit=maskgit,
        inference_steps=1,
        sample_temperature=config.dynamics.sample_temperature,
        mask_schedule="cosine",
    )

    if is_guided:
        return GenieReduxGuided(tokenizer, dynamics)

    latent_action_model = LatentActionModel(
        dim=config.lam.dim,
        codebook_size=config.lam.codebook_size,
        image_size=config.lam.image_size,
        patch_size=config.lam.patch_size,
        wandb_mode=config.train.wandb_mode,
        temporal_patch_size=config.lam.temporal_patch_size,  # temporal patch size
        num_blocks=config.lam.num_blocks,  # nb of blocks in st transformer
        dim_head=config.lam.dim_head,  # hidden size in transfo
        heads=config.lam.heads,  # nb of heads for multi head transfo
        ff_mult=config.lam.ff_mult,  # 32 * 64 = 2048 MLP size in transfo out
        vq_loss_w=config.lam.vq_loss_weight,  # commit loss weight
        recon_loss_w=config.lam.recons_loss_weight,  # reconstruction loss weight
    )

    return GenieRedux(tokenizer, latent_action_model, dynamics)
