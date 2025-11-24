import os
import yaml
import json
import logging
import torch

from src.models import NanoConfig, Nano

logger = logging.getLogger(__name__)


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pth")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pth")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pth")
    model_data = torch.load(model_path, map_location=device)
    # load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pth")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data
